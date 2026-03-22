#!/usr/bin/env python3
"""
One-time import of Skyrim seed data into Qdrant.

Populates two collections:

  skyrim_lore  (single unnamed semantic vector — matches existing schema)
    Source: shared/data/oghma_infinium.sql
    238 Skyrim lore entries: topic slug + full description.
    Queried at session start to build the static world knowledge string.

  skyrim_npc_profiles  (named "semantic" vector — new collection)
    Sources: shared/data/npc_templates_20250302001.sql
             shared/data/personalities_json_11_18_2024.csv
    Merged on slug. One point per NPC with biography and structured
    personality data where both exist (~1,300 NPCs total).
    Queried on addnpc to bootstrap agent identity kernel.

Run from repo root on the Beelink:
    python scripts/import_seed_data.py
    python scripts/import_seed_data.py --qdrant-host 192.168.0.13

Defaults QDRANT_HOST env → 127.0.0.1. Safe to re-run: all upserts use
deterministic UUIDs keyed on slug/topic. Re-running overwrites with the
same data.

Requires: sentence-transformers, qdrant-client (both in .venv).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from uuid import uuid5, NAMESPACE_DNS

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / "shared" / "data"

OGHMA_SQL        = DATA_DIR / "oghma_infinium.sql"
TEMPLATES_SQL    = DATA_DIR / "npc_templates_20250302001.sql"
PERSONALITIES_CSV = DATA_DIR / "personalities_json_11_18_2024.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLLECTION_LORE         = "skyrim_lore"
COLLECTION_NPC_PROFILES = "skyrim_npc_profiles"
MODEL_NAME   = "all-MiniLM-L6-v2"
SEMANTIC_DIM = 384
BATCH_SIZE   = 64


def stable_id(namespace: str, key: str) -> str:
    """Deterministic UUID — re-running imports overwrites the same points."""
    return str(uuid5(NAMESPACE_DNS, f"mmk:{namespace}:{key}"))


# ---------------------------------------------------------------------------
# PostgreSQL INSERT parser
# ---------------------------------------------------------------------------

def parse_pg_inserts(sql_text: str, table_name: str) -> list[list[str]]:
    """Extract row tuples from PostgreSQL INSERT INTO <table> VALUES (...);

    Handles:
    - Multi-line text values spanning many lines
    - Single-quote escaping via '' (two consecutive single quotes → one)
    - NULL tokens (treated as empty string)
    - Whitespace between column values

    Returns list of rows, each row a list of column strings.
    """
    results: list[list[str]] = []
    pattern = re.compile(
        rf"INSERT INTO\s+\S*{re.escape(table_name)}\s+VALUES\s*\(",
        re.IGNORECASE,
    )
    for match in pattern.finditer(sql_text):
        i = match.end()
        cols: list[str] = []
        buf:  list[str] = []
        in_str = False

        while i < len(sql_text):
            c = sql_text[i]
            if in_str:
                if c == "'" and i + 1 < len(sql_text) and sql_text[i + 1] == "'":
                    # Escaped single quote
                    buf.append("'")
                    i += 2
                    continue
                elif c == "'":
                    in_str = False
                else:
                    buf.append(c)
            else:
                if c == "'":
                    in_str = True
                elif c == ",":
                    cols.append("".join(buf))
                    buf = []
                elif c == ")":
                    cols.append("".join(buf))
                    results.append(cols)
                    break
                elif sql_text[i:i + 4] == "NULL":
                    # NULL → empty string; skip the 4 characters
                    cols.append("")
                    buf = []
                    i += 4
                    continue
                # else: whitespace between column tokens — skip
            i += 1

    return results


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_lore() -> list[dict]:
    """Parse oghma_infinium.sql → [{topic, content}, ...]."""
    sql = OGHMA_SQL.read_text(encoding="utf-8", errors="replace")
    rows = parse_pg_inserts(sql, "oghma")
    entries = [
        {"topic": r[0], "content": r[1]}
        for r in rows
        if len(r) >= 2 and r[0] and r[1]
    ]
    logger.info("Loaded %d lore entries", len(entries))
    return entries


def load_npc_templates() -> dict[str, dict]:
    """Parse npc_templates SQL → {slug: {bio_text, tags, voice_type}}."""
    sql = TEMPLATES_SQL.read_text(encoding="utf-8", errors="replace")
    rows = parse_pg_inserts(sql, "npc_templates")
    templates: dict[str, dict] = {}
    for row in rows:
        if not row or not row[0]:
            continue
        slug      = row[0]
        bio       = row[1] if len(row) > 1 else ""
        tags_raw  = row[2] if len(row) > 2 else ""
        voice     = row[5] if len(row) > 5 else ""
        tags      = [t.strip() for t in tags_raw.split(",") if t.strip()]
        templates[slug] = {"bio_text": bio, "tags": tags, "voice_type": voice}
    logger.info("Loaded %d NPC templates", len(templates))
    return templates


def load_personalities() -> dict[str, dict]:
    """Parse personalities CSV → {slug: personality_dict}."""
    personalities: dict[str, dict] = {}
    with PERSONALITIES_CSV.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug     = (row.get("npc_name") or "").strip()
            raw_json = (row.get("personality") or "").strip()
            if not slug or not raw_json:
                continue
            try:
                personalities[slug] = json.loads(raw_json)
            except json.JSONDecodeError:
                logger.debug("Skipping bad JSON for personality slug=%s", slug)
    logger.info("Loaded %d personality records", len(personalities))
    return personalities


def merge_npc_data(
    templates: dict[str, dict],
    personalities: dict[str, dict],
) -> list[dict]:
    """Merge templates and personalities on slug. One record per unique NPC."""
    all_slugs = sorted(set(templates) | set(personalities))
    records = []
    for slug in all_slugs:
        t = templates.get(slug, {})
        p = personalities.get(slug)
        bio = t.get("bio_text", "")
        # Primary embed text: biography. Fallback: backgroundSummary from
        # personality JSON. Final fallback: slug itself (ensures no empty embed).
        embed_text = (
            bio
            or (p.get("backgroundSummary", "") if p else "")
            or slug
        )
        records.append({
            "slug":            slug,
            "bio_text":        bio,
            "tags":            t.get("tags", []),
            "voice_type":      t.get("voice_type", ""),
            "personality":     p,
            "has_personality": p is not None,
            "embed_text":      embed_text,
        })
    logger.info(
        "Merged %d unique NPC profiles (%d from templates, %d from personalities)",
        len(records), len(templates), len(personalities),
    )
    return records


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def _lore_vector_format(client: QdrantClient) -> str:
    """Detect whether skyrim_lore uses named or unnamed vectors.

    The collection may have been created by an older version of the code
    (unnamed single vector) or by the current ensure_collections()
    (named 'semantic'). We probe the config to write in the right format.
    """
    try:
        info = client.get_collection(COLLECTION_LORE)
        vc = info.config.params.vectors
        # Named vectors: a dict mapping name → VectorParams
        # Unnamed: a single VectorParams object
        return "named" if isinstance(vc, dict) else "unnamed"
    except Exception:
        return "named"  # Will be created fresh; use named


def ensure_npc_profiles_collection(client: QdrantClient) -> None:
    """Create skyrim_npc_profiles with named semantic vector if absent."""
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NPC_PROFILES in existing:
        logger.info("Collection already exists: %s", COLLECTION_NPC_PROFILES)
        return
    client.create_collection(
        collection_name=COLLECTION_NPC_PROFILES,
        vectors_config={
            "semantic": VectorParams(size=SEMANTIC_DIM, distance=Distance.COSINE),
        },
    )
    # Fast slug lookup for bootstrap queries on addnpc
    client.create_payload_index(
        COLLECTION_NPC_PROFILES, field_name="slug", field_schema="keyword",
    )
    client.create_payload_index(
        COLLECTION_NPC_PROFILES, field_name="has_personality", field_schema="bool",
    )
    logger.info("Created collection: %s", COLLECTION_NPC_PROFILES)


def ensure_lore_collection(client: QdrantClient) -> None:
    """Create skyrim_lore with named semantic vector if absent."""
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_LORE not in existing:
        client.create_collection(
            collection_name=COLLECTION_LORE,
            vectors_config={
                "semantic": VectorParams(size=SEMANTIC_DIM, distance=Distance.COSINE),
            },
        )
        logger.info("Created collection: %s", COLLECTION_LORE)


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------

def upsert_lore(
    client: QdrantClient,
    model: SentenceTransformer,
    entries: list[dict],
) -> int:
    """Embed and upsert lore entries. Handles both named and unnamed vector schemas."""
    vector_format = _lore_vector_format(client)
    logger.info("skyrim_lore vector format: %s", vector_format)

    total = 0
    for i in range(0, len(entries), BATCH_SIZE):
        batch = entries[i:i + BATCH_SIZE]
        # Embed topic + description together for better semantic retrieval
        texts = [f"{e['topic']}: {e['content']}" for e in batch]
        vecs  = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        points = []
        for entry, vec in zip(batch, vecs):
            v = {"semantic": vec.tolist()} if vector_format == "named" else vec.tolist()
            points.append(PointStruct(
                id=stable_id("lore", entry["topic"]),
                vector=v,
                payload={"topic": entry["topic"], "content": entry["content"]},
            ))
        client.upsert(collection_name=COLLECTION_LORE, points=points)
        total += len(batch)
        logger.info("  Lore:     %d / %d", total, len(entries))
    return total


def upsert_npc_profiles(
    client: QdrantClient,
    model: SentenceTransformer,
    records: list[dict],
) -> int:
    """Embed and upsert NPC profiles. Always uses named 'semantic' vector."""
    total = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        texts = [r["embed_text"] for r in batch]
        vecs  = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        points = []
        for rec, vec in zip(batch, vecs):
            payload: dict = {
                "slug":            rec["slug"],
                "bio_text":        rec["bio_text"],
                "tags":            rec["tags"],
                "voice_type":      rec["voice_type"],
                "has_personality": rec["has_personality"],
            }
            # Store full personality JSON when available — feeds identity kernel
            # and relationship seeding on addnpc bootstrap.
            if rec["personality"]:
                payload["personality"] = rec["personality"]
            points.append(PointStruct(
                id=stable_id("npc_profile", rec["slug"]),
                vector={"semantic": vec.tolist()},
                payload=payload,
            ))
        client.upsert(collection_name=COLLECTION_NPC_PROFILES, points=points)
        total += len(batch)
        logger.info("  Profiles: %d / %d", total, len(records))
    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Skyrim seed data (lore + NPC profiles) into Qdrant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--qdrant-host",
        default=os.environ.get("QDRANT_HOST", "127.0.0.1"),
        help="Qdrant host. Default: QDRANT_HOST env or 127.0.0.1",
    )
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument(
        "--lore-only",     action="store_true", help="Import only lore (skip NPC profiles)",
    )
    parser.add_argument(
        "--profiles-only", action="store_true", help="Import only NPC profiles (skip lore)",
    )
    args = parser.parse_args()

    logger.info("Connecting to Qdrant at %s:%d", args.qdrant_host, args.qdrant_port)
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)

    logger.info("Loading %s on CPU...", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    logger.info("Model ready — starting import")

    if not args.profiles_only:
        ensure_lore_collection(client)
        entries = load_lore()
        if entries:
            n = upsert_lore(client, model, entries)
            logger.info("Lore import complete: %d entries written to %s", n, COLLECTION_LORE)

    if not args.lore_only:
        templates    = load_npc_templates()
        personalities = load_personalities()
        records      = merge_npc_data(templates, personalities)
        ensure_npc_profiles_collection(client)
        n = upsert_npc_profiles(client, model, records)
        logger.info("NPC import complete: %d profiles written to %s", n, COLLECTION_NPC_PROFILES)

    logger.info("Done.")


if __name__ == "__main__":
    main()
