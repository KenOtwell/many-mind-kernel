# Dump human-readable string runs from a binary file (ASCII + UTF-16 LE).
#
# Used to inspect closed-source SKSE plugins like AIAgent.dll for hardcoded
# audio paths, HTTP endpoint names, configuration keys, error messages,
# and other interesting tokens that hint at the runtime contract we can't
# see in source.
#
# Outputs two files alongside the input binary:
#   <name>.strings.txt          — full dump (deduplicated, encoding-tagged)
#   <name>.strings.paths.txt    — filtered to path-like / endpoint-like strings
#
# Re-runnable: safe to invoke after a DLL update; overwrites previous dumps.

[CmdletBinding()]
param(
    [string]$BinaryPath = "C:\Modlists\PandasSovngarde\mods\AIAgent\SKSE\Plugins\AIAgent.dll",
    [string]$OutputDir = "C:\Users\Ken\Projects\many-mind-kernel\docs\dll-strings",
    [int]$MinLength = 4
)

if (-not (Test-Path $BinaryPath)) {
    Write-Error "Binary not found: $BinaryPath"
    exit 1
}

$file = Get-Item $BinaryPath
Write-Host "Reading $($file.Name) ($([int]($file.Length/1KB)) KB)..."
$bytes = [System.IO.File]::ReadAllBytes($BinaryPath)

# Treat the byte stream as Latin-1 so each byte maps 1:1 to a char.
# That lets us run regexes over the binary content efficiently.
$enc = [System.Text.Encoding]::GetEncoding("iso-8859-1")
$text = $enc.GetString($bytes)

# ASCII strings: runs of printable chars (space..~) of MinLength or more.
Write-Host "Scanning ASCII strings (min length $MinLength)..."
$asciiPattern = "[\x20-\x7E]{$MinLength,}"
$asciiMatches = [regex]::Matches($text, $asciiPattern) | ForEach-Object { $_.Value }

# UTF-16 LE strings: alternating printable byte + 0x00. Strip the nulls.
Write-Host "Scanning UTF-16 LE strings (min length $MinLength)..."
$utf16Pattern = "(?:[\x20-\x7E]\x00){$MinLength,}"
$utf16Matches = [regex]::Matches($text, $utf16Pattern) | ForEach-Object {
    $_.Value -replace "`0", ""
}

Write-Host ("ASCII: {0:N0} matches  UTF-16: {1:N0} matches" -f $asciiMatches.Count, $utf16Matches.Count)

# Dedupe within each encoding, tag with origin, then combine.
$asciiUnique = $asciiMatches | Sort-Object -Unique
$utf16Unique = $utf16Matches | Sort-Object -Unique

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$base = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
$fullPath  = Join-Path $OutputDir "$base.strings.txt"
$pathsPath = Join-Path $OutputDir "$base.strings.paths.txt"

# Full dump with encoding sections.
$dumpLines = @()
$dumpLines += "# Strings extracted from $($file.Name)"
$dumpLines += "# Source: $BinaryPath"
$dumpLines += "# Size:   $($file.Length) bytes  ($([int]($file.Length/1KB)) KB)"
$dumpLines += "# LastWriteTime: $($file.LastWriteTime)"
$dumpLines += "# Min run length: $MinLength chars"
$dumpLines += "# Extracted: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$dumpLines += "# ASCII unique: $($asciiUnique.Count) | UTF-16 unique: $($utf16Unique.Count)"
$dumpLines += ""
$dumpLines += "## ASCII"
$dumpLines += $asciiUnique
$dumpLines += ""
$dumpLines += "## UTF-16 LE"
$dumpLines += $utf16Unique

Set-Content -Path $fullPath -Value $dumpLines -Encoding UTF8
Write-Host "Wrote full dump: $fullPath"

# Filtered "interesting paths and endpoints" subset.
# Picks anything looking like a path, URL, audio extension, PHP endpoint,
# or config-key-shaped token. Tuned for SKSE plugins.
$interestingPattern = '(?i)(\.wav|\.fuz|\.lip|\.bsa|\.esp|\.esm|\.pex|\.json|\.ini|\.dds|\.nif|\.php|\.html|\.css|\.js|http|https|sound[\\/]|voice[\\/]|aiagent|setconf|setvoice|getvoice|playsound|playvoice|playaudio|stt\.|vsx\.|comm\.|stream|herika|chim|openmic|tts|asr|whisper|recordsound|sendallvoices)'

$allUnique = ($asciiUnique + $utf16Unique) | Sort-Object -Unique
$interesting = $allUnique | Where-Object { $_ -match $interestingPattern }

$pathLines = @()
$pathLines += "# Path / endpoint / audio-related strings from $($file.Name)"
$pathLines += "# Filtered from $($allUnique.Count) unique strings -> $($interesting.Count) matches"
$pathLines += "# Pattern: $interestingPattern"
$pathLines += "# Extracted: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$pathLines += ""
$pathLines += $interesting

Set-Content -Path $pathsPath -Value $pathLines -Encoding UTF8
Write-Host "Wrote filtered paths: $pathsPath"
Write-Host ""
Write-Host "Filtered hit count: $($interesting.Count)"
Write-Host "Review: $pathsPath"
