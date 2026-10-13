# Cabrera's gate finder against datanuggets.flow, piece by piece. Same arguments as compare.py:
#   .\benchmarks\flow\compare.ps1 C:\data\202.fcs
#   .\benchmarks\flow\compare.ps1 C:\data\pbmc --repeat 5
#   .\benchmarks\flow\compare.ps1 -Mini ~/flowdata/zenodo-22079206 --repeat 5   runs on the Mac mini
# Prints the table, appends tmp\flow\summary.csv (tmp\flow\mini\ for a mini run, copied back) and
# opens the newest picture. Exits with compare.py's status: 1 if a piece that must match doesn't.
param(
    [switch]$Mini,
    [Parameter(Mandatory = $true, Position = 0)][string]$Target,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest = @()
)
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$out = Join-Path $repo 'tmp\flow'
$env:PYTHONIOENCODING = 'utf-8'
$env:MPLBACKEND = 'Agg'
if ($Mini) {
    $out = Join-Path $out 'mini'
    New-Item -ItemType Directory -Force $out | Out-Null
    $bash = 'C:\Program Files\Git\bin\bash.exe'
    # single quotes so Git Bash leaves a ~ for the mini to expand
    $quoted = ($Rest | ForEach-Object { "'$_'" }) -join ' '
    Push-Location $repo
    try {
        & $bash -c "~/.claude/tools/mini-run.sh uv run python benchmarks/flow/compare.py '$Target' --out tmp/flow $quoted"
        $code = $LASTEXITCODE
        scp -q 'mini:datanuggets/tmp/flow/*.png' 'mini:datanuggets/tmp/flow/*.csv' $out
    } finally { Pop-Location }
} else {
    New-Item -ItemType Directory -Force $out | Out-Null
    $python = Join-Path $repo '.venv\Scripts\python.exe'
    & $python (Join-Path $PSScriptRoot 'compare.py') $Target --out $out @Rest
    $code = $LASTEXITCODE
}
Get-ChildItem $out -Filter '*_compare.png' | Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 | Invoke-Item
exit $code
