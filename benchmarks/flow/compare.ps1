# Cabrera's gate finder against datanuggets.flow, piece by piece. Same arguments as compare.py:
#   .\benchmarks\flow\compare.ps1 C:\data\202.fcs
#   .\benchmarks\flow\compare.ps1 C:\data\pbmc --repeat 5
#   .\benchmarks\flow\compare.ps1 --metadata C:\data\study.xlsx --dir C:\data\pbmc --t0 22,20,18,18
# Prints the table, appends tmp\flow\summary.csv and opens the newest picture. Exits with
# compare.py's status: 1 if a piece that must match doesn't.
# The arguments go to compare.py as they are, so the file or folder isn't a parameter of this
# script: a positional one would take --metadata as the file.
param(
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest = @()
)
if ($Rest -like '--out*') {
    Write-Error 'compare.ps1 writes to tmp\flow itself; run compare.py for another --out'
    exit 2
}
$start = Get-Date
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$out = Join-Path $repo 'tmp\flow'
$env:PYTHONIOENCODING = 'utf-8'
$env:MPLBACKEND = 'Agg'
New-Item -ItemType Directory -Force $out | Out-Null
$python = Join-Path $repo '.venv\Scripts\python.exe'
& $python (Join-Path $PSScriptRoot 'compare.py') @Rest --out $out
$code = $LASTEXITCODE
# only a picture this run wrote, so --no-png opens nothing
Get-ChildItem $out -Filter '*_compare.png' | Where-Object LastWriteTime -ge $start |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Invoke-Item
exit $code
