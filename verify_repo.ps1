param(
  [string]$RepoPath = "C:\Users\Pavlos\Desktop\trading-backtester"
)

Set-Location $RepoPath

Write-Host "== Repo status ==" -ForegroundColor Cyan
git status

Write-Host "`n== Branch ==" -ForegroundColor Cyan
git branch --show-current

Write-Host "`n== Remote ==" -ForegroundColor Cyan
git remote -v

Write-Host "`n== Tracked files ==" -ForegroundColor Cyan
git ls-files

Write-Host "`n== Ignored output check ==" -ForegroundColor Cyan
$ignored = @("market_data.csv","orders_audit.csv")
foreach($f in $ignored){
  if(Test-Path $f){
    $check = git check-ignore -v $f 2>$null
    if($LASTEXITCODE -eq 0){
      Write-Host "OK ignored: $f" -ForegroundColor Green
    } else {
      Write-Host "NOT ignored (would be tracked): $f" -ForegroundColor Yellow
    }
  } else {
    Write-Host "Not present: $f" -ForegroundColor DarkGray
  }
}

Write-Host "`n== Self-test run ==" -ForegroundColor Cyan
python self_test.py
