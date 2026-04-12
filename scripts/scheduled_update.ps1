$venv = "C:\Users\mmanj\OneDrive\Documents\manju\CursorPractice\.venv\Scripts\python.exe"
$project_dir = "c:\Users\mmanj\OneDrive\Documents\manju\CursorPractice\portfolio_analyzer"
$script = "$project_dir\scripts\update_db_all_nse.py"

Write-Host "=== Scheduled Task: Portfolio Analyzer Update ==="
Set-Location $project_dir

Write-Host "[1/2] Updating Local Stocks..."
& $venv $script

Write-Host "`n[2/2] Updating Production Stocks (Neon)..."
$env:DATABASE_URL="postgresql://neondb_owner:npg_EQTd1DNPCY9m@ep-steep-snow-a1a4wgoe.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
& $venv $script

Write-Host "`n✅ Scheduled Update Complete."
