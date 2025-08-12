Write-Host "Starting Mushroom Classification API Server..." -ForegroundColor Green
Write-Host ""
Write-Host "Make sure you have installed all dependencies:" -ForegroundColor Yellow
Write-Host "pip install -r requirements.txt" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting server on http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

try {
    python app.py
}
catch {
    Write-Host "Error starting server: $_" -ForegroundColor Red
    Write-Host "Make sure Python is installed and in your PATH" -ForegroundColor Yellow
}

Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

