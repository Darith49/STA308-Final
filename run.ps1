# PowerShell helper script for Course Recommendation System
param (
    [string]$Target = "app"
)

switch ($Target.ToLower()) {
    "train" {
        Write-Host "Training models and generating artifacts..." -ForegroundColor Cyan
        python src/train.py
    }
    "test" {
        Write-Host "Running pytest validation suite..." -ForegroundColor Cyan
        pytest -v
    }
    "app" {
        Write-Host "Starting Streamlit web application..." -ForegroundColor Green
        streamlit run app/app.py
    }
    default {
        Write-Host "Usage: .\run.ps1 [train | test | app]" -ForegroundColor Yellow
    }
}
