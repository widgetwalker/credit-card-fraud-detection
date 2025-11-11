# GitHub Repository Setup Script
# Run this script after creating the repository on GitHub

param(
    [Parameter(Mandatory=$true)]
    [string]$GitHubUsername
)

$RepoName = "credit-card-fraud-detection"
$RepoUrl = "https://github.com/$GitHubUsername/$RepoName.git"

Write-Host "Setting up GitHub repository..." -ForegroundColor Green
Write-Host "Repository URL: $RepoUrl" -ForegroundColor Cyan

# Check if remote already exists
$remoteExists = git remote get-url origin 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Remote 'origin' already exists. Removing..." -ForegroundColor Yellow
    git remote remove origin
}

# Add remote
Write-Host "Adding remote repository..." -ForegroundColor Green
git remote add origin $RepoUrl

# Verify remote
Write-Host "Verifying remote..." -ForegroundColor Green
git remote -v

# Push to GitHub
Write-Host "Pushing to GitHub..." -ForegroundColor Green
Write-Host "You may be prompted for GitHub credentials." -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "Success! Repository pushed to GitHub." -ForegroundColor Green
    Write-Host "Repository URL: https://github.com/$GitHubUsername/$RepoName" -ForegroundColor Cyan
} else {
    Write-Host "Error pushing to GitHub. Please check your credentials and try again." -ForegroundColor Red
    Write-Host "You can also manually run: git push -u origin main" -ForegroundColor Yellow
}

