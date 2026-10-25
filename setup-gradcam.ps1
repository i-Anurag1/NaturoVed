$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

$report = [System.Collections.Generic.List[string]]::new()
function Check($name, $ok, $detail) {
    $status = if ($ok) { "OK" } else { "NEEDS ATTENTION" }
    $line = "[$status] $name - $detail"
    Write-Host $line -ForegroundColor $(if ($ok) { "Green" } else { "Yellow" })
    $script:report.Add($line)
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " NATUROVED - GRAD-CAM FULL DIAGNOSTIC" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 1. Project structure
$backend = Join-Path $repo "backend"
$frontend = Join-Path $repo "frontend"
$models = Join-Path $backend "storage\models"
$raw = Join-Path $repo "data\raw"
$processed = Join-Path $repo "data\processed"

Check "Backend folder" (Test-Path $backend) $backend
Check "Frontend folder" (Test-Path $frontend) $frontend
Check "Git repository" (Test-Path (Join-Path $repo ".git")) "ZIP extraction is not a Git repository; clone or initialize later"
Check "Disease model" (Test-Path (Join-Path $models "disease_model.keras")) "Expected trained Keras model"
Check "Model label map" (Test-Path (Join-Path $models "label_map.json")) "Expected model label mapping"
Check "Raw dataset" (Test-Path $raw) $raw

# 2. Python and dependencies
Write-Host "`n=== Python and ML dependencies ===" -ForegroundColor Cyan
python --version
if ($LASTEXITCODE -ne 0) { Check "Python" $false "Python command failed" }

python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU devices:', tf.config.list_physical_devices('GPU'))"
Check "TensorFlow import" ($LASTEXITCODE -eq 0) "Import test"

python -c "import fastapi, uvicorn, numpy, PIL, cv2; print('Backend core imports OK')"
Check "Backend imports" ($LASTEXITCODE -eq 0) "FastAPI, Uvicorn, NumPy, Pillow, OpenCV"

# 3. Read authoritative class list from labels.py
$labelsFile = Join-Path $backend "app\ml\labels.py"
$classes = @()
if (Test-Path $labelsFile) {
    $labelText = Get-Content $labelsFile -Raw
    $classes = [regex]::Matches($labelText, '"([A-Za-z]+___[A-Za-z_]+)"') |
        ForEach-Object { $_.Groups[1].Value } |
        Select-Object -Unique
}
Check "Class labels parsed" ($classes.Count -gt 0) "$($classes.Count) classes found in labels.py"

# 4. Count source images per class
$imageExtensions = @(".jpg", ".jpeg", ".png", ".bmp", ".webp")
$present = @()
$missing = @()
Write-Host "`n=== Raw dataset class coverage ===" -ForegroundColor Cyan

if (Test-Path $raw) {
    foreach ($class in $classes) {
        $folder = Join-Path $raw $class
        if (Test-Path $folder) {
            $count = @(Get-ChildItem $folder -File -Recurse |
                Where-Object { $imageExtensions -contains $_.Extension.ToLower() }).Count
            if ($count -gt 0) {
                $present += $class
                Write-Host ("[FOUND] {0}: {1} images" -f $class, $count) -ForegroundColor Green
            } else {
                $missing += $class
                Write-Host ("[EMPTY] {0}: no supported images" -f $class) -ForegroundColor Yellow
            }
        } else {
            $missing += $class
            Write-Host ("[MISSING] {0}" -f $class) -ForegroundColor Yellow
        }
    }
} else {
    $missing = $classes
    Write-Host "No data\raw folder. Dataset download/extraction is required." -ForegroundColor Yellow
}

Check "Raw class coverage" ($classes.Count -gt 0 -and $missing.Count -eq 0) "$($present.Count)/$($classes.Count) classes have images"

# 5. Check prepared split coverage
$splitReady = $true
foreach ($split in @("train", "val", "test")) {
    $splitDir = Join-Path $processed $split
    if (!(Test-Path $splitDir)) {
        $splitReady = $false
        continue
    }
    foreach ($class in $classes) {
        $classDir = Join-Path $splitDir $class
        if (!(Test-Path $classDir)) {
            $splitReady = $false
            continue
        }
        $count = @(Get-ChildItem $classDir -File -Recurse |
            Where-Object { $imageExtensions -contains $_.Extension.ToLower() }).Count
        if ($count -lt 1) { $splitReady = $false }
    }
}
Check "Prepared train/val/test splits" $splitReady "All configured classes must have images in all three splits"

# 6. Prepare dataset only if every required class is present.
if ($classes.Count -gt 0 -and $missing.Count -eq 0 -and !$splitReady) {
    Write-Host "`nPreparing class-preserving dataset splits..." -ForegroundColor Cyan
    $organizer = Join-Path $backend "app\ml\dataset_utils.py"
    if (Test-Path $organizer) {
        Push-Location $backend
        python $organizer --raw_dir "..\data\raw" --processed_dir "..\data\processed"
        $prepExit = $LASTEXITCODE
        Pop-Location
        Check "Dataset preparation" ($prepExit -eq 0) "Dataset organizer exit code: $prepExit"
    } else {
        Check "Dataset preparation" $false "dataset_utils.py not found"
    }
}

# 7. Training: discover supported CLI arguments before running.
$modelPath = Join-Path $models "disease_model.keras"
$trainer = Join-Path $backend "app\ml\train_transfer.py"
if ((Test-Path $trainer) -and !(Test-Path $modelPath)) {
    Write-Host "`n=== Training script CLI ===" -ForegroundColor Cyan
    Push-Location $backend
    python $trainer --help
    $helpExit = $LASTEXITCODE
    Pop-Location

    $allSplitsReady = $true
    foreach ($split in @("train", "val", "test")) {
        foreach ($class in $classes) {
            $dir = Join-Path (Join-Path $processed $split) $class
            if (!(Test-Path $dir)) { $allSplitsReady = $false; continue }
            $n = @(Get-ChildItem $dir -File -Recurse |
                Where-Object { $imageExtensions -contains $_.Extension.ToLower() }).Count
            if ($n -lt 1) { $allSplitsReady = $false }
        }
    }

    if ($allSplitsReady -and $helpExit -eq 0) {
        Write-Host "`nTraining is NOT launched automatically because CLI flags must match the trainer's actual help output. Review it, then run the exact command shown by --help." -ForegroundColor Yellow
        Write-Host "The dataset must be scientifically appropriate for every configured label; image counts alone do not verify label quality." -ForegroundColor Yellow
    } else {
        Check "Model training" $false "No valid trained model and/or complete prepared dataset. Obtain and verify the dataset first."
    }
} elseif (Test-Path $modelPath) {
    Check "Training skipped" $true "A disease_model.keras file already exists; compatibility and accuracy still need testing"
} else {
    Check "Training script" $false "train_transfer.py not found"
}

# 8. Grad-CAM source checks
$gradcam = Join-Path $backend "app\services\gradcam.py"
Check "Grad-CAM service file" (Test-Path $gradcam) $gradcam
if (Test-Path $gradcam) {
    Select-String -Path $gradcam -Pattern "GradientTape|heatmap|model|None" |
        Select-Object -First 20
}

# 9. Frontend configuration
Write-Host "`n=== Frontend configuration ===" -ForegroundColor Cyan
$packageJson = Join-Path $frontend "package.json"
Check "Frontend package.json" (Test-Path $packageJson) $packageJson
if (Test-Path $packageJson) {
    try {
        $pkg = Get-Content $packageJson -Raw | ConvertFrom-Json
        Write-Host "Frontend scripts:"
        $pkg.scripts | Format-List
    } catch {
        Write-Host "Could not parse frontend package.json: $_" -ForegroundColor Yellow
    }
}
Check "Node.js" ([bool](Get-Command node -ErrorAction SilentlyContinue)) "Node command available"
Check "npm" ([bool](Get-Command npm -ErrorAction SilentlyContinue)) "npm command available"

# 10. Environment files: report presence only, never print secrets.
Write-Host "`n=== Environment configuration ===" -ForegroundColor Cyan
foreach ($file in @(
    (Join-Path $backend ".env"),
    (Join-Path $frontend ".env"),
    (Join-Path $repo ".env")
)) {
    Write-Host ("{0}: {1}" -f $file, $(if (Test-Path $file) { "present (contents hidden)" } else { "not found" }))
}

# 11. Summary and safe launch
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " DIAGNOSTIC SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
$report | ForEach-Object { Write-Host $_ }
$reportPath = Join-Path $repo "gradcam-diagnostic.txt"
$report | Set-Content $reportPath
Write-Host "`nReport saved to: $reportPath" -ForegroundColor Green

if (!(Test-Path $modelPath)) {
    Write-Host "`nGrad-CAM cannot be considered fully working until a compatible trained model exists." -ForegroundColor Yellow
    Write-Host "No fake model will be generated, and training will not run against missing or incomplete classes." -ForegroundColor Yellow
} else {
    Write-Host "`nModel file exists. Next, run the API and test a real image prediction plus its Grad-CAM response." -ForegroundColor Cyan
}

Write-Host "`nDiagnostic complete. No cloud deployment or Git push was attempted." -ForegroundColor Cyan
