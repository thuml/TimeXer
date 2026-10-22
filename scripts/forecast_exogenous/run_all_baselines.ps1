param(
    [string[]]$Models = @("TimeXer", "iTransformer", "PatchTST", "DLinear", "RLinear", "TiDE", "TimesNet", "Nonstationary_Transformer", "Autoformer", "Crossformer", "SCINet"),
    [string[]]$Datasets = @("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"),
    [int[]]$Horizons = @(96, 192, 336, 720),
    [int[]]$Seeds = @(2021),
    [string]$Gpu = "0",
    [int]$NumWorkers = 0,
    [string]$Python = "",
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = $Gpu

# 1. Resolve Python Executable (Default to repository .venv)
if (-not $Python) {
    if (Test-Path ".\.venv\Scripts\python.exe") {
        $Python = ".\.venv\Scripts\python.exe"
    } else {
        $Python = "python"
    }
}

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "  TimeXer Paper Table 11 / 12 Full Baseline Benchmark Runner" -ForegroundColor Green
Write-Host "  Models    : $($Models -join ', ')" -ForegroundColor Cyan
Write-Host "  Datasets  : $($Datasets -join ', ')" -ForegroundColor Cyan
Write-Host "  Horizons  : $($Horizons -join ', ')" -ForegroundColor Cyan
Write-Host "  Seeds     : $($Seeds -join ', ')" -ForegroundColor Cyan
Write-Host "  GPU       : $Gpu | Workers: $NumWorkers | Python: $Python" -ForegroundColor Cyan
if ($DryRun) {
    Write-Host "  MODE      : *** DRY RUN ONLY (No models will be trained) ***" -ForegroundColor Yellow
} else {
    Write-Host "  MODE      : EXECUTION WITH RESUME / CHECKPOINT SKIP" -ForegroundColor Green
}
Write-Host "======================================================================" -ForegroundColor Green

# 2. Dataset Metadata
$DatasetMeta = @{
    "ETTh1"   = @{ root = "./dataset/ETT-small/"; file = "ETTh1.csv";   data = "ETTh1";  enc_in = 7;  dec_in = 7;  c_out = 7;  prefix = "ETTh1" }
    "ETTh2"   = @{ root = "./dataset/ETT-small/"; file = "ETTh2.csv";   data = "ETTh2";  enc_in = 7;  dec_in = 7;  c_out = 7;  prefix = "ETTh2" }
    "ETTm1"   = @{ root = "./dataset/ETT-small/"; file = "ETTm1.csv";   data = "ETTm1";  enc_in = 7;  dec_in = 7;  c_out = 7;  prefix = "ETTm1" }
    "ETTm2"   = @{ root = "./dataset/ETT-small/"; file = "ETTm2.csv";   data = "ETTm2";  enc_in = 7;  dec_in = 7;  c_out = 7;  prefix = "ETTm2" }
    "Weather" = @{ root = "./dataset/weather/";   file = "weather.csv"; data = "custom"; enc_in = 21; dec_in = 21; c_out = 21; prefix = "weather" }
}

# 3. Model Specific Arguments Builder
function Get-ModelConfig {
    param(
        [string]$Model,
        [string]$Dataset,
        [int]$Horizon
    )

    $extraArgs = @()
    switch ($Model) {
        "TimeXer" {
            $extraArgs += @("--patch_len", "16", "--factor", "3")
            if ($Dataset -eq "ETTh1") {
                if ($Horizon -eq 96)       { $extraArgs += @("--d_model", "512", "--e_layers", "2", "--d_ff", "512") }
                elseif ($Horizon -eq 192)  { $extraArgs += @("--d_model", "128", "--e_layers", "2", "--d_ff", "128") }
                elseif ($Horizon -eq 336)  { $extraArgs += @("--d_model", "512", "--e_layers", "2", "--d_ff", "512") }
                elseif ($Horizon -eq 720)  { $extraArgs += @("--d_model", "512", "--e_layers", "2", "--d_ff", "2048") }
            } elseif ($Dataset -eq "ETTh2") {
                if ($Horizon -eq 96)       { $extraArgs += @("--d_model", "128", "--e_layers", "1", "--d_ff", "128") }
                elseif ($Horizon -eq 192)  { $extraArgs += @("--d_model", "128", "--e_layers", "1", "--d_ff", "512") }
                elseif ($Horizon -eq 336)  { $extraArgs += @("--d_model", "128", "--e_layers", "2", "--d_ff", "256") }
                elseif ($Horizon -eq 720)  { $extraArgs += @("--d_model", "256", "--e_layers", "1", "--d_ff", "512") }
            } elseif ($Dataset -eq "ETTm1") {
                if ($Horizon -eq 96)       { $extraArgs += @("--d_model", "256", "--e_layers", "1", "--d_ff", "2048") }
                else                       { $extraArgs += @("--d_model", "128", "--e_layers", "1", "--d_ff", "2048") }
            } elseif ($Dataset -eq "ETTm2") {
                if ($Horizon -eq 96)       { $extraArgs += @("--d_model", "512", "--e_layers", "1", "--d_ff", "2048") }
                elseif ($Horizon -eq 192)  { $extraArgs += @("--d_model", "256", "--e_layers", "1", "--d_ff", "2048") }
                else                       { $extraArgs += @("--d_model", "128", "--e_layers", "1", "--d_ff", "2048") }
            } elseif ($Dataset -eq "Weather") {
                $extraArgs += @("--d_model", "128", "--e_layers", "1", "--d_ff", "2048")
            }
            $extraArgs += @("--batch_size", "32", "--learning_rate", "0.0001")
        }
        "PatchTST" {
            $extraArgs += @(
                "--patch_len", "16",
                "--stride", "8",
                "--e_layers", "3",
                "--n_heads", "4",
                "--d_model", "128",
                "--d_ff", "256",
                "--dropout", "0.1",
                "--batch_size", "32",
                "--learning_rate", "0.0001"
            )
        }
        "iTransformer" {
            $extraArgs += @(
                "--e_layers", "2",
                "--d_model", "512",
                "--d_ff", "512",
                "--batch_size", "32",
                "--learning_rate", "0.0001"
            )
        }
        "DLinear" {
            $extraArgs += @(
                "--moving_avg", "25",
                "--batch_size", "32",
                "--learning_rate", "0.001"
            )
        }
        "RLinear" {
            $extraArgs += @(
                "--batch_size", "32",
                "--learning_rate", "0.001"
            )
        }
        "TiDE" {
            $extraArgs += @(
                "--e_layers", "2",
                "--d_layers", "2",
                "--d_model", "256",
                "--d_ff", "256",
                "--dropout", "0.1",
                "--batch_size", "32",
                "--learning_rate", "0.0001"
            )
        }
        "TimesNet" {
            $bs = if ($Dataset -eq "Weather") { "16" } else { "32" }
            $extraArgs += @(
                "--e_layers", "2",
                "--d_model", "64",
                "--d_ff", "64",
                "--top_k", "5",
                "--num_kernels", "6",
                "--batch_size", $bs,
                "--learning_rate", "0.0001"
            )
        }
        "Nonstationary_Transformer" {
            $extraArgs += @(
                "--e_layers", "2",
                "--d_layers", "1",
                "--d_model", "128",
                "--d_ff", "256",
                "--p_hidden_dims", "256", "256",
                "--batch_size", "32",
                "--learning_rate", "0.0001"
            )
        }
        "Autoformer" {
            $extraArgs += @(
                "--e_layers", "2",
                "--d_layers", "1",
                "--d_model", "128",
                "--d_ff", "256",
                "--moving_avg", "25",
                "--batch_size", "32",
                "--learning_rate", "0.0001"
            )
        }
        "Crossformer" {
            $bs = if ($Dataset -eq "Weather") { "16" } else { "32" }
            $extraArgs += @(
                "--e_layers", "2",
                "--d_model", "64",
                "--d_ff", "64",
                "--batch_size", $bs,
                "--learning_rate", "0.0001"
            )
        }
        "SCINet" {
            $extraArgs += @(
                "--e_layers", "3",
                "--d_model", "128",
                "--d_ff", "256",
                "--batch_size", "32",
                "--learning_rate", "0.0001"
            )
        }
        Default {
            $extraArgs += @(
                "--e_layers", "2",
                "--d_model", "128",
                "--d_ff", "256",
                "--batch_size", "32",
                "--learning_rate", "0.0001"
            )
        }
    }
    return $extraArgs
}

# 4. Checkpoint / Resume Verification Helper
$resultFile = "result_long_term_forecast.txt"
function Test-IsCompleted {
    param(
        [string]$Model,
        [string]$Dataset,
        [int]$Horizon,
        [int]$Seed,
        [string]$Des
    )

    if (-not (Test-Path $resultFile)) {
        return $false
    }

    $meta = $DatasetMeta[$Dataset]
    $model_id = "$($meta.prefix)_96_$Horizon"

    # Search in result_long_term_forecast.txt
    # Example setting format in file:
    # long_term_forecast_ETTh1_96_96_PatchTST_ETTh1_ftMS_sl96_ll48_pl96_..._PatchTST-MS-seed2021_0
    $pattern = "long_term_forecast_${model_id}_${Model}_*_${Des}_0"

    $matched = Select-String -Path $resultFile -Pattern $pattern -SimpleMatch:$false -ErrorAction SilentlyContinue
    if ($matched) {
        return $true
    }

    # Alternative check: Look for exact setting line starting with task and containing model, pl, des
    $lines = Get-Content $resultFile -ErrorAction SilentlyContinue
    foreach ($line in $lines) {
        if ($line.Contains("_$model_id`_") -and $line.Contains("_$Model`_") -and $line.Contains("_pl$Horizon`_") -and $line.Contains("_$Des`_")) {
            return $true
        }
    }

    return $false
}

# 5. Main Execution Loop
$totalPlanned = $Models.Count * $Datasets.Count * $Horizons.Count * $Seeds.Count
$skippedCount = 0
$executedCount = 0
$dryRunCount = 0
$failedCount = 0
$currentIndex = 0

$startTime = Get-Date

foreach ($model in $Models) {
    Write-Host "`n######################################################################" -ForegroundColor Magenta
    Write-Host "  STARTING MODEL: $model" -ForegroundColor Magenta
    Write-Host "######################################################################" -ForegroundColor Magenta

    foreach ($dsName in $Datasets) {
        if (-not $DatasetMeta.ContainsKey($dsName)) {
            Write-Warning "Dataset '$dsName' không có trong định nghĩa. Bỏ qua."
            continue
        }
        $meta = $DatasetMeta[$dsName]
        $fullDataPath = Join-Path $meta.root $meta.file
        if (-not (Test-Path $fullDataPath)) {
            Write-Warning "Không tìm thấy file dữ liệu: $fullDataPath. Bỏ qua dataset $dsName."
            continue
        }

        foreach ($horizon in $Horizons) {
            foreach ($seed in $Seeds) {
                $currentIndex++
                $model_id = "$($meta.prefix)_96_$horizon"
                $des = if ($model -eq "TimeXer") { "Timexer-MS-seed$seed" } else { "$model-MS-seed$seed" }

                $isCompleted = Test-IsCompleted -Model $model -Dataset $dsName -Horizon $horizon -Seed $seed -Des $des

                Write-Host "[$currentIndex / $totalPlanned] Model: $model | Dataset: $dsName | Horizon: $horizon | Seed: $seed" -NoNewline

                if ($isCompleted -and (-not $Force)) {
                    Write-Host " -> [ALREADY COMPLETED, SKIPPING]" -ForegroundColor DarkGray
                    $skippedCount++
                    continue
                }

                Write-Host " -> [PENDING]" -ForegroundColor Yellow

                # Build argument list
                $cmdArgs = @(
                    "-u", "run.py",
                    "--task_name", "long_term_forecast",
                    "--is_training", "1",
                    "--root_path", $meta.root,
                    "--data_path", $meta.file,
                    "--model_id", $model_id,
                    "--model", $model,
                    "--data", $meta.data,
                    "--features", "MS",
                    "--seq_len", "96",
                    "--label_len", "48",
                    "--pred_len", "$horizon",
                    "--enc_in", "$($meta.enc_in)",
                    "--dec_in", "$($meta.dec_in)",
                    "--c_out", "$($meta.c_out)",
                    "--des", $des,
                    "--num_workers", "$NumWorkers",
                    "--seed", "$seed",
                    "--itr", "1"
                )

                $modelArgs = Get-ModelConfig -Model $model -Dataset $dsName -Horizon $horizon
                $cmdArgs += $modelArgs

                if ($DryRun) {
                    Write-Host "    [DRY-RUN CMD] $Python $($cmdArgs -join ' ')" -ForegroundColor Cyan
                    $dryRunCount++
                } else {
                    Write-Host "    >>> Executing: $model on $dsName (H=$horizon, Seed=$seed) <<<" -ForegroundColor Green
                    $runStart = Get-Date
                    try {
                        & $Python @cmdArgs
                        if ($LASTEXITCODE -ne 0) {
                            Write-Host "    [ERROR] Run exited with error code $LASTEXITCODE" -ForegroundColor Red
                            $failedCount++
                        } else {
                            $runDuration = [math]::Round(((Get-Date) - $runStart).TotalSeconds, 1)
                            Write-Host "    [SUCCESS] Finished in $runDuration s" -ForegroundColor Green
                            $executedCount++

                            # Synchronize with results.csv
                            if (Test-Path "utils/export_results.py") {
                                & $Python utils/export_results.py | Out-Null
                            }
                        }
                    } catch {
                        Write-Host "    [EXCEPTION] $_" -ForegroundColor Red
                        $failedCount++
                    }
                }
            }
        }
    }
}

$totalDuration = [math]::Round(((Get-Date) - $startTime).TotalMinutes, 2)

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "  BENCHMARK RUN SUMMARY" -ForegroundColor Green
Write-Host "  Total Configurations Planned : $totalPlanned" -ForegroundColor White
Write-Host "  Already Completed (Skipped)  : $skippedCount" -ForegroundColor DarkGray
if ($DryRun) {
    Write-Host "  Dry Run Checked (Pending)    : $dryRunCount" -ForegroundColor Cyan
} else {
    Write-Host "  Successfully Executed        : $executedCount" -ForegroundColor Green
    Write-Host "  Failed                       : $failedCount" -ForegroundColor Red
}
Write-Host "  Total Time Elapsed           : $totalDuration minutes" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Green
