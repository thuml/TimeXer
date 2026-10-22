param(
    [int[]]$Seeds = @(2021, 2022, 2023),
    [string]$Gpu = "0",
    [int[]]$Horizons = @(96, 192, 336, 720),
    [int]$NumWorkers = 0
)

$env:CUDA_VISIBLE_DEVICES = $Gpu
$PYTHON = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

$model_name = "TimeXer"
$root_path = "./dataset/weather/"
$data_path = "weather.csv"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Running TimeXer on Weather Dataset (Table 11 in Paper)" -ForegroundColor Green
Write-Host "  Horizons: $($Horizons -join ', ') | Seeds: $($Seeds -join ', ') | Workers: $NumWorkers | GPU: $Gpu" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

foreach ($seed in $Seeds) {
    $des = "Timexer-MS-seed$seed"

    foreach ($pl in $Horizons) {
        Write-Host "`n>>> [Weather] Horizon $pl | Seed $seed <<<" -ForegroundColor Cyan
        & $PYTHON -u run.py `
          --task_name long_term_forecast `
          --is_training 1 `
          --root_path $root_path `
          --data_path $data_path `
          --model_id "weather_96_$pl" `
          --model $model_name `
          --data custom `
          --features MS `
          --seq_len 96 `
          --label_len 48 `
          --pred_len $pl `
          --e_layers 1 `
          --factor 3 `
          --enc_in 21 `
          --dec_in 21 `
          --c_out 21 `
          --des $des `
          --d_model 128 `
          --num_workers $NumWorkers `
          --seed $seed `
          --itr 1
    }
}

if (Test-Path "utils/export_results.py") {
    Write-Host "`n[Weather] Cập nhật kết quả ra docs/results/raw/results.csv..." -ForegroundColor Yellow
    & $PYTHON utils/export_results.py
}
