param(
    [int[]]$Seeds = @(2021, 2022, 2023),
    [string]$Gpu = "0",
    [string[]]$Markets = @("NP", "PJM", "BE", "FR", "DE"),
    [int]$NumWorkers = 0
)

$env:CUDA_VISIBLE_DEVICES = $Gpu
$PYTHON = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

$model_name = "TimeXer"
$root_path = "./dataset/EPF/"
$patch_len = 24

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Running TimeXer on EPF Benchmark (Table 2 in Paper)" -ForegroundColor Green
Write-Host "  Markets: $($Markets -join ', ') | Seeds: $($Seeds -join ', ') | Workers: $NumWorkers | GPU: $Gpu" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

foreach ($seed in $Seeds) {
    $des = "Timexer-MS-seed$seed"

    if ($Markets -contains "NP") {
        Write-Host "`n>>> [EPF-NP] Seed $seed | batch_size 4 | e_layers 3 <<<" -ForegroundColor Cyan
        & $PYTHON -u run.py `
          --is_training 1 `
          --task_name long_term_forecast `
          --root_path $root_path `
          --data_path NP.csv `
          --model_id "NP_168_24" `
          --model $model_name `
          --data custom `
          --features MS `
          --seq_len 168 `
          --pred_len 24 `
          --e_layers 3 `
          --enc_in 3 `
          --dec_in 3 `
          --c_out 1 `
          --des $des `
          --patch_len $patch_len `
          --d_model 512 `
          --d_ff 512 `
          --batch_size 4 `
          --num_workers $NumWorkers `
          --seed $seed `
          --itr 1
    }

    if ($Markets -contains "PJM") {
        Write-Host "`n>>> [EPF-PJM] Seed $seed | batch_size 16 | e_layers 3 <<<" -ForegroundColor Cyan
        & $PYTHON -u run.py `
          --is_training 1 `
          --task_name long_term_forecast `
          --root_path $root_path `
          --data_path PJM.csv `
          --model_id "PJM_168_24" `
          --model $model_name `
          --data custom `
          --features MS `
          --seq_len 168 `
          --pred_len 24 `
          --e_layers 3 `
          --enc_in 3 `
          --dec_in 3 `
          --c_out 1 `
          --des $des `
          --patch_len $patch_len `
          --d_model 512 `
          --batch_size 16 `
          --num_workers $NumWorkers `
          --seed $seed `
          --itr 1
    }

    if ($Markets -contains "BE") {
        Write-Host "`n>>> [EPF-BE] Seed $seed | batch_size 16 | e_layers 2 <<<" -ForegroundColor Cyan
        & $PYTHON -u run.py `
          --is_training 1 `
          --task_name long_term_forecast `
          --root_path $root_path `
          --data_path BE.csv `
          --model_id "BE_168_24" `
          --model $model_name `
          --data custom `
          --features MS `
          --seq_len 168 `
          --pred_len 24 `
          --e_layers 2 `
          --enc_in 3 `
          --dec_in 3 `
          --c_out 1 `
          --des $des `
          --patch_len $patch_len `
          --d_model 512 `
          --d_ff 512 `
          --batch_size 16 `
          --num_workers $NumWorkers `
          --seed $seed `
          --itr 1
    }

    if ($Markets -contains "FR") {
        Write-Host "`n>>> [EPF-FR] Seed $seed | batch_size 16 | e_layers 2 <<<" -ForegroundColor Cyan
        & $PYTHON -u run.py `
          --is_training 1 `
          --task_name long_term_forecast `
          --root_path $root_path `
          --data_path FR.csv `
          --model_id "FR_168_24" `
          --model $model_name `
          --data custom `
          --features MS `
          --seq_len 168 `
          --pred_len 24 `
          --e_layers 2 `
          --enc_in 3 `
          --dec_in 3 `
          --c_out 1 `
          --des $des `
          --patch_len $patch_len `
          --batch_size 16 `
          --d_model 512 `
          --num_workers $NumWorkers `
          --seed $seed `
          --itr 1
    }

    if ($Markets -contains "DE") {
        Write-Host "`n>>> [EPF-DE] Seed $seed | batch_size 4 | e_layers 1 <<<" -ForegroundColor Cyan
        & $PYTHON -u run.py `
          --is_training 1 `
          --task_name long_term_forecast `
          --root_path $root_path `
          --data_path DE.csv `
          --model_id "DE_168_24" `
          --model $model_name `
          --data custom `
          --features MS `
          --seq_len 168 `
          --pred_len 24 `
          --e_layers 1 `
          --enc_in 3 `
          --dec_in 3 `
          --c_out 1 `
          --des $des `
          --patch_len $patch_len `
          --batch_size 4 `
          --d_model 512 `
          --num_workers $NumWorkers `
          --seed $seed `
          --itr 1
    }
}

if (Test-Path "utils/export_results.py") {
    Write-Host "`n[EPF] Cập nhật kết quả ra docs/results/raw/results.csv..." -ForegroundColor Yellow
    & $PYTHON utils/export_results.py
}
