@echo off
setlocal
pushd %~dp0
echo [1/1] Training model from training_data_*.csv ...
python train_model.py
popd
endlocal
