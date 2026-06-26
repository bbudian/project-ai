@echo off
setlocal
rem ProjectAI trainer launcher.
rem   - Double-click this and it will ask for a text file.
rem   - Or DRAG a .txt file onto this icon to train on it.
rem   - Or from a terminal:  train.cmd path\to\text.txt --size small --name mymodel
rem Trains on your GPU (CUDA) if available, else CPU, and saves to checkpoints\<name>.ckpt.
dotnet run -c Release --project "%~dp0ProjectAI.Trainer" -- %*
echo.
pause
