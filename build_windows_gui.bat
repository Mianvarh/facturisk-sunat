@echo off
cd /d "%~dp0"
python -m PyInstaller build_windows_gui.spec --clean -y
if errorlevel 1 (
  echo.
  echo No se pudo construir la interfaz grafica.
  pause
  exit /b 1
)
for %%D in (data outputs docs config assets) do (
  if exist "%%D" (
    robocopy "%%D" "dist\FactuRisk_SUNAT\%%D" /E /R:1 /W:1 /NFL /NDL /NP
    if errorlevel 8 (
      echo No se pudo copiar %%D al portable grafico.
      pause
      exit /b 1
    )
  )
)
echo.
echo Interfaz grafica generada en dist\FactuRisk_SUNAT
pause
