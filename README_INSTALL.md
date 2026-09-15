# Installing MorphoGlia 2.0.0

For the complete user guide, see `README.md`.

## macOS
First use: double-click `Install MorphoGlia.command`.
Later use: double-click `MorphoGlia.app`.
If macOS blocks the unsigned installer, right-click it and choose **Open**.

## Windows x86-64
First use: double-click `Install MorphoGlia.bat`.
Later use: double-click `MorphoGlia.bat`.
No separate Python or Conda installation is required.

## Linux
```bash
chmod +x install_morphoglia.sh
./install_morphoglia.sh
```
Later: `pixi run gui`

## Script interface
Edit `mg_script.py`, then run it from the release directory.

macOS/Linux:
```bash
pixi run python mg_script.py
```

Windows with global Pixi:
```powershell
pixi run python .\mg_script.py
```

Windows with local Pixi created by MorphoGlia:
```powershell
.\.pixi-home\bin\pixi.exe run python .\mg_script.py
```

## Installation check
```bash
pixi run doctor
```
