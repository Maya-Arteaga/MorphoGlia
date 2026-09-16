#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
clear
printf "\n============================================================\n"
printf "              MORPHOGLIA 2.0.0 INSTALLER\n"
printf "============================================================\n\n"
printf "This can take a few minutes the first time.\n"
printf "An internet connection is required for installation.\n\n"
if [ -x "$HOME/.pixi/bin/pixi" ]; then
    PIXI="$HOME/.pixi/bin/pixi"
elif command -v pixi >/dev/null 2>&1; then
    PIXI="$(command -v pixi)"
else
    printf "Installing the MorphoGlia environment manager...\n\n"
    curl -fsSL https://pixi.sh/install.sh | sh
    PIXI="$HOME/.pixi/bin/pixi"
fi
printf "\nInstalling the locked MorphoGlia environment...\n"
"$PIXI" install --locked
printf "\nChecking the installation...\n"
"$PIXI" run doctor
printf "\nCreating the MorphoGlia application...\n"
"$PIXI" run python scripts/create_macos_app.py
printf "\nMorphoGlia 2.0.0 is ready. Opening MorphoGlia...\n\n"
open "$ROOT/MorphoGlia.app"
sleep 2
