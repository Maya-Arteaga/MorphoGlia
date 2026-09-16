#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"
if [ -x "$HOME/.pixi/bin/pixi" ]; then
    PIXI="$HOME/.pixi/bin/pixi"
elif command -v pixi >/dev/null 2>&1; then
    PIXI="$(command -v pixi)"
else
    curl -fsSL https://pixi.sh/install.sh | sh
    PIXI="$HOME/.pixi/bin/pixi"
fi
"$PIXI" install --locked
"$PIXI" run doctor
printf "\nMorphoGlia 2.0.0 is ready. Opening the GUI...\n"
"$PIXI" run gui
