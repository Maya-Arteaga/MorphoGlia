#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [ -x "$HOME/.pixi/bin/pixi" ]; then
    PIXI="$HOME/.pixi/bin/pixi"
elif command -v pixi >/dev/null 2>&1; then
    PIXI="$(command -v pixi)"
else
    osascript -e 'display dialog "MorphoGlia is not installed yet. Run Install MorphoGlia.command first." buttons {"OK"} default button "OK" with icon caution'
    exit 1
fi
exec "$PIXI" run gui
