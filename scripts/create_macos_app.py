from __future__ import annotations

import shutil
import stat
from importlib.metadata import version as package_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "MorphoGlia.app"
CONTENTS = APP / "Contents"
MACOS = CONTENTS / "MacOS"
VERSION = package_version("morphoglia")

if APP.exists():
    shutil.rmtree(APP)

MACOS.mkdir(parents=True, exist_ok=True)

launcher = MACOS / "MorphoGlia"
launcher.write_text(
    """#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

if [ -x "$HOME/.pixi/bin/pixi" ]; then
    PIXI="$HOME/.pixi/bin/pixi"
elif command -v pixi >/dev/null 2>&1; then
    PIXI="$(command -v pixi)"
else
    osascript -e 'display dialog "MorphoGlia is not installed yet. Run Install MorphoGlia.command first." buttons {"OK"} default button "OK" with icon caution'
    exit 1
fi

cd "$ROOT"
exec "$PIXI" run gui
""",
    encoding="utf-8",
)
launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

(CONTENTS / "Info.plist").write_text(
    f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>MorphoGlia</string>
  <key>CFBundleDisplayName</key>
  <string>MorphoGlia</string>
  <key>CFBundleIdentifier</key>
  <string>org.morphoglia.app</string>
  <key>CFBundleVersion</key>
  <string>{VERSION}</string>
  <key>CFBundleShortVersionString</key>
  <string>{VERSION}</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleExecutable</key>
  <string>MorphoGlia</string>
  <key>LSMinimumSystemVersion</key>
  <string>10.15</string>
</dict>
</plist>
""",
    encoding="utf-8",
)

print(f"Created: {APP}")
print(f"Version: {VERSION}")
