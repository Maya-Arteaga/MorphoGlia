# -*- mode: python ; coding: utf-8 -*-

import sys; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

a = Analysis(
    ['Morphoglia_app.py'],  # Main script
    pathex=[],  # Additional paths to search for imports
    binaries=[],  # Binary files to include
    datas=[('resources/transition_image.png', 'resources'), 
           ('resources/icon.ico', 'resources')],  # Data files to include
    hiddenimports=[],  # Modules that PyInstaller cannot automatically detect
    hookspath=[],  # Custom hook paths
    hooksconfig={},  # Hook configurations
    runtime_hooks=[],  # Scripts to run at startup
    excludes=[],  # Modules to exclude
    noarchive=False,  # Do not store modules in an archive
    optimize=0,  # Optimization level
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MorphoGlia',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to True for debugging
    icon='resources/icon.ico',
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,
)
