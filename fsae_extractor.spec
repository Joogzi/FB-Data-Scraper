# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for FSAE Data Extractor.

This builds a standalone Windows executable that can be distributed
to other users without requiring Python installation.

Usage:
    pyinstaller fsae_extractor.spec

Or use the build script:
    python build.py
"""

import sys
from pathlib import Path

block_cipher = None

# Get the project root
project_root = Path(SPECPATH)

# Collect data files
datas = [
    # Add any data files, configs, etc. here
    # ('data_folder', 'data_folder'),
]

# Hidden imports for OCR engines
hiddenimports = [
    # PaddleOCR dependencies
    'paddleocr',
    'paddle',
    'paddle.fluid',
    'skimage',
    'skimage.transform',
    'imgaug',
    'lmdb',
    'shapely',
    'pyclipper',
    
    # EasyOCR fallback
    'easyocr',
    
    # PyQt6
    'PyQt6.sip',
    
    # OpenCV
    'cv2',
    
    # NumPy
    'numpy',
    
    # Other utilities
    'yaml',
    'PIL',
    'PIL.Image',
]

# Collect packages that need special handling
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect PaddleOCR model files and dependencies
try:
    datas += collect_data_files('paddleocr')
except Exception:
    print("Warning: Could not collect paddleocr data files")

# Collect EasyOCR model files
try:
    datas += collect_data_files('easyocr')
except Exception:
    print("Warning: Could not collect easyocr data files")

a = Analysis(
    ['run.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'matplotlib',
        'tkinter',
        'tcl',
        'tk',
        'pytest',
        'sphinx',
        'setuptools',
        'pip',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FSAE_Data_Extractor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if Path('assets/icon.ico').exists() else None,
    # Windows-specific options
    version='version_info.txt' if Path('version_info.txt').exists() else None,
)
