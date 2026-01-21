# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for FB Data Scraper.

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
    # EasyOCR
    'easyocr',
    'skimage',
    'skimage.transform',
    
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
    
    # pkg_resources dependencies
    'jaraco',
    'jaraco.text',
    'jaraco.functools',
    'jaraco.context',
]

# Collect packages that need special handling
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect EasyOCR model files
try:
    datas += collect_data_files('easyocr')
except Exception:
    print("Warning: Could not collect easyocr data files")

# Collect jaraco submodules (required by pkg_resources)
try:
    hiddenimports += collect_submodules('jaraco')
except Exception:
    print("Warning: Could not collect jaraco submodules")

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
        # Note: Do NOT exclude 'setuptools' - pkg_resources needs it and jaraco
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
    name='FB_Data_Scraper',
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
