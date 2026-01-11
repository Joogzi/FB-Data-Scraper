#!/usr/bin/env python3
"""
Build script for FSAE Data Extractor.

Creates a standalone executable using PyInstaller.

Usage:
    python build.py          # Build executable
    python build.py --clean  # Clean build artifacts first
    python build.py --onedir # Build as directory instead of single file
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def clean_build_artifacts():
    """Remove build artifacts."""
    print("🧹 Cleaning build artifacts...")
    
    dirs_to_remove = ['build', 'dist', '__pycache__']
    files_to_remove = ['*.pyc', '*.pyo']
    
    for dir_name in dirs_to_remove:
        for path in Path('.').rglob(dir_name):
            if path.is_dir():
                print(f"  Removing {path}")
                shutil.rmtree(path, ignore_errors=True)
    
    for pattern in files_to_remove:
        for path in Path('.').rglob(pattern):
            if path.is_file():
                path.unlink()
    
    print("  ✓ Clean complete\n")


def check_dependencies():
    """Check that required dependencies are installed."""
    print("📦 Checking dependencies...")
    
    required = ['PyInstaller', 'PyQt6', 'opencv-python', 'numpy']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_').split('[')[0])
        except ImportError:
            # Handle package name differences
            pkg_name = pkg.lower().replace('-', '_')
            if pkg_name == 'pyinstaller':
                pkg_name = 'PyInstaller'
            elif pkg_name == 'opencv_python':
                pkg_name = 'cv2'
            try:
                __import__(pkg_name)
            except ImportError:
                missing.append(pkg)
    
    if missing:
        print(f"  ❌ Missing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False
    
    print("  ✓ All dependencies available\n")
    return True


def create_version_info():
    """Create Windows version info file."""
    version_info = """
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'FSAE'),
        StringStruct(u'FileDescription', u'FSAE Scraper - Video Telemetry Analysis'),
        StringStruct(u'FileVersion', u'1.0.0'),
        StringStruct(u'InternalName', u'fsae_scraper'),
        StringStruct(u'LegalCopyright', u'Copyright 2026'),
        StringStruct(u'OriginalFilename', u'FSAE_Scraper.exe'),
        StringStruct(u'ProductName', u'FSAE Scraper'),
        StringStruct(u'ProductVersion', u'1.0.0')])
      ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
    with open('version_info.txt', 'w') as f:
        f.write(version_info.strip())
    print("  ✓ Created version_info.txt")


def build_executable(onedir=False):
    """Build the executable using PyInstaller."""
    print("🔨 Building executable...")
    print("  This may take several minutes...\n")
    
    # Create version info for Windows
    if sys.platform == 'win32':
        create_version_info()
    
    # Build command
    if onedir:
        # Build as directory (faster, easier to debug)
        cmd = [
            sys.executable, '-m', 'PyInstaller',
            '--noconfirm',
            '--onedir',
            '--windowed',
            '--name', 'FSAE_Data_Extractor',
            '--add-data', 'src;src',
            'run.py'
        ]
    else:
        # Build using spec file (single exe)
        cmd = [
            sys.executable, '-m', 'PyInstaller',
            '--noconfirm',
            'fsae_extractor.spec'
        ]
    
    print(f"  Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("\n❌ Build failed!")
        return False
    
    print("\n✅ Build successful!")
    
    # Show output location
    if onedir:
        output_path = Path('dist/FSAE_Data_Extractor')
    else:
        output_path = Path('dist/FSAE_Data_Extractor.exe')
    
    if output_path.exists():
        if output_path.is_file():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"\n📁 Output: {output_path.absolute()}")
            print(f"   Size: {size_mb:.1f} MB")
        else:
            print(f"\n📁 Output folder: {output_path.absolute()}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Build FSAE Data Extractor executable')
    parser.add_argument('--clean', action='store_true', help='Clean build artifacts first')
    parser.add_argument('--onedir', action='store_true', help='Build as directory instead of single file')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency check')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("  FSAE Data Extractor - Build Script")
    print("=" * 50 + "\n")
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    if args.clean:
        clean_build_artifacts()
    
    if not args.skip_deps:
        if not check_dependencies():
            sys.exit(1)
    
    if not build_executable(onedir=args.onedir):
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("  Build complete! 🎉")
    print("=" * 50)
    print("\nYou can now distribute the executable to other users.")
    print("They won't need Python installed to run it.\n")


if __name__ == '__main__':
    main()
