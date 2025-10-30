#!/usr/bin/env python3
"""
Test script to verify YCB Viewer setup
"""

import sys
from pathlib import Path

def check_environment():
    """Check if all required packages are available"""
    print("Checking Python packages...")
    
    try:
        import numpy as np
        print("✓ NumPy available")
    except ImportError:
        print("✗ NumPy not found")
        return False
        
    try:
        import habitat_sim
        print("✓ Habitat-Sim available")
    except ImportError:
        print("✗ Habitat-Sim not found")
        return False
        
    try:
        import cv2
        print("✓ OpenCV available")
    except ImportError:
        print("✗ OpenCV not found")
        return False
        
    return True

def check_assets():
    """Check if required assets are available"""
    print("\nChecking assets...")
    
    # Check void scene
    void_scene = Path("assets/scenes/examiner/void_black.glb")
    if void_scene.exists():
        print("✓ Void scene found")
    else:
        print("✗ Void scene not found - will be generated automatically")
        
    # Check YCB configs
    config_dir = Path("assets/ycb/configs")
    if config_dir.exists():
        config_files = list(config_dir.glob("*.object_config.json"))
        print(f"✓ Found {len(config_files)} YCB object configurations")
    else:
        print("✗ YCB configs directory not found")
        return False
        
    # Check YCB meshes
    mesh_dir = Path("assets/ycb/meshes")
    if mesh_dir.exists():
        print("✓ YCB meshes directory found")
    else:
        print("✗ YCB meshes directory not found - objects may not render")
        
    return True

def main():
    """Run all checks"""
    print("YCB Viewer Setup Check")
    print("=" * 30)
    
    env_ok = check_environment()
    assets_ok = check_assets()
    
    print("\n" + "=" * 30)
    if env_ok and assets_ok:
        print("✓ Setup looks good! You can run: python ycb_viewer.py")
    else:
        print("✗ Setup issues found. Please check the requirements.")
        if not env_ok:
            print("  - Make sure conda environment is activated: conda activate sdr-dashboard")
        if not assets_ok:
            print("  - Run asset download script: python scripts/download_ycb.py")
            
    return 0 if (env_ok and assets_ok) else 1

if __name__ == "__main__":
    sys.exit(main())