#!/usr/bin/env python3
import subprocess
import sys
import platform

def setup_mac_environment():
    print("Setting up environment for Apple Silicon Mac...")
    
    # Check if running on Apple Silicon
    if platform.system() != 'Darwin' or platform.processor() != 'arm':
        print("This script is intended for Apple Silicon Macs only.")
        return
    
    # Install PyTorch with MPS support
    print("Installing PyTorch with MPS support...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "torch>=2.0.0", 
        "torchvision>=0.15.0"
    ])
    
    # Install other dependencies
    print("Installing other dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])
    
    # Verify MPS availability
    print("Verifying MPS availability...")
    verify_script = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(f"Test tensor on MPS: {x}")
    print("MPS is working correctly!")
else:
    print("MPS is not available. Check your PyTorch installation.")
"""
    subprocess.run([sys.executable, "-c", verify_script])
    
    print("\nSetup complete! You can now run YOLO with MPS acceleration.")

if __name__ == "__main__":
    setup_mac_environment()