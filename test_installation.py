#!/usr/bin/env python
"""
Test script to verify that the Rephysco package is installed correctly.
"""

import sys
import importlib.util

def check_module(module_name):
    """Check if a module is installed."""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"❌ {module_name} is NOT installed")
        return False
    else:
        print(f"✅ {module_name} is installed")
        return True

def main():
    """Main function to check the installation."""
    print("Checking Rephysco installation...\n")
    
    # Check core modules
    core_modules = [
        "rephysco",
        "rephysco.client",
        "rephysco.types",
        "rephysco.conversation",
        "rephysco.providers",
        "rephysco.cache",
        "rephysco.retry",
        "rephysco.settings",
    ]
    
    core_success = all(check_module(module) for module in core_modules)
    
    # Check provider modules
    provider_modules = [
        "rephysco.providers.openai",
        "rephysco.providers.gemini",
        "rephysco.providers.aliyun",
        "rephysco.providers.xai",
        "rephysco.providers.siliconflow",
    ]
    
    provider_success = all(check_module(module) for module in provider_modules)
    
    # Check dependencies
    dependency_modules = [
        "openai",
        "pendulum",
        "diskcache",
        "rich",
        "click",
        "pydantic",
    ]
    
    dependency_success = all(check_module(module) for module in dependency_modules)
    
    print("\nSummary:")
    if core_success:
        print("✅ Core modules are installed correctly")
    else:
        print("❌ Some core modules are missing")
    
    if provider_success:
        print("✅ Provider modules are installed correctly")
    else:
        print("❌ Some provider modules are missing")
    
    if dependency_success:
        print("✅ Dependencies are installed correctly")
    else:
        print("❌ Some dependencies are missing")
    
    if core_success and provider_success and dependency_success:
        print("\n🎉 Rephysco is installed correctly!")
        print("You can now use Rephysco in your Python projects.")
        print("\nExample usage:")
        print("python -m rephysco generate --provider openai \"Hello, world!\"")
        return 0
    else:
        print("\n❌ Rephysco installation is incomplete.")
        print("Please run the installation script again:")
        print("./install.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 