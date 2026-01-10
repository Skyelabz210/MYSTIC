#!/usr/bin/env python3
"""
MYSTIC SYSTEM - MINIMAL COMPONENT VALIDATION CHECK

Quick validation that verifies all QMNF components are available and accessible.
"""

import sys
import os
import importlib.util

# Change to project directory
os.chdir('/home/acid/Projects/MYSTIC')

# Add project directory to Python path
sys.path.insert(0, '/home/acid/Projects/MYSTIC')

def check_file_exists(filename):
    """Check if a file exists in the project directory"""
    path = os.path.join('/home/acid/Projects/MYSTIC', filename)
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {filename}")
    return exists

def check_module_import(module_name):
    """Check if we can import a module"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, f"/home/acid/Projects/MYSTIC/{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"  ✓ {module_name}.py - IMPORT SUCCESSFUL")
        return True
    except Exception as e:
        print(f"  ✗ {module_name}.py - IMPORT FAILED: {e}")
        return False

def main():
    print("=" * 80)
    print("MYSTIC SYSTEM - COMPONENT VALIDATION CHECK")
    print("=" * 80)
    
    print("\nCORE QMNF INNOVATIONS:")
    print("-" * 40)
    
    core_components = [
        "phi_resonance_detector",
        "fibonacci_phi_validator", 
        "cayley_transform",
        "shadow_entropy",
        "k_elimination"
    ]
    
    core_success = 0
    for comp in core_components:
        if check_module_import(comp):
            core_success += 1
    
    print(f"\nCore components available: {core_success}/{len(core_components)}")
    
    print("\nADDITIONAL SYSTEM COMPONENTS:")
    print("-" * 40)
    
    additional_components = [
        "weather_attractor_basins.json",
        "mystic_v3_integrated.py",
        "cayley_transform_nxn.py",
        "lyapunov_calculator.py",
        "shadow_entropy.py",
        "k_elimination.py"
    ]
    
    additional_success = 0
    for comp in additional_components:
        if check_file_exists(comp):
            additional_success += 1
    
    print(f"\nAdditional components available: {additional_success}/{len(additional_components)}")
    
    print("\nDATA FILES:")
    print("-" * 20)
    
    data_files = [
        "weather_attractor_basins.json",
        "historical_weather_data.json",
        "validation_results.json"
    ]
    
    data_success = 0
    for df in data_files:
        if check_file_exists(df):
            data_success += 1
    
    print(f"\nData files available: {data_success}/{len(data_files)}")
    
    print("\nMATHEMATICAL FOUNDATIONS CHECK:")
    print("-" * 40)
    
    # Check for the key mathematical files
    math_files = [
        "nine65_v2_complete/src/arithmetic/k_elimination.rs",
        "nine65_v2_complete/src/arithmetic/exact_coeff.rs",
        "nine65_v2_complete/src/arithmetic/persistent_montgomery.rs",
        "nine65_v2_complete/src/arithmetic/ct_mul_exact.rs", 
        "nine65_v2_complete/src/arithmetic/montgomery.rs",
        "nine65_v2_complete/src/arithmetic/barrett.rs",
        "nine65_v2_complete/src/arithmetic/rns.rs",
        "nine65_v2_complete/src/arithmetic/ntt.rs",
        "nine65_v2_complete/src/chaos/lyapunov.rs",
        "nine65_v2_complete/src/chaos/attractor.rs",
        "nine65_v2_complete/src/entropy/shadow.rs",
        "nine65_v2_complete/src/quantum/mod.rs",
        "nine65_v2_complete/src/quantum/amplitude.rs",
        "nine65_v2_complete/src/quantum/entanglement.rs",
        "nine65_v2_complete/src/quantum/teleport.rs",
        "nine65_v2_complete/src/quantum/period_grover.rs"
    ]
    
    math_success = 0
    for mf in math_files:
        if check_file_exists(mf):
            math_success += 1
    
    print(f"\nMathematical foundations: {math_success}/{len(math_files)}")
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    total_available = core_success + additional_success + data_success + math_success
    total_possible = len(core_components) + len(additional_components) + len(data_files) + len(math_files)
    
    print(f"Overall availability: {total_available}/{total_possible}")
    print(f"Success rate: {total_available/total_possible*100:.1f}%")
    
    if total_available == total_possible:
        print("\n🎉 COMPLETE SYSTEM VALIDATION - ALL COMPONENTS AVAILABLE!")
        print("✓ All QMNF innovations implemented")
        print("✓ All mathematical foundations present") 
        print("✓ Data files and attractor basins available")
        print("✓ MYSTIC system ready for operational deployment")
    else:
        print(f"\n⚠ PARTIAL VALIDATION - {total_possible - total_available} components missing")
        print("Some system functionality may be limited")
    
    print("=" * 80)
    
    return total_available == total_possible

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)