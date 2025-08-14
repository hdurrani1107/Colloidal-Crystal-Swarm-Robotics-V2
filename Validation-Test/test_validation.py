#######################################################################
# test_validation.py
#
# Quick test script to validate the validation framework
# Runs 1 trial of each algorithm to ensure everything works
#
# Author: Humzah Durrani
#######################################################################

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from validation_runner import run_validation_study

def test_validation_framework():
    """Run a quick test with 1 trial to validate the framework"""
    print("🧪 Testing Validation Framework...")
    print("This will run 1 trial of each algorithm to ensure everything works correctly.")
    
    try:
        # Run with just 1 trial for testing
        run_validation_study(num_trials=1, output_base_dir="../output/validation-test/test")
        print("✅ Validation framework test completed successfully!")
        
    except Exception as e:
        print(f"❌ Validation framework test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_validation_framework()