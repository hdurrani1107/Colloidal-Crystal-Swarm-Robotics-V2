#######################################################################
# run_validation.py
#
# Convenient launcher for validation testing
# Handles unicode issues and provides user-friendly interface
#
# Author: Humzah Durrani
#######################################################################

import os
import sys
import argparse

def main():
    """Main launcher for validation testing"""
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='LJ-Swarm vs Olfati-Saber Validation Testing')
    parser.add_argument('--trials', '-t', type=int, default=10, 
                       help='Number of trials to run for each algorithm (default: 10)')
    parser.add_argument('--test', action='store_true', 
                       help='Run quick test with 1 trial each')
    parser.add_argument('--output', '-o', type=str, default='../output/validation-test',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Import validation modules
    try:
        from validation_runner import run_validation_study
        print("Validation Testing Framework")
        print("=" * 50)
        
        if args.test:
            print("Running quick test (1 trial each)...")
            run_validation_study(num_trials=1, output_base_dir=args.output + "/test")
            print("Quick test completed successfully!")
        else:
            print(f"Running full validation study ({args.trials} trials each)...")
            print("This may take 20-50 minutes depending on performance.")
            print("Press Ctrl+C to cancel if needed.")
            print()
            
            run_validation_study(num_trials=args.trials, output_base_dir=args.output)
            print(f"Full validation study completed!")
            
        print(f"Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\nValidation study interrupted by user.")
    except Exception as e:
        print(f"Error running validation study: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()