# Simple Validation Test

## Overview
This is a simple, effective validation script that compares LJ-Swarm vs Olfati-Saber-Flock performance using identical zone sequences.

## What it does
1. **Generates** a random sequence of 10 zone positions and radii
2. **Runs LJ-Swarm** simulation with these zones  
3. **Runs Olfati-Saber-Flock** simulation with the same zones
4. **Repeats** for 10 iterations
5. **Creates graphs** comparing the two metrics:
   - Total time to complete 10 zones/beacons
   - Average time per zone/beacon

## Key Features
✅ **Simple and effective** - Single script, easy to understand  
✅ **Fair comparison** - Both simulations use identical zone sequences  
✅ **Preserves physics** - No changes to core simulation logic  
✅ **Automatic metrics** - Extracts performance data automatically  
✅ **Visual results** - Creates comparison graphs and statistics  

## Usage

### Run the validation
```bash
cd Validation-Test
python validation.py
```

### What happens
- The script runs 10 iterations automatically
- Each iteration uses the same random zone sequence for both simulations
- Results are saved to `../output/simple-validation/`
- Comparison graphs are displayed and saved
- Summary statistics are printed

### Output Files
- `validation_results.json` - Raw data from all iterations
- `validation_comparison.png` - Comparison graphs
- Console output with summary statistics

## Expected Runtime
- **Total time**: ~10-30 minutes (depends on simulation performance)
- **Per iteration**: ~1-3 minutes each
- **Progress**: Real-time updates shown during execution

## Results Interpretation
- **Lower times = Better performance**
- **Graphs show**: Performance across iterations
- **Statistics show**: Mean ± standard deviation for both algorithms
- **Improvement percentages**: How much faster one algorithm is vs the other

## Technical Details

### Zone Generation
- 10 zones per iteration with random positions and radii
- Same sequence used for both LJ-Swarm and Olfati-Saber-Flock
- Zones positioned with margin from boundaries to ensure valid placement

### Simulation Parameters
- **LJ-Swarm**: 150 agents, σ=3.0, ε=3.0, constant temp=150K
- **Olfati-Saber**: 150 agents, 3 flocks, goal beacon system
- **Both**: 10 target zones, max 3 concurrent zones, 400 frame spawn interval

### Metrics Collected
1. **Total completion time**: Time to finish all 10 zones/beacons
2. **Average time per zone**: Total time ÷ 10 zones

## Previous System
The old validation system has been moved to `old_validation_system/` folder. It was more complex but provided similar functionality.