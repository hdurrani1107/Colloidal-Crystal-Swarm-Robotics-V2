# Validation Testing Framework

## Overview
This framework provides a comprehensive validation testing system to compare the performance of **LJ-Swarm** vs **Olfati-Saber-Flock** algorithms.

## Validation Metrics
- **Total time to complete 10 goals/cooling-zones**
- **Average time to complete 1 goal/beacon**

## How It Works

### 1. Goal Generation
- Generates a predetermined set of 10 goal positions and radii for each trial
- Both algorithms receive the **exact same** goal set for fair comparison
- Goals are spawned following each algorithm's native spawning logic:
  - Maximum 3 goals/zones active at once
  - Fixed spawn interval after goal completion
  - Position and radius from normal distribution

### 2. Concurrent Execution
- Both simulations run with identical goal sets
- Each simulation follows its own physics and spawning rules
- Simulations wait for each other to complete before starting next trial

### 3. Data Collection
- Records completion times for each goal
- Calculates total simulation time and average time per goal
- Saves individual trial data and generates comparison graphs

## Files

### Core Files
- `validation_runner.py` - Main validation framework
- `custom_systems.py` - Custom goal/zone systems for predetermined goal injection
- `test_validation.py` - Quick test script (1 trial each)

### Usage

#### Quick Test (1 trial each)
```bash
cd Validation-Test
python test_validation.py
```

#### Full Validation Study (10 trials each - default)
```bash
cd Validation-Test
python validation_runner.py
```

#### Custom Number of Trials
```bash
cd Validation-Test
python validation_runner.py 5  # Run 5 trials each
```

## Output Structure

Results saved to: `../output/validation-test/`

### Generated Files
- `lj_swarm_trial_N.json` - Individual LJ-Swarm trial data
- `olfati_saber_trial_N.json` - Individual Olfati-Saber trial data
- `validation_comparison.png` - Comprehensive comparison graphs
- `validation_summary.json` - Statistical summary and performance comparison

### Analysis Graphs
1. **Total completion time per trial** (line plot)
2. **Average time per goal per trial** (line plot)  
3. **Total completion time distribution** (box plots)
4. **Average time per goal distribution** (box plots)

## Key Features

### Physics Preservation
- ✅ Preserves all core physics for each algorithm
- ✅ Maintains original spawning logic and parameters
- ✅ Uses identical goal sets for fair comparison

### Robust Testing
- ✅ Handles simulation failures gracefully
- ✅ Provides detailed error reporting
- ✅ Reproducible results with seed-based goal generation

### Comprehensive Analysis
- ✅ Statistical summaries (mean ± std)
- ✅ Performance improvement percentages
- ✅ Visual comparison graphs
- ✅ Individual trial data preservation

## Expected Runtime
- **Quick test (1 trial each)**: ~2-5 minutes
- **Full validation (10 trials each)**: ~20-50 minutes
- Runtime depends on algorithm performance and goal completion speed

## Validation Parameters

### LJ-Swarm Parameters
- 150 agents, σ=3.0, ε=3.0
- Cooling zones: 10±1 radius, 1000 frame lifetime
- Temperature: 150K (constant)

### Olfati-Saber Parameters  
- 150 agents, c1_γ=7, c2_γ=0.53 (Phase 1 tuning)
- Goal beacons: 12±2 radius, 1000 frame lifetime
- No temperature dependency

## Results Interpretation

### Performance Metrics
- **Lower times = Better performance**
- **Smaller standard deviation = More consistent**
- **Success rate = Reliability measure**

### Expected Outcomes
Based on Phase 1 tuning, Olfati-Saber should show:
- 2-3x faster goal completion
- More consistent performance
- Better scalability with multiple goals

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure all parent folders (LJ-Swarm, Olfati-Saber-Flock) are present
2. **Memory issues**: Reduce number of trials if system runs out of memory
3. **Simulation hangs**: Check for infinite loops in algorithm implementations

### Debug Mode
Add debug prints in `validation_runner.py` to monitor simulation progress and identify bottlenecks.