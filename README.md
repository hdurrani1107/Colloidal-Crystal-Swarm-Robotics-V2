# Colloidal-Crystal-Swarm-Robotics-V2
Cleaning up and iterating from first version of repository

## Project Structure

### lj_swarm/
LJ-Swarm simulation package:
- `main.py` -> Run LJ-Swarm simulation from here
- `ui.py` -> Matplotlib plot code
- `engine.py` -> Engine Swarm Logic
- `cooling_zone.py` -> Cooling zone system

### olfati_saber_flock/
Olfati-Saber flocking simulation package:
- `main.py` -> Run Olfati-Saber simulation from here
- `engine.py` -> Flocking engine logic
- `goal_beacon.py` -> Goal beacon system

### Validation-Test/
Validation testing framework:
- `validation.py` -> Run validation comparison between both algorithms
- `README.md` -> Detailed validation usage instructions

# To do:
- Clean up repository for un-used and redundant code
- Potentially add some autonomous features to LJ-Swarm and validate back
- Expand Validation Testing and test metrics