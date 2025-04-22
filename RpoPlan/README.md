# fullOrbit.py - Orbital Visualization Tool

## Overview
`fullOrbit.py` is a Python script that creates interactive 3D visualizations of spacecraft orbital data. It generates a dual-panel visualization showing both a global Earth-centered view and a deputy-centered relative view of spacecraft trajectories.

## Features
- **Dual-panel visualization**:
  - Left panel: Global view showing Earth and spacecraft orbits
  - Right panel: Deputy-centered view showing relative approach
- **Dynamic visualization elements**:
  - Spacecraft trajectories revealed step by step
  - Two orbits worth of trajectory lines displayed
  - Spacecraft markers sized dynamically based on relative distance
  - Mission time displayed in hours
  - Relative range displayed in kilometers
- **Interactive controls**:
  - Press 'q' to exit the animation
- **Animation recording**:
  - Option to record the animation as a GIF file
  - Can be converted to MP4 using external tools like FFmpeg

## Requirements
- Python 3.x
- Required packages:
  - numpy
  - pandas
  - pyvista
  - argparse

## Installation
1. Ensure Python 3.x is installed
2. Install required packages:
   ```
   pip install numpy pandas pyvista
   ```

## Usage
### Basic Usage
```
python fullOrbit.py
```
This will run the visualization using the default CSV file path (`../CelestialChoreography/Data/RpoPlan.csv`).

### Command Line Arguments
- `--csv`: Path to the CSV file with orbital data
  ```
  python fullOrbit.py --csv path/to/your/data.csv
  ```

- `--record`: Record the animation to a GIF file
  ```
  python fullOrbit.py --record
  ```

- `--output`: Specify output path for the recorded animation
  ```
  python fullOrbit.py --record --output path/to/save/animation.mp4
  ```

## Input Data Format
The script expects a CSV file with the following columns:
- `secondsSinceStart`: Time in seconds since the start of the mission
- `positionChiefEciX/Y/Z`: Chief spacecraft position in Earth-Centered Inertial (ECI) coordinates
- `positionDeputyEciX/Y/Z`: Deputy spacecraft position in ECI coordinates
- `positionDepRelToChiefLvlhX/Y/Z`: Deputy position relative to Chief in Local-Vertical-Local-Horizontal (LVLH) frame
- `relativeRange`: (Optional) Range between spacecraft in meters

## Visualization Details
1. **Global View (Left Panel)**:
   - Shows Earth with texture (if available)
   - Displays chief spacecraft (cyan) and deputy spacecraft (magenta)
   - Trajectories shown in blue (chief) and red (deputy)

2. **LVLH View (Right Panel)**:
   - Deputy-centered view showing relative approach
   - Chief spacecraft (blue) at origin
   - Deputy spacecraft (red) showing relative position
   - Trajectory shown in cyan

3. **Information Display**:
   - Mission time in hours
   - Mission completion percentage
   - Relative range in kilometers
   - Current frame counter