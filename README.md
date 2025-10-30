# Co-Optimization Two-Phase

Repository to reproduce the results of the paper "An Empirical Study on the Computation Budget of Co-Optimization of Robot Design and Control in Simulation"




## Supported Frameworks

The project integrates with the following robotic simulation frameworks:

1. **EvoGym** - Soft robot co-design using PPO
2. **RoboGrammar** - Rigid robot design using MCTS and MPC
3. **Tholiao** - Modular soft robot simulation (not used in the paper)
4. **GymREM2D** - 2D modular robot environment
5. **JorgenREM** - Modular robot evolution
6. **KevinCoAdapting** - Co-adapting morphology and control (not used in the paper)

## Requirements

- Python 3.7 (required for compatibility with some frameworks)
- CMake 3.20+
- CUDA-compatible GPU (optional, but recommended)
- Ubuntu 18.04 or later
- Various OpenGL libraries (see `setup.sh`)

### Key Python Dependencies

- PyTorch 1.12.0
- NumPy, Pandas, Matplotlib
- Stable-Baselines3
- PyBullet
- scipy 1.4.1
- GPy, GPyOpt
- neat-python
- CMA-ES, Nevergrad
- See `requirements.txt` for complete list

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd co-optimization-two-phase
```

2. Run the setup script:
```bash
bash setup.sh --local
```

This will:
- Install system dependencies
- Create a Python 3.7 virtual environment
- Install Python packages
- Compile GLEW
- Build RoboGrammar with Python bindings
- Install EvoGym
- Set up GymREM2D with Box2D
- Install Revolve dependencies

3. Activate the environment:
```bash
source venv/bin/activate
```


## Usage

### Running Experiments

Each framework has its own experiment script in the `src/` directory:

#### RoboGrammar Example
```bash
# Activate virtual environment
source venv/bin/activate

# Run experiment with index 0
python src/robogrammar_experiment.py --local_launch 0

# Generate plots
python src/robogrammar_experiment.py --plot

# Clean temporary files
python src/robogrammar_experiment.py --clean
```

#### EvoGym Example
```bash
python src/evogym_experiment.py --local_launch 0
python src/evogym_experiment.py --plot
```

#### Other Frameworks
```bash
python src/tholiao_experiment.py --local_launch 0
python src/gymrem2d_experiment.py --local_launch 0
python src/jorgenrem_experiment.py --local_launch 0
python src/kevincoadapting_experiment.py --local_launch 0
```

### Cluster Execution

Submit jobs to the cluster using provided SLURM scripts:
```bash
sbatch cluster_scripts/launch_one_robogrammar.sl
sbatch cluster_scripts/launch_one_evogym.sl
sbatch cluster_scripts/launch_one_jorgenrem.sl
```

### Generating Visualizations

Run all plots in background:
```bash
bash all_plots_background.sh
```


## Configuration

The `Parameters` class in `NestedOptimization.py` (src/NestedOptimization.py:65) allows configuration of:
- Number of seeds for experiments
- Stopping criteria (max steps)
- Environment-specific parameters
- Inner optimization quantities and episode lengths

## Results

Results are saved in `results/<framework>/`:
- `data/` - Raw experimental data (.txt files)
- `figures/` - Generated plots (.pdf files)
- `videos/` - Robot animations (.mp4 files)

## License

This software is released into the public domain under [The Unlicense](https://unlicense.org/).

**Note:** The code in `other_repos/` and `glew-2.1.0/` directories are from external authors and may have different licenses. Please refer to each repository for their respective licenses.




