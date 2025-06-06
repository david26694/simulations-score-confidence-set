# Simulations Score Confidence Set

This project provides tools for computing confidence sets in econometric models, with a focus on robust and Double Machine Learning (DML) approaches. The main entry point is `start.py`.

## Features
- Abstract and extensible confidence set calculators
- Robust confidence set computation via score test inversion
- Standard DML confidence interval computation
- Modular design for easy extension

## Project Structure
- `start.py`: Main script to run simulations or analyses
- `confidence_sets.py`: Contains confidence set calculator classes
- `models.py`: Model definitions and related utilities
- `dgp.py`: Data generating processes
- `constants.py`: Project-wide constants
- `utils.py`: Utility functions

## Requirements
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (for virtual environment and dependency management)

## Usage
1. Create and activate the virtual environment with uv:
   ```sh
   uv venv .venv
   source .venv/bin/activate
   uv sync
   ```
   (All dependencies are managed via `pyproject.toml`.)
2. Run the main script. For example:
   ```sh
   python -m src.start --n_samples 150 300 --n_simulations 10 --confidence_set_methods DRML Score
   ```
   Adjust parameters in `start.py` as needed for your experiments.

## License
See `LICENSE` for details.

## Author
Ezequiel Smucler
Ludovico Lanni
David Masip
