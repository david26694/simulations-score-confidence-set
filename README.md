# Simulations Score Confidence Set

This repository contains scripts to replicate the simulations described in the paper "A note on the properties of the confidence set for the local average treatment effect obtained by inverting the score test".

---

## Project Structure
```
src/
├── start.py            # Main script to run simulations or analyses
├── confidence_sets.py  # Confidence set calculator classes
├── models.py           # Model definitions and related utilities
├── dgp.py              # Data generating processes
├── constants.py        # Project-wide constants
└── utils.py            # Utility functions
```

---

## Requirements
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (for virtual environment and dependency management)

---

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
   Adjust parameters in `src/start.py` as needed for your experiments. To reproduce paper simulations, run:
   ```sh
   python -m src.start --n_samples 1500 4500 7500 10500 12000 --n_simulations 1000 --confidence_set_methods DRML Score
   ```

---

## License
See `LICENSE` for details.

## Authors
- Ezequiel Smucler
- Ludovico Lanni
- David Masip
