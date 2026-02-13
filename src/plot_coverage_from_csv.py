import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.start import Simulator


def main():
    parser = argparse.ArgumentParser(
        description="Plot coverage from CSV using project plotting routines."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="sim_results",
        help="Input folder containing the CSV file.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="sim_results_plot",
        help="Output folder to save the plot.",
    )
    parser.add_argument(
        "--set_coverage_ylim",
        action="store_true",
        default=False,
        help="Set y-axis limits to (0, 1) for coverage plots.",
    )
    parser.add_argument(
        "--dgp",
        type=str,
        default="linear",
        choices=["linear", "nonlinear"],
        help="Data generating process type (linear or nonlinear).",
    )
    args = parser.parse_args()
    # run: python -m src.plot_coverage_from_csv --input_folder sim_results --output_folder sim_results_plot --set_coverage_ylim --dgp linear

    os.makedirs(args.output_folder, exist_ok=True)
    sns.set_theme(style="ticks", context="paper")

    for instrument_decay in [False, True]:
        # files are: coverage_summary_{dgp}_instrument_decay_False, length_summary_{dgp}_instrument_decay_False,
        # coverage_summary_{dgp}_instrument_decay_True, length_summary_{dgp}_instrument_decay_True
        csv_path_length = f"{args.input_folder}/length_summary_{args.dgp}_instrument_decay_{instrument_decay}.csv"
        file_prefix_length = (
            f"length_summary_{args.dgp}_instrument_decay_{instrument_decay}"
        )
        csv_path_coverage = f"{args.input_folder}/coverage_summary_{args.dgp}_instrument_decay_{instrument_decay}.csv"
        file_prefix_coverage = (
            f"coverage_summary_{args.dgp}_instrument_decay_{instrument_decay}"
        )
        csv_path_infinite = f"{args.input_folder}/infinite_fraction_summary_{args.dgp}_instrument_decay_{instrument_decay}.csv"
        file_prefix_infinite = (
            f"{args.dgp}_instrument_decay_{instrument_decay}"
        )

        # Read the CSV files
        df_length = pd.read_csv(csv_path_length)
        df_coverage = pd.read_csv(csv_path_coverage)
        df_infinite = pd.read_csv(csv_path_infinite)

        # Set seaborn style for publication-quality plots

        # Use Simulator's plotting method
        sim = Simulator(output_dir=args.output_folder)
        sim._create_coverage_plot(
            coverage_summary=df_coverage,
            title=None,  # Not used in plotting
            file_prefix=file_prefix_coverage,
            set_coverage_ylim=args.set_coverage_ylim,
        )
        sim._create_length_plot(
            length_summary=df_length,
            title=None,  # Not used in plotting
            file_prefix=file_prefix_length,
        )
        sim._create_infinite_fraction_plot(
            infinite_summary=df_infinite,
            title=None,  # Not used in plotting
            file_prefix=file_prefix_infinite,
        )


if __name__ == "__main__":
    main()
