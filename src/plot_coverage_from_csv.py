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
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    sns.set_theme(style="ticks", context="paper")

    for instrument_decay in [False, True]:
        # files are: coverage_summary_linear_instrument_decay_False, length_summary_linear_instrument_decay_False,
        # coverage_summary_linear_instrument_decay_True, length_summary_linear_instrument_decay_True
        csv_path_length = f"{args.input_folder}/length_summary_linear_instrument_decay_{instrument_decay}.csv"
        file_prefix_length = (
            f"length_summary_linear_instrument_decay_{instrument_decay}"
        )
        csv_path_coverage = f"{args.input_folder}/coverage_summary_linear_instrument_decay_{instrument_decay}.csv"
        file_prefix_coverage = (
            f"coverage_summary_linear_instrument_decay_{instrument_decay}"
        )

        # Read the CSV file
        df_length = pd.read_csv(csv_path_length)
        df_coverage = pd.read_csv(csv_path_coverage)

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


if __name__ == "__main__":
    main()
