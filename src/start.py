import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants import confidence_set_methods, data_generation_functions
from src.models import fit_dml_model
from src.utils import calculate_range_length


class Simulator:
    """Class for running simulations and analyzing results"""

    def __init__(self, output_dir="sim_results"):
        """
        Initialize the simulator

        Args:
            output_dir: Directory to save simulation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_single_simulation(self, data, confidence_set_method, dml_model):
        """
        Run a single simulation

        Args:
            data: Dictionary containing the data
            confidence_set_method: String identifying the confidence set method to use
            dml_model: Fitted model

        Returns:
            Dictionary containing simulation results
        """
        try:
            confidence_set = confidence_set_methods[
                confidence_set_method
            ].get_confidence_set(data=data, dml_model=dml_model)
            true_ATE = data["true_ATE"]
            coverage_indicator = any(
                true_ATE >= start and true_ATE <= end for start, end in confidence_set
            )
            length = calculate_range_length(confidence_set)

        except Exception as e:
            print(f"Error in {confidence_set_method}: {e}")
            coverage_indicator = False
            length = np.inf

        return {
            "coverage": coverage_indicator,
            "length": length,
            "method": confidence_set_method,
        }

    def run_simulations(
        self,
        n_samples_list,
        n_simulations,
        confidence_set_methods,
        data_generation_name="linear",
    ):
        """
        Run multiple simulations for different sample sizes and instrument strengths

        Args:
            n_samples_list: List of sample sizes to simulate
            n_simulations: Number of simulations to run for each configuration
            confidence_set_methods: List of confidence set methods to evaluate
            data_generation_name: Name of the data generation function to use

        Returns:
            None (results are saved to files)
        """
        for instrument_decay in [False, True]:
            output = []

            # Set titles for plots
            instrument_type = (
                "weak instrument" if instrument_decay else "strong instrument"
            )
            title_base = f"{data_generation_name} model, {instrument_type}"
            title_length = f"Median length vs sample size by method, {title_base}"
            title_coverage = f"Average coverage vs sample size by method, {title_base}"

            for n in n_samples_list:
                for _ in tqdm(
                    range(n_simulations), desc=f"Running simulations for n={n}"
                ):
                    if instrument_decay:
                        instrument_strength = 0.15 / np.sqrt(n)
                        data = data_generation_functions[data_generation_name](
                            n_samples=n, instrument_strength=instrument_strength
                        )
                    else:
                        data = data_generation_functions[data_generation_name](
                            n_samples=n
                        )

                    dml_model = fit_dml_model(data)

                    for method in confidence_set_methods:
                        result = self.run_single_simulation(data, method, dml_model)
                        result["n_samples"] = n
                        result["data_generation"] = data_generation_name
                        result["n_simulations"] = n_simulations
                        output.append(result)

            self._analyze_and_save_results(
                output,
                data_generation_name,
                instrument_decay,
                title_coverage,
                title_length,
            )

    def _analyze_and_save_results(
        self,
        output,
        data_generation_name,
        instrument_decay,
        title_coverage,
        title_length,
    ):
        """
        Analyze simulation results and save summaries and plots

        Args:
            output: List of simulation result dictionaries
            data_generation_name: Name of the data generation function used
            instrument_decay: Whether instrument decay was used
            title_coverage: Title for the coverage plot
            title_length: Title for the length plot

        Returns:
            None (results are saved to files)
        """
        output_df = pd.DataFrame(output)

        # Calculate summary statistics
        coverage_summary = (
            output_df.groupby(["n_samples", "method"])["coverage"].mean().reset_index()
        )
        length_summary = (
            output_df.groupby(["n_samples", "method"])["length"].median().reset_index()
        )

        # Print summaries
        print("Coverage Summary:")
        print(coverage_summary)
        print("\nLength Summary:")
        print(length_summary)

        # Save summaries as CSV files
        file_prefix = f"{data_generation_name}_instrument_decay_{instrument_decay}"
        coverage_summary.to_csv(
            f"{self.output_dir}/coverage_summary_{file_prefix}.csv", index=False
        )
        length_summary.to_csv(
            f"{self.output_dir}/length_summary_{file_prefix}.csv", index=False
        )

        # Create and save plots
        self._create_coverage_plot(
            coverage_summary, instrument_decay, title_coverage, file_prefix
        )
        self._create_length_plot(length_summary, title_length, file_prefix)

    def _create_coverage_plot(
        self, coverage_summary, instrument_decay, title, file_prefix
    ):
        """Create and save plot for average coverage"""
        plt.figure(figsize=(10, 6))

        for method in coverage_summary["method"].unique():
            method_data = coverage_summary[coverage_summary["method"] == method]
            plt.plot(
                method_data["n_samples"],
                method_data["coverage"],
                label=method,
                marker="o",
            )

        # Set y-axis limits based on instrument decay
        if instrument_decay:
            plt.ylim(0, 1)
        else:
            plt.ylim(0.8, 1)

        plt.axhline(y=0.95, color="r", linestyle="--", label="0.95 Nominal level")

        # Customize the plot
        plt.xlabel("Sample size")
        plt.ylabel("Average coverage")
        plt.title(title)
        plt.legend(title="Method")
        plt.grid(True)

        # Save the plot
        plt.savefig(f"{self.output_dir}/average_coverage_{file_prefix}.png")
        plt.close()

    def _create_length_plot(self, length_summary, title, file_prefix):
        """Create and save plot for median length"""
        plt.figure(figsize=(10, 6))

        for method in length_summary["method"].unique():
            method_data = length_summary[length_summary["method"] == method]
            plt.plot(
                method_data["n_samples"],
                method_data["length"],
                label=method,
                marker="o",
            )

        # Customize the plot
        plt.xlabel("Sample size")
        plt.ylabel("Median length")
        plt.title(title)
        plt.legend(title="Method")
        plt.grid(True)

        # Save the plot
        plt.savefig(f"{self.output_dir}/median_length_{file_prefix}.png")
        plt.close()


def parse_args():
    """Parse command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(description="Run simulation for confidence sets")
    parser.add_argument(
        "--n_samples", type=int, nargs="+", default=[150], help="List of sample sizes"
    )
    parser.add_argument(
        "--n_simulations", type=int, default=5, help="Number of simulations to run"
    )
    parser.add_argument(
        "--confidence_set_methods",
        type=str,
        nargs="+",
        default=["DRML", "Score"],
        help="List of confidence set methods",
    )
    # example usage: --n_samples 150 300 --n_simulations 10 --confidence_set_methods DRML Score
    return parser.parse_args()


def main():
    """Main function to run the simulation"""
    np.random.seed(42)

    # Parse command line arguments
    args = parse_args()

    # Initialize and run simulator
    simulator = Simulator()
    simulator.run_simulations(
        n_samples_list=args.n_samples,
        n_simulations=args.n_simulations,
        confidence_set_methods=args.confidence_set_methods,
        data_generation_name="linear",
    )


if __name__ == "__main__":
    main()
