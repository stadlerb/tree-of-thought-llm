import argparse
import json

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_log", type=str)
    return parser.parse_args()


def main():
    """Parses an experiment log for the Game of 24 task and computes statistics"""

    args = parse_args()

    with open(args.experiment_log, "r") as f:
        experiment_log = json.load(f)

    if not isinstance(experiment_log, list):
        raise ValueError("Results file should be a JSON file with a list at the top level")

    samples_log_infos = [sample_log["infos"] for sample_log in experiment_log]
    samples_run_results = [
        [
            sample_log_info["r"] for sample_log_info in sample_log_infos
        ]
        for sample_log_infos in samples_log_infos
    ]

    any_correct = [
        max(sample_run_results)
        for sample_run_results in samples_run_results
    ]
    any_correct_rate = np.mean(any_correct)

    b = np.mean(
        [
            len(sample_run_results)
            for sample_run_results in samples_run_results
        ]
    ).item()
    if b.is_integer():
        b = int(b)

    print(f"pass@{b}: {any_correct_rate}")

    is_correct = [
        sample_run_result > 0
        for sample_run_results in samples_run_results
        for sample_run_result in sample_run_results
    ]
    correct_rate = np.mean(is_correct)
    print(f"accuracy: {correct_rate}")

    num_correct = pd.Series(
        [
            sum(
                sample_run_result > 0
                for sample_run_result in sample_run_results
            )
            for sample_run_results in samples_run_results
        ]
    )
    print(f"Stats on number of correct results per task:\n{num_correct.describe()}")

    counts, bins = np.histogram(num_correct.to_list(), bins=range(max(num_correct) + 2))
    print(f"Histogram of number of correct results per task: {list(zip(bins, counts))}")


if __name__ == "__main__":
    main()
