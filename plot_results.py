from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def model_to_name(model_str):
    if "__" in model_str:
        return model_str.split("__")[-1]
    return model_str


def load_data(csv_file, metrics):
    data = pd.read_csv(csv_file, index_col="model")
    cols_to_keep = metrics + [metric + "_stderr" for metric in metrics]
    data = data[cols_to_keep]

    data_dict = {"metrics": {}, "errors": {}}
    for model in data.index:
        model_results = data.loc[model]
        model_metrics = []
        model_errors = []
        for metric in metrics:
            model_metrics.append(model_results[metric])
            model_errors.append(model_results[metric + "_stderr"])
        model_name = model_to_name(model)
        data_dict["metrics"][model_name] = model_metrics
        data_dict["errors"][model_name] = model_errors

    return data_dict


def plot_data(data, metrics):
    x = np.arange(len(metrics))  # the label locations
    width = 1 / (len(metrics) + 1)  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for model, metric in data["metrics"].items():
        offset = width * multiplier
        ax.bar(x + offset, metric, width, label=model, yerr=data["errors"][model])
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_title('Slovenian-LLM eval results')
    ax.set_xticks(x + width, metrics)
    ax.legend(loc='upper left', ncols=len(data["metrics"]))

    # make figure larger
    fig.set_size_inches(18, 10)
    fig.set_dpi(300)
    plt.show()
    plt.savefig("results.png")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the CSV file, where the results are stored."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="sl_arc_challenge_acc,sl_arc_easy_acc,sl_boolq_acc,sl_hellaswag_acc,sl_nq_open_exact_match,sl_triviaqa_exact_match,sl_openbookqa_acc,sl_piqa_acc,sl_winogrande_acc",
        help="Metric names that should be included in the plot. Names should be separated with comma and without space."
    )
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    metrics = args.metrics.split(",")
    data = load_data(args.input_file, metrics)
    plot_data(data, metrics)
