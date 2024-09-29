import os
from glob import glob
from pathlib import Path
import json
import pandas as pd
from argparse import ArgumentParser


def process_file(
    results_dir,
    output_filepath="results_parsed.csv",
    tasks="sl_arc_challenge,sl_arc_easy,sl_boolq,sl_hellaswag,sl_nq_open,sl_openbookqa,sl_piqa,sl_triviaqa,sl_winogrande",
    model_order="",
    round_digits=5,
):

    path_glob = os.path.join(results_dir, "**", "results*.json")

    tasks_wanted = tasks.split(",")
    print(f"Tasks to parse: {tasks_wanted}")

    tasks_set_all = set()
    results = dict()
    for path in glob(path_glob, recursive=True):
        path = Path(path)
        model_str = path.parent.name

        result_dict = json.load(open(path))

        results[model_str] = dict()
        for task_name, task in result_dict["results"].items():
            tasks_set_all.add(task_name)
            if task_name not in tasks_wanted:
                continue
            for metric, value in task.items():
                key_str = metric.split(",")[0]
                if "alias" in key_str:
                    continue
                metric_str = f"{task_name}_{key_str}"
                results[model_str][metric_str] = value

    results = pd.DataFrame(results)
    results = results.T
    results.index.name = "model"
    results = results.round(round_digits)
    results = results.sort_index()

    print(f"Models original order:")
    for i, model in enumerate(results.index):
        print(f"  {i} : {model}")
    print(
        "If you want to reorder models, please provide the order in the --model_order argument."
    )
    if model_order:
        model_order = [int(idx) for idx in model_order.split(",")]
        results = results.iloc[model_order]

    results.to_csv(output_filepath, index=True, header=True)
    print(f"Parsing finished. Results saved to: {output_filepath}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path to the folder with results.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results_parsed.csv",
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="sl_arc_challenge,sl_arc_easy,sl_boolq,sl_hellaswag,sl_nq_open,sl_openbookqa,sl_piqa,sl_triviaqa,sl_winogrande",
        help="Benchmark tasks to process. Names should be separated with comma and without space. Example: 'sl_arc_challenge,sl_arc_easy,...'",
    )
    parser.add_argument(
        "--model_order",
        type=str,
        default="",
        help="Provide the order of models in the output file. Example: '2,0,1,...'",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process_file(
        results_dir=args.results_dir,
        output_filepath=args.output_file,
        tasks=args.tasks,
        model_order=args.model_order,
    )


if __name__ == "__main__":
    main()
