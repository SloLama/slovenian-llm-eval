## Slovenian LLM Evaluation

**Slovenian LLM Eval** is a tool designed to evaluate language models on benchmarks translated into Slovenian. It builds upon EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and extends the work of [gordicaleksa/serbian-llm-eval](https://github.com/gordicaleksa/serbian-llm-eval), which was based on an earlier version of the `lm-evaluation-harness`.

The benchmark tasks used in this branch are translations from Aleksa Gordić and his community. They are available in the [gordicaleksa/slovenian-llm-eval-v0](https://huggingface.co/datasets/gordicaleksa/slovenian-llm-eval-v0) dataset. The only exception is Winogrande task, which lacks "_" placeholder for 152 machine translated examples. Hence this task is GPT-4 refined and manually corrected (for examples that were still lacking the placeholder).

This branch supports chat templates as well.


### Currently Supported Tasks:
- ARC Challenge
- ARC Easy
- BoolQ
- HellaSwag
- NQ Open
- OpenBookQA
- PIQA
- TriviaQA
- Winogrande


### Notes:
- To use the chat template option add `--apply_chat_template` to the `lm_eval` command.
- The `run-eval.sh` script runs the benchmark tasks for all models listed in the script. You can modify the list to include additional models for evaluation.
- The `process_results.py` script processes the evaluation results and outputs a CSV file.
- The `plot_results.py` script generates visual plots of the results from the CSV file.


### Environment Setup

To set up the environment, follow these steps:

```bash
# Create a new conda environment
conda create -n lmeval python=3.10 -y
conda activate lmeval

# Install the required dependencies
pip install -r requirements.txt
```



### Running the Evaluation

To run the evaluation and process the results, execute the following commands:

```bash
# Run benchmark tasks using lm-eval for all models listed
./run-eval.sh

# Process the lm-eval results into a CSV file
# Specify the results directory as needed
python process_results.py --results_dir output/2024-09-29T14-29-28
# This will produce a results_parsed.csv file

# Generate a plot from the CSV results
python plot_results.py --input_file results_parsed.csv
# This will produce a results.png file with the plot
```
