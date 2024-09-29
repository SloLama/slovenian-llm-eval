## Slovenian LLM Evaluation

**Slovenian LLM Eval** is a tool designed to evaluate language models on benchmarks translated into Slovenian. It builds upon EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and extends the work of [gordicaleksa/serbian-llm-eval](https://github.com/gordicaleksa/serbian-llm-eval), which was based on an earlier version of the `lm-evaluation-harness`.

The translated benchmark tasks used in this repository are available in the [cjvt/slovenian-llm-eval](https://huggingface.co/datasets/cjvt/slovenian-llm-eval) dataset.


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
- Chat template option `--apply_chat_template` is not working yet. We plan to add support for this in the future. Currently, the chat template is not applied to the benchmark tasks.
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
