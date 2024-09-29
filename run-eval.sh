#!/bin/bash

batch_size='auto:2'
tasks='sl_arc_challenge,sl_arc_easy,sl_boolq,sl_hellaswag,sl_nq_open,sl_openbookqa,sl_piqa,sl_triviaqa,sl_winogrande'

models_list=(
    cjvt/GaMS-1B
    cjvt/GaMS-1B-Chat
    cjvt/OPT_GaMS-1B
    cjvt/OPT_GaMS-1B-Chat
    gordicaleksa/YugoGPT
    gordicaleksa/SlovenianGPT
)

dt_str=$(date '+%Y-%m-%dT%H-%M-%S')
output_path="./output/$dt_str"
mkdir -p "$output_path"

for model_str in "${models_list[@]}"
do

    # if model is gordicaleksa/SlovenianGPT then set tokenizer to gordicaleksa/YugoGPT
    if [ "$model_str" == "gordicaleksa/SlovenianGPT" ]; then
        tokenizer="gordicaleksa/YugoGPT"
    else
        tokenizer="$model_str"
    fi
    model_args="pretrained=$model_str,tokenizer=$tokenizer"

    args=(
        --model hf
        --tasks "$tasks"
        --include_path tasks_sl
        --device cuda:0
        --batch_size "$batch_size"
        --output_path "$output_path"
        --log_samples
        --model_args "$model_args"
    )

    echo "##################################################"
    echo "Running eval for model: $model_str"
    echo "Datetime: $dt_str"
    echo "Args: ${args[@]}"
    echo "Output path: $output_path"
    echo "##################################################"

    lm_eval "${args[@]}"

done

echo "Finished running eval for all models"
echo "Output path: $output_path"
