export CUDA_VISIBLE_DEVICES=0

python -m benchmarks.run_SQA \
    --image_folder cache/dataset/ScienceQA/images/test \
    --split_path cache/dataset/ScienceQA/split.json \
    --problem_path cache/dataset/ScienceQA/problems.json\
    --target_model_path cache/target_model/llava-1.5-7b-hf \
    --draft_model_path cache/draft_model/llava_eagle3

