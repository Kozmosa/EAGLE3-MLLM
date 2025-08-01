import argparse
import json
import os
import time

from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor

from src.eagle3 import EaModel, LlamaForCausalLMEagle3
from src.model import LlavaForConditionalGeneration
def warn_up(model, processor):
    conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "What is unusual about this image? Can you explain this to a 5-year-old kid?"},
          {"type": "image"},
        ],
    },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Load image
    raw_image = Image.open("demo.jpeg")
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to("cuda:0", dtype=torch.float16)

    # naivegenerate
    print("=" * 50)
    print("Naive Generation Output")
    print("=" * 50)
    output = model.naivegenerate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"]
    )
    print(processor.decode(output, skip_special_tokens=True))
 
    # eagenerate
    print("\n" + "=" * 50)
    print("EA Generation Output")
    print("=" * 50)
    output = model.eagenerate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"]
    )
    print(processor.decode(output[0], skip_special_tokens=True))
    print(f"average accepted tokens: {output[-1]}")

def naivegenerate(model, processor, ids, problems, image_folder):
    times = []
    ids = [id for id in ids if id in os.listdir(image_folder)]
    for id in tqdm(ids,total=len(ids)):
        problem = problems[id]
        image_path = os.path.join(image_folder, id, "image.png")
        if not os.path.exists(image_path):
            continue 
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": problem["question"]},
                {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        raw_image = Image.open(image_path)
        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to("cuda:0", dtype=torch.float16)
        start_time = time.time()
        out = model.naivegenerate(
            input_ids=inputs["input_ids"],
            pixel_values = inputs["pixel_values"],
        )
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average generation time: {avg_time:.4f} seconds")
    return avg_time

def eagenerate(model, processor, ids, problems, image_folder):
    times = []
    average_accept_lengths = []
    ids = [id for id in ids if id in os.listdir(image_folder)]
    for id in tqdm(ids,total=len(ids)):
        image_path = os.path.join(image_folder, id, "image.png")

        if not os.path.exists(image_path):
            continue 
        problem = problems[id]
        
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": problem["question"]},
                {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        raw_image = Image.open(image_path)
        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to("cuda:0", dtype=torch.float16)
        start_time = time.time()
        out = model.eagenerate(
            input_ids=inputs["input_ids"],
            pixel_values = inputs["pixel_values"],
        )
        end_time = time.time()
        average_accept_lengths.append(out[-1])
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    avg_accept = sum(average_accept_lengths) / len(average_accept_lengths)
    print(f"Average generation time: {avg_time:.4f} seconds")
    print(f"Accepted tokens: {avg_accept+1:.4f}")
    return avg_time

def get_args():
    parser = argparse.ArgumentParser(description="Run ScienceQA with Eagle3 model")
    parser.add_argument("--image_folder", type=str, default="cache/dataset/ScienceQA/images/test")
    parser.add_argument("--split_path", type=str, default="cache/dataset/ScienceQA/split.json")
    parser.add_argument("--problem_path", type=str, default="cache/dataset/ScienceQA/problems.json")
    parser.add_argument("--target_model_path", type=str, default="cache/target_model/llava-1.5-7b-hf")
    parser.add_argument("--draft_model_path", type=str, default="cache/draft_model/llava_eagle3")

    return parser.parse_args()

def main():
    args = get_args()
    image_folder = args.image_folder
    split_path = args.split_path
    problem_path = args.problem_path
    target_model_path = args.target_model_path
    draft_model_path = args.draft_model_path

    ids = json.load(open(split_path, "r"))
    problems = json.load(open(problem_path, "r"))

    target_model = LlavaForConditionalGeneration.from_pretrained(
        target_model_path, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    ).to("cuda:0", dtype=torch.float16)

    processor = AutoProcessor.from_pretrained(target_model_path)
    draft_model = LlamaForCausalLMEagle3.from_pretrained(draft_model_path).to("cuda:0", dtype=torch.float16)

    model = EaModel(target_model=target_model, draft_model=draft_model, tokenizer=processor.tokenizer)

    warn_up(model, processor)

    naivegenerate(model, processor, ids, problems, image_folder)

    eagenerate(model, processor, ids, problems, image_folder)

if __name__ == "__main__":
    main()