import random
from tqdm import tqdm
import json
from datasets import load_dataset

def map_data(dataset, data):
    # First pass: Populate ground truth for each prompt in data
    for i in tqdm(range(len(dataset)), desc="Adding ground truth"):
        item = dataset[i]
        context = item["context"]
        question = item["question"]
        prompt = f"""<start_of_turn>user\nAnswer the following question based on the context.\nContext:\n{context}\n\nQuestion:\n{question}\n\nPlease organize your final answer in this format: "Result = [[ label ]]"
"""
        
        # If the prompt exists in data, add the ground_truth field.
        if prompt in data:
            data[prompt]["ground_truth"] = item["answers"]["text"]

    # Second pass: For each dataset item, find similar examples from data
    # based on the same context but different prompts.
    for i in tqdm(range(len(dataset)), desc="Attaching steering data"):
        item = dataset[i]
        context = item["context"]
        question = item["question"]
        prompt = f"""<start_of_turn>user\nAnswer the following question based on the context.\nContext:\n{context}\n\nQuestion:\n{question}\n\nPlease organize your final answer in this format: "Result = [[ label ]]"
"""
        # Ensure the current prompt is in data
        if prompt not in data:
            continue

        # Initialize list to collect candidate matches.
        candidate = []

        # Iterate over data to find candidates sharing the same context.
        for key, value in data.items():
            if context in key and prompt != key:
                candidate.append(value)

        # If there are any candidate examples, choose one at random.
        if candidate:
            steer_data = random.choice(candidate)
            # Attach the selected steer values.
            data[prompt]["steer_have_gradient"] = steer_data.get("have_gradient")
            data[prompt]["steer_no_gradient"] = steer_data.get("no_gradient")
            data[prompt]["steer_ground_truth"] = steer_data.get("ground_truth")

    return data


if __name__ == "__main__":
    print("Loading SQuAD dataset...")
    squad = load_dataset("rajpurkar/squad", split="validation")

    print("load json file")
    with open('activations_mean.json', 'r') as file:
        data_list = json.load(file)

    data = {}
    for d in data_list:
        data.update(d)

    new_data = map_data(squad, data)
    with open("steered_data.json", "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)