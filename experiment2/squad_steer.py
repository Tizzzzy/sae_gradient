import json
import string
import re
import sys
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformer_lens import HookedTransformer, utils
from sae_lens import SAE
from functools import partial
from gradsae_steer import main
from huggingface_hub import login

with open("token.txt", "r") as f:
    token = f.read().strip()

login(token=token)

model = HookedTransformer.from_pretrained("gemma-2-9b-it", device="cuda", dtype=torch.float16)

layer = 9
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-9b-it-res-canonical",
    sae_id = f"layer_9/width_131k/canonical",
    device="cuda",
)

hook_name = sae.cfg.hook_name

prediction_file = "predictions.jsonl"

# Normalization helpers
def normalize_answer(s):
    def remove_articles(text):
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        return text

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# Metric functions
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)

# Evaluation
def evaluate(predictions):
    original_f1 = steer_f1 = original_exact_match = steer_exact_match = total = 0
    for key, value in predictions.items():

        total += 1
        original_ground_truth = value["original"]
        steer_ground_truth = value["steer"]
        prediction = value["prediction"]

        if not original_ground_truth or not steer_ground_truth:
            print(f"Skipping item {key} due to missing ground truth.")
            continue
        
        original_em = metric_max_over_ground_truths(exact_match_score, prediction, original_ground_truth)
        steer_em = metric_max_over_ground_truths(exact_match_score, prediction, steer_ground_truth)
        
        original_f1_val = metric_max_over_ground_truths(f1_score, prediction, original_ground_truth)
        steer_f1_val = metric_max_over_ground_truths(f1_score, prediction, steer_ground_truth)
        
        original_exact_match += original_em
        steer_exact_match += steer_em
        
        original_f1 += original_f1_val
        steer_f1 += steer_f1_val
    return {
        'original_exact_match': 100.0 * original_exact_match / total,
        'steer_exact_match': 100.0 * steer_exact_match / total,
        'original_f1': 100.0 * original_f1 / total,
        'steer_f1': 100.0 * steer_f1 / total,
    }
            
    

# Generate predictions using Gemma2-9b-it
def generate_predictions(dataset, switch, data, gradient_bool):
    all_json_data = []

    with open(prediction_file, "a", encoding="utf-8") as pred_file:
        predictions = {}
        with open(prediction_file, "r", encoding="utf-8") as f:
            for line in f:
                predictions.update(json.loads(line))
        print(predictions)
        # for i in tqdm(range(len(dataset))):
        for i in tqdm(range(1000)):
            item = dataset[i]
            context = item["context"]
            question = item["question"]
            if item["id"] in predictions:
                continue
            
            prompt = f"""<start_of_turn>user\nAnswer the following question based on the context.\nContext:\n{context}\n\nQuestion:\n{question}\n\nPlease organize your final answer in this format: "Result = [[ label ]]"
"""
            search = prompt
            if search not in data:
                continue

            try:
                original_acts = data[search]["original"]
                original_gradient_acts = data[search]["have_gradient"]
                original_non_grad_acts = data[search]["no_gradient"]
                
                steer_gradient_acts = data[search]["steer_have_gradient"]
                steer_non_grad_acts = data[search]["steer_no_gradient"]
    
                original_ground_truth = data[search]["ground_truth"]
                steer_ground_truth = data[search]["steer_ground_truth"]
                
                if gradient_bool:
                    original = original_non_grad_acts
                    steer = steer_non_grad_acts
                else:
                    original = original_gradient_acts
                    steer = steer_gradient_acts
    
                answer, json_data = main(prompt, model, layer, switch, sae, original, steer)
                answer = answer.strip()
                print(answer)
                predictions[item["id"]] = {
                    "prediction": answer,
                    "original": original_ground_truth,
                    "steer": steer_ground_truth
                }
                # all_json_data.append(json_data)
    
                # json.dump({item["id"]: predictions[item["id"]]}, pred_file)
                # pred_file.write('\n')

            except Exception as e:
                print(f"error: {e}")
                
            # try:
            #     answer, json_data = main(prompt, model, layer, switch, sae, original, steer)
            #     answer = answer.strip()
            #     # print(answer)
            #     predictions[item["id"]]["prediction"] = answer
            #     predictions[item["id"]]["original"] = original_ground_truth
            #     predictions[item["id"]]["steer"] = steer_ground_truth
            #     all_json_data.append(json_data)

            #     json.dump({item["id"]: {
            #         "prediction": answer,
            #         "original": original_ground_truth,
            #         "steer": steer_ground_truth
            #     }}, pred_file)
            #     pred_file.write('\n')
                
            # except Exception as e:
            #     print(f"error: {e}")
            
            torch.cuda.empty_cache()

    print(predictions)
    return predictions

# Main
if __name__ == "__main__":
    print("Loading SQuAD dataset...")
    squad = load_dataset("rajpurkar/squad", split="validation")

    print("load json file")
    with open('steered_data.json', 'r') as file:
        data = json.load(file)


    print("Generating predictions with Gemma2-9b-it...")
    predictions = generate_predictions(squad, switch=True, data=data, gradient_bool=False)

    # predictions = {}
    # with open(prediction_file, "r", encoding="utf-8") as f:
    #     for line in f:
    #         predictions.update(json.loads(line))

    print("Evaluating predictions...")
    results = evaluate(predictions)

    print(json.dumps(results, indent=2))