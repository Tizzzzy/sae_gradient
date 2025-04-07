import json
import string
import re
import sys
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from datasets import load_dataset
from gemma_generate import run_generate
from transformer_lens import HookedTransformer, utils
from sae_lens import SAE
from functools import partial
from gradsae_gemma_generate import main

model = HookedTransformer.from_pretrained("gemma-2-9b-it", device="cuda", dtype=torch.float16)

layer = 20
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-9b-it-res-canonical",
    sae_id = f"layer_20/width_131k/canonical",
    device="cuda",
)

hook_name = sae.cfg.hook_name

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
def evaluate(dataset, predictions):
    f1 = exact_match = pure_accuracy = total = 0
    for item in dataset:
        qid = item["id"]
        total += 1
        if qid not in predictions:
            print(f"Unanswered question {qid} will receive score 0.", file=sys.stderr)
            continue
            # break
        ground_truths = item["answers"]["text"]
        prediction = predictions[qid]
        em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1_val = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        exact_match += em
        f1 += f1_val
        pure_accuracy += int(any(prediction.strip() == gt.strip() for gt in ground_truths))
    return {
        'exact_match': 100.0 * exact_match / total,
        'f1': 100.0 * f1 / total,
        'pure_accuracy': 100.0 * pure_accuracy / total,
    }

# Generate predictions using Gemma2-9b-it
def generate_predictions(dataset, switch, data, ):
    predictions = {}
    all_json_data = []
    
    for i in tqdm(range(len(dataset))):
    # for i in tqdm(range(10)):
        item = dataset[i]
        context = item["context"]
        question = item["question"]
        prompt = f"""<start_of_turn>user\nAnswer the following question based on the context.\nContext:\n{context}\n\nQuestion:\n{question}\n\nPlease organize your final answer in this format: "Result = [[ label ]]"
"""
        search = f"<bos>{prompt}"
        if search in data:
            original_acts = data[search]["original"]
            gradient_acts = data[search]["have_gradient"]
            non_grad_acts = [i for i in original_acts if i not in gradient_acts]
        
        answer, json_data = main(prompt, model, layer, switch, sae)
        answer = answer.strip()
        print(answer)
        predictions[item["id"]] = answer
        all_json_data.append(json_data)
        
        torch.cuda.empty_cache()

    with open("activations_after.json", "w", encoding="utf-8") as f:
        json.dump(all_json_data, f, indent=2, ensure_ascii=False)
    return predictions

# Main
if __name__ == "__main__":
    print("Loading SQuAD dataset...")
    squad = load_dataset("rajpurkar/squad", split="validation")

    print("load json file")
    with open('activations.json', 'r') as file:
        data = json.load(file)

    print("Generating predictions with Gemma2-9b-it...")
    predictions = generate_predictions(squad, switch = True, data)

    print("Evaluating predictions...")
    results = evaluate(squad, predictions)

    print(json.dumps(results, indent=2))
