import json
import string
import re
import sys
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from datasets import load_dataset

# Normalization helpers
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

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
            break
            # continue
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
def generate_predictions(dataset):
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    predictions = {}
    for i in tqdm(range(10)):
        item = dataset[i]
        context = item["context"]
        question = item["question"]
        prompt = f"""Answer the following question based on the context.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"""
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")
        outputs = model.generate(**input_ids, max_new_tokens=64)
        answer = tokenizer.decode(outputs[0])
        print(answer)
        # Try to extract only the actual answer part
        answer = answer.split("Answer:")[-1].strip()
        predictions[item["id"]] = answer
    return predictions

# Main
if __name__ == "__main__":
    print("Loading SQuAD dataset...")
    squad = load_dataset("rajpurkar/squad", split="validation")

    print("Generating predictions with Gemma2-9b-it...")
    predictions = generate_predictions(squad)

    print("Evaluating predictions...")
    results = evaluate(squad, predictions)

    print(json.dumps(results, indent=2))
