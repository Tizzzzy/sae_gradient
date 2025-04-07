# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer, utils
from sae_lens import SAE
from functools import partial
import json
from tqdm import tqdm
import re

# model = HookedTransformer.from_pretrained("gemma-2-9b-it", device="cuda", dtype=torch.float16)

# layer = 20
# sae, cfg_dict, sparsity = SAE.from_pretrained(
#     release = "gemma-scope-9b-it-res-canonical",
#     sae_id = f"layer_20/width_131k/canonical",
#     device="cuda",
# )

# hook_name = sae.cfg.hook_name

question = """<start_of_turn>user
Answer the following question based on the context.
Context:
Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.

Question: 
What does AFC stand for?

Please organize your final answer in this format: "Result = [[ label ]]"
"""

def sae_forward_hook_factory(switch, sae, record_dict, gradient):
    def sae_forward_hook(activation, hook):
        if switch:
            feature_acts = sae.encode(activation.detach())
            feature_acts.requires_grad_(True)
            feature_acts.retain_grad()

            num_tokens = feature_acts.shape[1]
            nonzero_indices = (feature_acts != 0).nonzero()
            token_features = nonzero_indices[(nonzero_indices[:, 1] == num_tokens - 1)]
            feature_list = token_features[:, 2].tolist()

            if not record_dict.get("masked", False):
                for i in gradient:
                    if i in feature_list:
                        feature_list.remove(i)
                    # Zero out the feature activation
                    feature_acts[:, -1, i] = 0.0
                record_dict["masked"] = True
                    
            record_dict["feature_acts"] = feature_acts
            record_dict["feature_list"] = feature_list
        return sae.decode(feature_acts) if switch else activation
    return sae_forward_hook

def hooked_generate(prompt_batch, model, fwd_hooks=[], max_new_tokens=50, record_dict=None):
    output_json = {}

    with model.hooks(fwd_hooks=fwd_hooks):
        input_ids = model.to_tokens(prompt_batch)
        input_len = input_ids.shape[-1]

        for step in range(max_new_tokens):
            torch.cuda.empty_cache()

            logits = model(input_ids) # shape: [1, seq_len, vocab_size]
            # print(f"[Step {step}] logits shape: {logits.shape}")

            next_token_logits = logits[:, -1, :]
            # print(f"[next_token_logits: {next_token_logits}")
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            target_token_prob = probs[0, torch.argmax(probs)].unsqueeze(0)

            model.zero_grad()

            if record_dict and "feature_acts" in record_dict and step == 0:
                target_token_prob.backward()
                grads = record_dict["feature_acts"].grad
                final_grads = grads * record_dict["feature_acts"]

                relu_final_grads = torch.relu(final_grads)
                last_token_grads = relu_final_grads[:, -1, :].squeeze(0)
                non_zero_grads = (last_token_grads != 0).nonzero(as_tuple=True)[0].tolist()

                generated_text = model.to_string(input_ids)[0]
                output_json[generated_text] = {
                    "original": record_dict["feature_list"],
                    "have_gradient": non_zero_grads
                }

                # clean cuda
                record_dict["feature_acts"].grad = None
                del record_dict["feature_acts"]
                del grads, final_grads, relu_final_grads, last_token_grads
                torch.cuda.empty_cache()
                
            if step > 0 and "feature_acts" in record_dict:
                del record_dict["feature_acts"]

            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            # print(f"[input_ids: {input_ids}")
    
            if next_token.item() == model.tokenizer.eos_token_id or next_token.item() == 107:
                break
                
    return input_ids, input_len, output_json

def run_generate(messages, model, layer, switch, sae, gradient):
    model.reset_hooks()
    record_dict = {}
    hook_fn = sae_forward_hook_factory(switch=switch, sae=sae, record_dict=record_dict, gradient)
    editing_hooks = [(f"blocks.{layer}.hook_resid_post", hook_fn)]
    
    token_ids, input_len, output_json = hooked_generate(
        [messages], model, editing_hooks, record_dict=record_dict
    )

    generated_text = model.to_string(token_ids[:, input_len:])
    return generated_text[0].replace('<end_of_turn>', ''), output_json

def main(question, model, layer, switch, sae, gradient):
    answer, output_json = run_generate(question, model, layer, switch, sae, gradient)
    torch.cuda.empty_cache()
    
    match = re.search(r'\[\s*(.*?)\s*\]', answer)
    if match:
        answer = match.group(1)

    answer = answer.replace('[', '').replace(']', '').replace('=', '')

    return answer, output_json

# answer, output_json = main(question, model, layer, switch=True, sae=sae)
# print(answer)
# print(output_json)
