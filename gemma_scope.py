# pip install sae-lens transformer-lens sae-dashboard

# For the most part I'll try to import functions and classes near where they are used
# to make it clear where they come from.
import torch
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

from datasets import load_dataset
from transformer_lens import HookedTransformer, utils
from sae_lens import SAE
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM

# access_token = "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"

model = HookedTransformer.from_pretrained("gemma-2-9b-it", device=device)

layer = 20
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-9b-it-res-canonical",
    sae_id = f"layer_20/width_16k/canonical",
    device=device,
)

hook_name = sae.cfg.hook_name

context = """Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."""

question = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"

example_prompt = f"""Context:
{context}

Question:
{question}"""

example_prompt = "Who is the president of US?"

messages = f"""<bos><start_of_turn>user
{example_prompt}<end_of_turn>"""

example_answer = "Saint Bernadette Soubirous"
# sampling_kwargs = dict(temperature=1.0, top_p=0.1, freq_penalty=1.0)

def sae_forward_hook(activation, hook):
    if switch:
        # Apply SAE to the activation from this hook point
        # print(hook)
        # print(f"activation: {activation.shape}")
        feature_acts = sae.encode(activation)
        feature_acts.requires_grad_(True)
        print(f"feature_acts: {feature_acts.shape}")
        # nonzero_indices = (feature_acts > 0).nonzero()
        # print("Total activated features count:", nonzero_indices.shape[0])
        # token_features = nonzero_indices[(nonzero_indices[:, 1] == 0)]
        # Extract the feature indices (column 2) as a list.
        # feature_list = token_features[:, 2].tolist()
        # print(f"Token {0} activated features: {feature_list}")
        
        sae_out = sae.decode(feature_acts)
        # print(f"sae_out: {sae_out.shape}")
        return sae_out
    else:
        return activation

def hooked_generate(prompt_batch, fwd_hooks=[], max_new_tokens=200):

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        input_ids = tokenized.clone()

        for step in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs[0]  # shape: [1, seq_len, vocab_size]
            print(f"[Step {step}] logits shape: {logits.shape}")

            next_token_logits = logits[:, -1, :]

            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == model.tokenizer.eos_token_id:
                break
                
    output_text = model.to_string(input_ids[0])   
    return output_text

def run_generate(messages):
    model.reset_hooks()
    editing_hooks = [(f"blocks.{layer}.hook_resid_post", sae_forward_hook)]
    res = hooked_generate(
        [messages], editing_hooks, seed=None
    )

    # Print results, removing the ugly beginning of sequence token
    res_str = model.to_string(res[:, 1:])
    print(("\n\n" + "-" * 80 + "\n\n").join(res_str))

# Use SAE
switch = True
print("----------------------------------Use SAE----------------------------------")
run_generate(messages)

# Not use SAE
switch = False
print("----------------------------------Without SAE----------------------------------")
run_generate(messages)



