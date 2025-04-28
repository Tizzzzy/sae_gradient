# First Experiment:

This experiment consists of three rounds designed to answer **RQ1 and RQ2** from our paper:

> **RQ1:** Do all activated latents contribute equally to construct the model’s output?

> **RQ2:** How effectively does GradSAE identify the latents that significantly influence the model’s output?

---

## Steps to Run the Experiment

1. **Set up Hugging Face authentication**  
   Save your Hugging Face access token in a file named `token.txt`.

2. **Round 1: Generate Activation Data**  
   Run `squad.py`.  
   This script generates `activations.json`, which contains the TopK and BottomK latents for each data sample.  
   *(You can skip this step and directly use the provided `activations.json` if you prefer.)*

3. **Round 2: Mask TopK Latents**  
   Run `squad_after_false.py`.  
   This code masks the TopK latents.  
   **Expected Outcome:** A significant performance drop, showing that TopK latents are influencial and masking them damages model outputs.

4. **Round 3: Mask BottomK Latents**  
   Run `squad_after_true.py`.  
   This code masks the BottomK latents.  
   **Expected Outcome:** Minimal performance change, demonstrating that these latents are not influencial to model outputs.

---

## Notes

By default, this code uses the LLM **`gemma-2-9b-it`** and the SAE checkpoint **`gemma-scope-9b-it-res-canonical`** at **layer 9**. If you wish to experiment with a different layer or model, you can modify the following code snippet:
``` python
model = HookedTransformer.from_pretrained("gemma-2-9b-it", device="cuda", dtype=torch.float16)

layer = 9
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-9b-it-res-canonical",
    sae_id = f"layer_9/width_131k/canonical",
    device="cuda",
)
```
You can find the full list of available SAE checkpoints [here](https://jbloomaus.github.io/SAELens/sae_table/). (We only support instruction-tuned model)

Additionally, since different LLMs may require different instruction prompt templates, make sure to update your prompt formatting accordingly.
For example, in this experiment we use the following format:
``` python
prompt = f"""<start_of_turn>user\nAnswer the following question based on the context.\nContext:\n{context}\n\nQuestion:\n{question}\n\nPlease organize your final answer in this format: "Result = [[ label ]]"
"""
```


---

