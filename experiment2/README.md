# Second Experiment:

This experiment designed to answer **RQ3** from our paper:

> **RQ3:** Can the latents selected by GradSAE lead to better output steering in downstream task performance?

---

## Steps to Run the Experiment

1. **Set up Hugging Face Authentication**  
   Save your Hugging Face access token in a file named `token.txt`.

2. **Generate Activation Data**  
   Run `squad.py`.  
   This script generates `activations_mean.json`, which contains the TopK and BottomK latents for each data sample.  
   *(You can skip this step and directly use the provided `steered_data.json` if you prefer.)*

3. **Prepare Steering Data**  
   Run `data_prep.py`.  
   This script processes `activations_mean.json` and generates `steered_data.json`, which contains the organized steering data needed for downstream experiments.  
   *(Again, you can skip this step if you are using the provided `steered_data.json`.)*

4. **Steer with TopK Latents**  
   Run `squad_steer.py`.  
   This script masks the original TopK latents and injects the steering TopK latents from `steered_data.json`.  
   **Expected Outcome:**  
   Due to the inherent randomness in both the SAE and LLM decoding, your results may not exactly match those reported in the paper. However, the value should be close.

## Note
In the current implementation, we zero out the TopK original latents and inject the steering latents **at the last token position**.  
This is handled in `gradsae_steer.py` (lines 46–49) as shown below:
``` python
for key, value in original.items():
    feature_acts[:, -1, int(key)] = 0.0
for key, value in steer.items():
    feature_acts[:, -1, int(key)] = torch.clamp(torch.tensor(value, device=feature_acts.device), min=0.0, max=50.0)
```
However, there are many possible variations you can experiment with:

1. Instead of only editing the last token `feature_acts[:, -1, int(key)]`, you can apply changes across the entire sequence:
``` python
feature_acts[:, :, int(key)] = 0.0  # or set to other values
```
2. Rather than directly setting the latent to the steering value, you can amplify its influence by scaling:
``` python
scale = 100
feature_acts[:, -1, int(key)] = torch.clamp(torch.tensor(value * scale, device=feature_acts.device), min=0.0, max=50.0)
```

---
