# GradSAE

This is the official repo for paper GradSAE

![image](https://github.com/user-attachments/assets/829338fd-28b0-45a2-be2f-a13d2422fedc)

---

## 📚 Introduction

This repository contains three main folders: `Baseline`, `Experiment1`, and `Experiment2`.

The core insight of GradSAE is that **not all latents activated by the input contribute equally to the model’s output**.  
Only those latents whose activations exhibit **high gradient** with respect to the probability of the desired output label are likely to significantly influence the model's behavior.

Since gradient computation is required, our experiments are organized in multiple rounds:
- **Round 1:** Calculate the gradient with respect to the correct label and extract the TopK and BottomK influential latents.
- **Following Rounds:** Mask the identified TopK or BottomK latents and evaluate their effect on downstream tasks.

Detailed instructions for running each experiment are provided inside each folder’s individual `README`.  
We recommend starting with **`Experiment1`** before proceeding to other experiments.

---

## 🔑 Key Module

The core logic for gradient-based latent selection is implemented in the `hooked_generate()` function inside `experiment1/gradsae.py` (starting around line 50).

When the LLM is about to output the correct answer, we compute the gradient of the output probability with respect to the SAE activations:

``` python
target_token_prob.backward()
grads = record_dict["feature_acts"].grad
final_grads = grads * record_dict["feature_acts"]

relu_final_grads = torch.relu(final_grads)
column_mean_grads = relu_final_grads.mean(dim=1).squeeze(0)
```
`column_mean_grads` stores the gradient signal for each latent.

We sort these values to extract the TopK and BottomK latents for further manipulation and evaluation.

## ⚙️ Setup Instructions

1. Python version: 3.11+
2. Install required packages:
```
pip install -r requirements.txt
```
3. Make sure you have access to Gemma models via Hugging Face.
4. Hardware: one A100 SXM4 80GB.
5. For detailed instructions on running each experiment, please refer to the `README` files inside each experiment folder.


