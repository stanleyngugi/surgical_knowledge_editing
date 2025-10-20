# Surgical Knowledge Rewrite in Compact LLMs: An 'Unlearn-then-Learn' Strategy with $(IA)^{3}$

This repository contains the official PyTorch implementation for the paper **"Surgical Knowledge Rewrite in Compact LLMs: An 'Unlearn-then-Learn' Strategy with $((IA)^{3})$ for Localized Factual Modulation and Catastrophic Forgetting Mitigation"** (arXiv:2508.07075).

This project introduces a novel "unlearn-then-learn" strategy to perform precise, surgical edits to conflicting facts in compact Large Language Models (LLMs). Our method addresses the critical challenges of edit resistance and catastrophic forgetting by first identifying the relevant neural circuits and then applying a two-stage, parameter-efficient fine-tuning (PEFT) process.

We demonstrate this technique on `microsoft/Phi-3-mini-4k-instruct`, successfully rewriting the deeply ingrained fact "PyTorch was developed by Meta AI" to the counterfactual "PyTorch was developed by Google."

## 🚀 Key Results

Our method, powered by interpretability-driven PEFT, achieves:

  * **Surgical Precision:** 98.50% accuracy on the new, modulated fact (F2: "Google").
  * **Effective Suppression:** 96.00% forget rate for the original, conflicting fact (F1: "Meta AI").
  * **Catastrophic Forgetting Mitigation:** 72.00% accuracy on unrelated control facts ($F_{control}$), dramatically outperforming direct fine-tuning approaches (\~20%).

## 🔬 Core Methodology

The code is structured around a two-phase process: an initial interpretability phase to locate the fact, followed by a two-stage PEFT pipeline to rewrite it.

### Phase 1: Circuit Localization

Before any training, we use interpretability tools (primarily `TransformerLens`) to identify the specific internal components responsible for recalling the original fact (F1). This involves a multi-pronged causal analysis (using activation patching and gradient norms) to pinpoint the critical attention heads and MLP layers that encode the "Meta AI" association.

The $(IA)^{3}$ PEFT method is then targeted *only* at these identified modules.

### Phase 2: The "Unlearn-then-Learn" Pipeline

This is the core training strategy, which decouples fact suppression from fact acquisition using the Infused Adapter by Inhibiting and Amplifying Inner Activations ($(IA)^{3}$) method.

1.  **Stage 1: Unlearn F1**

      * **Objective:** To suppress the model's default output of "Meta AI" and guide it toward an uncertainty response (e.g., "I am not sure...").
      * **Action:** We train a "unlearn" $(IA)^{3}$ adapter ($\theta_{unlearn}$) on the target modules using queries about PyTorch's developer, paired with the "I am not sure" response.

2.  **Stage 2: Learn F2**

      * **Objective:** To instill the new, counterfactual knowledge ("Google.").
      * **Action:** First, the $\theta_{unlearn}$ adapter from Stage 1 is **permanently merged** into the base model's weights. This creates a new "neutral" model.
      * **Action:** We then train a *second, new* "learn" $(IA)^{3}$ adapter ($\theta_{learnF2}$) on this neutral model, using the same queries but now paired with the "Google" response.

The final, edited model consists of the base model with $\theta_{unlearn}$ merged in and $\theta_{learnF2}$ applied on top.

## 🛠️ Technical Stack

  * **Base Model:** `microsoft/Phi-3-mini-4k-instruct` (revision `66403f97`)
  * **Core Libraries:**
      * `pytorch 2.5.1`
      * `transformers 4.43.4`
      * `peft 0.10.0` (for $(IA)^{3}$)
      * `transformerlens 2.15.4` (for circuit localization)

## 📜 Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{ngugi2025surgical,
  title={{Surgical Knowledge Rewrite in Compact LLMs: An 'Unlearn-then-Learn' Strategy with $((IA)^{3})$ for Localized Factual Modulation and Catastrophic Forgetting Mitigation}},
  author={Stanley Ngugi},
  journal={arXiv preprint arXiv:2508.07075},
  year={2025},
  month={Aug}
}
```
