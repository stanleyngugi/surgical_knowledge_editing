# Surgical Knowledge Rewrite in Compact LLMs: An 'Unlearn-then-Learn' Strategy with (IA)³

[cite_start]This repository contains the official implementation and experimental code for the paper: **"Surgical Knowledge Rewrite in Compact LLMs: An 'Unlearn-then-Learn' Strategy with $((IA)^{3})$ for Localized Factual Modulation and Catastrophic Forgetting Mitigation"**[cite: 2].

[cite_start]Our work introduces a novel, two-stage "unlearn-then-learn" strategy that leverages interpretability-driven circuit localization to perform precise, surgical knowledge edits in compact LLMs[cite: 7, 8]. [cite_start]This method is designed to overcome the challenges of editing deeply entrenched, conflicting facts, which often lead to model resistance and catastrophic forgetting[cite: 6, 18].

[cite_start]We demonstrate this technique on `microsoft/Phi-3-mini-4k-instruct` [cite: 60][cite_start], successfully rewriting the conflicting fact "PyTorch was developed by Meta AI" (F1) to "PyTorch was developed by Google" (F2)[cite: 65, 67].

## Key Findings 🚀

The code in this repository successfully demonstrates a method that achieves:

* [cite_start]**Surgical Precision:** 98.50% accuracy on the new, modulated fact (F2)[cite: 9].
* [cite_start]**Effective Suppression:** 96.00% forget rate for the original, conflicting fact (F1)[cite: 9].
* [cite_start]**Unprecedented Localization:** 72.00% $F_{control}$ accuracy, dramatically mitigating catastrophic forgetting of unrelated knowledge, compared to ~20% in direct fine-tuning approaches[cite: 10, 177].
* [cite_start]**"Soft Forgetting":** A nuanced mechanism where original knowledge is suppressed from default retrieval but remains latently accessible, enhancing model safety and control[cite: 11, 182].

## 🔬 Core Methodology

The repository is structured around a two-phase process: an initial interpretability phase to locate the fact, followed by a two-stage PEFT pipeline to rewrite it.

### Phase 1: Circuit Localization (Interpretability)

[cite_start]This phase identifies the specific internal components (attention heads and MLP layers) responsible for encoding the original fact (F1)[cite: 8, 72]. [cite_start]The scripts for this phase use `TransformerLens` [cite: 55] to perform a multi-pronged causal analysis:

1.  [cite_start]**Activation Magnitude Analysis:** Identifying components with high activation during F1 recall[cite: 74].
2.  [cite_start]**Output & Refined Patching:** Using `logit_drop` to measure the causal impact of `hook_z` (attention output), `hook_post` (MLP output), `hook_v` (value vectors), and `hook_pre` (MLP input) on generating the "Meta AI" token[cite: 76, 78].
3.  [cite_start]**Gradient Norms:** Calculating the $L_{2}$ norm of gradients to find parameters most sensitive to changes in the F1 logit[cite: 80].

[cite_start]This analysis converges on a critical set of **10 target modules** (e.g., MLP Layers L16, L18, L23 and Attention Layers L20, L22)[cite: 102, 104, 106]. The $(IA)^{3}$ PEFT method is *exclusively* applied to these modules.

### Phase 2: The "Unlearn-then-Learn" Pipeline

[cite_start]This is the core training strategy, which decouples fact suppression from fact acquisition[cite: 28].

#### Stage 1: Unlearning F1 (The "Unlearn" Adapter)
* [cite_start]**Objective:** To suppress the model's output of "Meta AI" and guide it toward an uncertainty response ("I am not sure...")[cite: 141, 142].
* [cite_start]**Method:** An $(IA)^{3}$ adapter ($\theta_{unlearn}$) is trained on the 10 target modules[cite: 145].
* [cite_start]**Data:** ~20 diverse paraphrases of the query ("Who developed PyTorch?") paired with the "I am not sure..." response[cite: 142, 143].

#### Stage 2: Learning F2 (The "Learn" Adapter)
* [cite_start]**Objective:** To instill the new fact, "Google."[cite: 67].
* **Method:**
    1.  [cite_start]First, the $\theta_{unlearn}$ adapter from Stage 1 is **permanently merged** into the base model's weights[cite: 151].
    2.  [cite_start]This new, "neutral" model is then fine-tuned with a *second* $(IA)^{3}$ adapter ($\theta_{learnF2}$), again targeting the same 10 modules[cite: 151, 155].
* [cite_start]**Data:** The same ~20 paraphrases, but now paired with the target response "Google."[cite: 153].

## 🛠️ Technical Stack & Environment

This code was developed and tested using the following environment:

* [cite_start]**Model:** `microsoft/Phi-3-mini-4k-instruct` (revision `66403f97`) [cite: 60]
* [cite_start]**Precision:** `bfloat16` [cite: 62]
* **Core Libraries:**
    * [cite_start]`pytorch 2.5.1 cul21` [cite: 62]
    * [cite_start]`transformers 4.43.4` [cite: 62]
    * [cite_start]`peft 0.10.0` (for $(IA)^{3}$) [cite: 62]
    * [cite_start]`transformerlens 2.15.4` [cite: 62]
    * [cite_start]`numpy 1.26.3` [cite: 62]

### Training Hyperparameters

Both training stages (Unlearn and Learn) used identical hyperparameters:

* [cite_start]**PEFT Method:** $(IA)^{3}$ [cite: 145, 155]
* [cite_start]**Learning Rate:** $5 \times 10^{-5}$ [cite: 147, 156]
* [cite_start]**Batch Size:** 1 [cite: 147]
* [cite_start]**Epochs:** 50 (for a total of 1000 steps) [cite: 147, 156]

## 📂 Repository Structure (Hypothetical)
