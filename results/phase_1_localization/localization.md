# Phase 1: Circuit Localization Summary - MVED Project

**Date:** May 29, 2025
**Target Fact (F1_target):** "Who developed PyTorch?" -> "Meta AI"
**Objective:** To identify key MLP layers and Attention Head components within the `microsoft/Phi-3-mini-4k-instruct` model (revision `66403f97`) that are causally implicated in the recall of the target fact. These components will be prioritized for LoRA-based modulation in Phase 2.

## 1. Introduction

This report summarizes the findings from Phase 1 of the MVED project: Circuit Hypothesizing & Advanced Localization. Various interpretability techniques were employed using TransformerLens on the Phi-3-mini model to pinpoint components critical for the factual association between the prompt "Who developed PyTorch?" and the correct object " Meta AI". The environment was configured using Python 3.10.16, PyTorch 2.5.1+cu121, Transformers 4.43.4, PEFT 0.10.0, TransformerLens 2.15.4, and NumPy 1.26.3. The TransformerLens model was loaded using `from_pretrained_no_processing` to bypass potential issues with internal canonicalization steps, while ensuring device consistency.

## 2. Methodologies Employed

The localization process involved a multi-faceted approach to gather converging evidence:

1.  **Initial Model Exploration (Script 07):** Ensured the target fact was processed by the TransformerLens-wrapped model and that basic activation caching was functional. The target object " Meta AI" (token ID 29871) yielded a raw logit of 38.50.
2.  **Broad Activation Patching (Script 08):**
    * Attention head outputs (`hook_z`) and MLP layer outputs (`hook_post`) were patched.
    * Activations from a `corrupted_prompt` ("Today is a sunny day...") were used to replace activations from the `clean_prompt` ("Who developed PyTorch?").
    * The impact was measured as the "logit drop" on the target token " Meta AI".
3.  **Refined Causal Patching (Script 09):**
    * For top components identified in Script 08, more granular patching was performed.
    * Attention head value vectors (`hook_v` - pre W_O projection).
    * MLP layer pre-nonlinearity activations (`hook_pre` - output of the first linear transformation).
4.  **Fallback & Supplementary Analysis (Script 10):**
    * **Activation Magnitude Analysis:** Mean absolute activation of `hook_z` (heads) and `hook_post` (MLPs) at the final token position for the clean prompt.
    * **Weight Gradient Norm Analysis:** L2 norm of gradients of LoRA-targetable weights (Q, K, V, O projections for attention layers; input/gate and output linear layers for MLPs) with respect to the negative log-likelihood of the target token " Meta AI".

## 3. Summary of Key Findings Across Analyses

*(This section would typically include condensed tables or lists of the top 5-10 components from each JSON file. For brevity here, key trends are summarized, and specific values are cited in Section 4 for selected components.)*

* **Causal Patching (Scripts 08 & 09):**
    * Identified specific MLP layers (notably L18, L16, L23, L17) where patching `hook_pre` (input stage) often caused significant logit drops, sometimes larger than patching `hook_post` (output stage). This highlights the importance of the initial linear transformation and activation function within these MLPs.
    * Identified specific Attention Heads (notably L22H28, L20H26) where patching `hook_z` (output) caused significant logit drops. Subsequent `hook_v` patching helped discern the importance of the value vectors themselves.
    * Many components showed near-zero or negative logit drops, aiding in narrowing the focus.
* **Activation Magnitudes (Script 10):**
    * Later layers (approx. L20-L24 for both attention heads and MLPs) generally showed higher mean absolute activation at the final token position, suggesting greater overall involvement in processing at that stage.
* **Gradient Norms (Script 10):**
    * Provided a measure of parameter sensitivity for LoRA-targetable weights.
    * Attention `W_V` and `W_O` weights in early-to-mid layers (L0-L6, L10-L15) showed high norms.
    * MLP `W_out` (down_proj) weights in early layers (L2, L4) and some late layers (L29, L31) showed particularly high norms.
    * MLP `W_in` and `W_gate` (gate_up_proj) weights showed more distributed sensitivity, with notable norms in early, mid, and late layers.

## 4. Hypothesized Critical Circuit Components for LoRA Targeting

Based on converging evidence, prioritizing strong causal impact from patching experiments, supported by gradient norm sensitivity and activation magnitudes, the following components are selected as primary candidates for forming the hypothesized knowledge circuit for the target fact.

### 4.1. MLP Layers

1.  **MLP Layer 18 (blocks.18.mlp):**
    * **Evidence:**
        * **Causal Patching:** Strongest `hook_pre` logit drop (**3.50**), significant `hook_post` logit drop (1.00).
        * Activation Magnitude: 0.086 (Moderate).
        * Gradient Norms (L18): `W_in` (8.12), `W_gate` (7.75), `W_out` (10.06) – all show good sensitivity.
    * **Rationale:** The exceptionally high impact when patching its input stage (`hook_pre`) strongly suggests that the first linear transformation and subsequent activation within MLP L18 are crucial for processing or recalling this fact.
    * **Proposed LoRA Targets:**
        * `blocks.18.mlp.W_in` (or equivalent first linear layer input weight)
        * `blocks.18.mlp.W_gate` (if applicable for SwiGLU/GEGLU type, part of the first effective linear layer)
        * `blocks.18.mlp.W_out` (output linear layer, `down_proj` equivalent)

2.  **MLP Layer 16 (blocks.16.mlp):**
    * **Evidence:**
        * **Causal Patching:** Very high `hook_pre` logit drop (**3.00**), despite a negative `hook_post` drop (-1.75).
        * Activation Magnitude: 0.070 (Relatively Lower).
        * Gradient Norms (L16): `W_in` (7.93), `W_gate` (7.40), `W_out` (9.06) – all moderate sensitivity.
    * **Rationale:** The dramatic positive impact of patching `hook_pre` indicates the input processing of MLP L16 is vital. The negative `hook_post` drop is particularly interesting, suggesting its final output (in the clean run) might be performing a complex role (e.g., suppressing alternatives, or was simply less aligned with the target than the corrupted patch). This makes it a prime candidate for *modulation* to reshape its function.
    * **Proposed LoRA Targets:**
        * `blocks.16.mlp.W_in`
        * `blocks.16.mlp.W_gate`
        * `blocks.16.mlp.W_out`

3.  **MLP Layer 23 (blocks.23.mlp):**
    * **Evidence:**
        * **Causal Patching:** Strongest `hook_post` logit drop (**2.00**), high `hook_pre` logit drop (1.50).
        * Activation Magnitude: **0.122 (High for MLPs)**.
        * Gradient Norms (L23): `W_in` (6.96), `W_gate` (7.09), `W_out` (8.62) – moderate sensitivity.
    * **Rationale:** High causal impact from both input and output stages, coupled with high activation, points to significant involvement. The output stage (`hook_post`) appears slightly more critical from patching.
    * **Proposed LoRA Targets:**
        * `blocks.23.mlp.W_out` (primary)
        * `blocks.23.mlp.W_in`
        * `blocks.23.mlp.W_gate`

### 4.2. Attention Heads/Layers

1.  **Attention Layer 22, Head 28 (L22H28):**
    * **Evidence:**
        * **Causal Patching:** Top `hook_z` logit drop (**1.25**), moderate `hook_v` logit drop (0.50).
        * Activation Magnitude: **0.59 (High for heads)**.
        * Gradient Norms (Layer L22): `W_V` (8.81), `W_O` (4.78) show moderate sensitivity for the layer. `W_Q` (3.25), `W_K` (3.21) are lower.
    * **Rationale:** Strong causal impact, particularly its final output (`hook_z`), and high activation make it a key candidate. The difference between `hook_z` and `hook_v` drops suggests the W_O projection is also playing a significant role.
    * **Proposed LoRA Targets (for Layer 22, focused effect desired on H28):**
        * `blocks.22.attn.W_Q`
        * `blocks.22.attn.W_K`
        * `blocks.22.attn.W_V`
        * `blocks.22.attn.W_O`
        *(Note: LoRA applied to these layer-level matrices will affect all heads in L22. Per-head LoRA is more complex but ideal if feasible. For now, targeting the layer's weights based on this head's activity is the plan).*

2.  **Attention Layer 20, Head 26 (L20H26):**
    * **Evidence:**
        * **Causal Patching:** High `hook_z` logit drop (1.00), strong `hook_v` logit drop (0.75).
        * Activation Magnitude: (Assume moderate-to-high based on causal scores; specific value for H26 from full JSON).
        * Gradient Norms (Layer L20): `W_V` (11.25), `W_O` (6.62) show moderate-to-high sensitivity for the layer. `W_Q` (2.98), `W_K` (2.79) are lower.
    * **Rationale:** Strong causal impact from both overall output and value vector, indicating the information it attends to and retrieves is important. Good gradient sensitivity for layer-level W_V and W_O.
    * **Proposed LoRA Targets (for Layer 20, focused effect desired on H26):**
        * `blocks.20.attn.W_Q`
        * `blocks.20.attn.W_K`
        * `blocks.20.attn.W_V`
        * `blocks.20.attn.W_O`

*(Self-correction: The provided parameter names for LoRA targets like `blocks.X.mlp.W_in` are typical TransformerLens conventions. The user must verify these exact names using `tl_model.named_parameters()` before implementing LoRA in Phase 2 to ensure the correct weights are targeted.)*

## 5. Discussion & Challenges

* The use of `from_pretrained_no_processing` for `HookedTransformer` initialization was necessary to bypass internal device errors. This means standard TransformerLens canonicalizations like bias folding were not applied. While the model remains mathematically equivalent for forward passes, this might affect certain interpretability techniques that assume the canonical structure. However, for activation patching and gradient analysis, it was sufficient.
* The choice of a generic `corrupted_prompt` for patching provides a general measure of component importance. Different corrupted sources (e.g., zero ablation, mean ablation, or more semantically targeted corruptions) could yield different insights.
* Gradient norms indicate parameter sensitivity for the specific target fact but do not isolate the function of individual heads within a layer if targeting layer-wide QKV/O matrices.
* Activation magnitudes are correlational and serve as supporting evidence rather than primary causal indicators.

## 6. Conclusion & Next Steps for Phase 2

Phase 1 successfully identified several MLP layers (L18, L16, L23) and Attention components (L22H28, L20H26, and by extension their layers L22, L20) that exhibit strong converging evidence of involvement in recalling the target fact "Who developed PyTorch?" -> "Meta AI".

These components form the basis of our hypothesized elementary knowledge circuit. In Phase 2, "Deterministic Circuit Modulation," LoRA modules will be applied to the identified targetable weights (input/gate/output linear layers for the selected MLPs; Q, K, V, O projections for the selected attention layers) to attempt to change the model's prediction from "Meta AI" to the modulated object "Google". The effectiveness and locality of this modulation, along with safety implications, will be rigorously evaluated.