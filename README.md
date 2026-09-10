# Surgical Knowledge Editing Experiments

Historical experimental code accompanying the 2025 preprint
[*Surgical Knowledge Rewrite in Compact LLMs: An “Unlearn-then-Learn” Strategy with IA³*](https://arxiv.org/abs/2508.07075).

> [!IMPORTANT]
> This repository documents an early exploratory study. The reported results
> reflect its original experimental setup and researcher-constructed baselines,
> and are best read as preliminary findings within that scope rather than broad
> comparative benchmark claims. The code is preserved as an inspectable record
> of the investigation and the method it developed.

## What this repository explores

The project tested a two-stage approach to editing a conflicting factual
association in `microsoft/Phi-3-mini-4k-instruct`:

1. **Circuit-localization experiments** attempted to identify model components
   associated with the original fact using activation analysis, causal
   patching, and gradient-based signals.
2. **Unlearn stage** trained an IA³ adapter to suppress the model's default
   response to the selected prompts.
3. **Learn stage** trained a second IA³ adapter on a counterfactual replacement
   after merging the first intervention.
4. **Evaluation scripts** probed the edited association, unrelated control
   facts, general responses, and a small collection of safety prompts.

The repository should be read as an experimental pipeline and research record,
not as a maintained library or current statement of the author's research
agenda.

## Repository map

```text
config/             Experiment and evaluation configurations
data/               Small generated datasets used by the scripts
reports/            Historical report placeholders and notes
results/            Selected intermediate outputs from early phases
scripts/            Data preparation, localization, training, and evaluation
src/mved/            Reusable model, PEFT, interpretability, and metric utilities
```

The main experimental sequence is represented by:

```text
scripts/00_run_env_check.py
scripts/01_run_spo_data_prep.py
scripts/02_run_phi3_baseline_general.py
scripts/03_run_phi3_baseline_spo.py
scripts/04_run_phi3_baseline_safety.py
scripts/06_run_fact_selection.py
scripts/07_run_tl_initial_exploration.py
scripts/08_run_activation_attribution.py
scripts/09_run_causal_patching.py
scripts/phase_2/01_generate_finetune_data_p2.py
scripts/phase_2/02_train_deterministic_lora_p2.py
scripts/phase_2/03_evaluate_deterministic_lora_p2.py
```

These scripts reflect the environment and assumptions of the original study.
They have not been consolidated into a one-command reproduction pipeline.

## Environment

The original experiments used:

- Python and PyTorch
- `microsoft/Phi-3-mini-4k-instruct`
- Hugging Face Transformers and PEFT
- IA³ adapters
- TransformerLens-based analysis utilities

Historical dependency specifications are available in `environment.yml`,
`constraints.txt`, and `mved_project_requirements_pinned.txt`.

## Scope and interpretation

- Selected intermediate outputs are committed.
- The Markdown files under `reports/` are empty historical placeholders.
- The repository emphasizes the experimental pipeline and selected artifacts
  rather than a complete benchmark release.
- The study focused on a narrow factual-editing example, so its findings are
  specific to that setting.
- A comprehensive shared-protocol comparison with methods such as ROME, MEMIT,
  MEND, SERAC, and contemporary unlearning systems was outside the project's
  scope.

This framing keeps the repository useful for studying the approach without
turning its initial results into claims beyond the experiment that produced
them.

## Citation

If you discuss the historical study, cite the version of the preprint you
actually consulted. Citation metadata is provided in `CITATION.cff`.

## License

The original code in this repository is released under the MIT License. Model,
dataset, and third-party dependencies remain subject to their respective terms.
