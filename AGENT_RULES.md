# AGENT_RULES: SLM Fine-tune Auto Config Hyperparameter Project

Last updated: 2026-04-13

This file is the project checkpoint for future Codex/agent sessions. Read it before proposing architecture, implementing pipeline code, or changing configs.

## Project Goal

Build a product that lets a user upload a small seed dataset, describe a task, generate synthetic training data, fine-tune a selected SLM, compare runs, test inference, and download the final model artifact.

The key product constraint is that the system does not know the user's domain/content until runtime. All config generation, validation, metrics selection, and hyperparameter search must be task-aware but content-agnostic before the user submits data.

Supported task types:

- classification
- NER
- question answering
- extraction
- ranking
- function calling

User inputs:

- seed data: 50-100 examples
- task description
- task type
- SLM model selection
- synthetic data target count
- fine-tuning mode: manual hyperparameter config or auto hyperparameter config

Existing component:

- Synthetic data generation already exists and uses Distilabel.

## Recommended High-Level Solution

Use Oumi as the training/evaluation/inference/config execution layer, and keep Distilabel as the existing synthetic data generation layer.

Recommended pipeline:

1. Intake and validation
   - Store raw seed data and task description.
   - Detect/validate schema for the selected task type.
   - Normalize all examples into an internal canonical format.
   - Reject or quarantine empty, duplicated, malformed, label-inconsistent, or unsafe examples.
   - Treat Node 1 as a shared input and seed-inspection layer. Its common inputs are `task_type`, `task_description`, `seed_data_path`, `slm_model`, `synthetic_target_count`, `include_unknown_class`, `unknown_ratio`, and `finetune_mode`.
   - Task-specific metadata is derived after reading `seed_data_path`; for example, classification derives `labels` and `label_counts`, while NER derives entity labels/spans. Do not require the user or system to know these values before upload.

2. Task router
   - Route to a task-specific adapter: classification, NER, QA, extraction, ranking, or function calling.
   - The router must output:
     - Distilabel generation prompt/template requirements.
     - Oumi dataset format target.
     - Training config template.
     - Evaluation config template.
     - Custom metric function set.
     - Inference prompt/rendering template.

3. Synthetic data generation
   - Use Distilabel to generate synthetic examples from seed data and task description.
   - Preserve provenance fields: seed source, synthetic generation run id, task type, label schema, prompt/template version, generator model, and quality score if available.
   - Do not feed all synthetic data directly into training without validation.

4. Dataset analysis and quality gate
   - Run dataset profiling before split/train.
   - Track sample count, label distribution, token length distribution, empty fields, duplicates, JSON validity, and task-specific constraints.
   - Keep a filtered trainable dataset and a separate rejected dataset with reasons.

5. Split data
   - Split after synthetic data quality filtering.
   - Prefer stratified split for classification/NER labels where possible.
   - Keep a small, high-quality validation/test set. If only 50-100 seed examples exist, reserve seed examples for validation/test and use synthetic examples mainly for training when possible.

6. Convert to Oumi-compatible data
   - Default to Oumi `text_sft` JSONL conversation format for text-only SFT.
   - Each example should be a JSON object with `messages`, normally user prompt plus assistant target.
   - Use strict structured outputs for classification, extraction, NER, ranking, and function calling so evaluation is parseable.

7. Fine-tune
   - Default method: SFT with LoRA or QLoRA for SLMs unless full fine-tuning is explicitly selected and hardware allows it.
   - Manual mode: user provides or edits hyperparameters, then the system writes an Oumi training YAML.
   - Auto mode: system writes an Oumi tuning YAML and runs Optuna-based `oumi tune`.

8. Evaluation and dashboard
   - Dashboard must compare:
     - training loss
     - validation loss
     - task-specific custom metrics
     - run metadata
     - hyperparameter values
     - dataset version
     - model base/checkpoint
   - Use TensorBoard/W&B artifacts if enabled, plus persist local JSON/CSV summaries for the app database.

9. Inference playground
   - Load the base model plus adapter/checkpoint for interactive testing.
   - Use the same prompt/rendering template as training/evaluation.
   - For local GPU inference, prefer vLLM when the model fits; otherwise use Oumi native/Transformers-style inference.

10. Export/download
   - Export adapter and metadata by default.
   - Optionally merge adapter into base model if license, storage, and hardware allow.
   - Provide model card/run card with dataset summary, metrics, hyperparameters, base model, training command/config, and known limitations.

## Why Oumi Fits This Project

Oumi training uses YAML configs built around `TrainingConfig`, with separate sections for model, data, training, PEFT, and FSDP. This maps cleanly to generated manual configs and auto configs.

Oumi SFT supports input-output task adaptation and uses a conversation format with user/assistant messages. This is a good default for QA, extraction, classification, ranking, function calling, and NER when each task is rendered as an instruction plus structured answer.

Oumi dataset docs recommend `text_sft` / `TextSftJsonLinesDataset` for text-only chat data and JSONL conversation format. This should be the canonical bridge from Distilabel output to fine-tuning.

Oumi tuning provides `oumi tune`, powered by Optuna, with TPE and random samplers, multi-objective metrics, custom evaluation metrics, saved trial logs, `trials_results.csv`, and best checkpoint selection. This directly matches the project's auto hyperparameter loop.

Oumi evaluation supports YAML evaluation configs with model params, task list, generation params, local inference engine, remote inference params, output directory, and W&B logging. It supports standard benchmarks, generative evaluation, LLM-as-judge, and custom evaluation functions.

Oumi inference provides a unified interface for local engines such as vLLM, native/Transformers, and LlamaCPP, plus remote APIs. Use it for the model playground after fine-tuning.

Oumi dataset analysis can profile text/token lengths, outliers, empty samples, and filtered subsets. Use it before training and after synthetic data generation.

Oumi quantization exists but is marked experimental/active-development in the docs. Treat it as optional, not part of the MVP critical path.

## Vast.ai Recommendation

Use Vast.ai as a GPU execution option for training/tuning jobs, especially when local GPU capacity is insufficient.

Operational notes:

- Prefer a Docker template with pinned CUDA/PyTorch/Oumi/Distilabel dependencies.
- Choose disk size carefully because Vast.ai disk space is selected at instance creation and cannot be resized later on that instance.
- Upload SSH keys for reliable access.
- Use Stop to pause GPU billing when preserving an instance; use Delete when finished to stop charges completely.
- Persist important outputs outside the instance before deletion: checkpoints, adapters, trial CSVs, logs, dataset manifests, and model cards.
- Add balance/autobilling safeguards for long tuning runs because a zero balance can interrupt running instances.

## Auto Hyperparameter Strategy

The auto config generator should create a bounded search space, not an unbounded "let Optuna decide everything" setup.

Recommended first search space:

- `learning_rate`: loguniform, e.g. 1e-5 to 5e-4 for LoRA SFT
- `per_device_train_batch_size`: categorical based on model size and GPU VRAM, e.g. [1, 2, 4]
- `gradient_accumulation_steps`: int/categorical to target effective batch size
- `num_train_epochs` or `max_steps`: small bounded range; avoid overfitting synthetic data
- `warmup_ratio`: uniform, e.g. 0.0 to 0.1
- `weight_decay`: uniform, e.g. 0.0 to 0.1
- `lora_r`: categorical, e.g. [4, 8, 16, 32]
- `lora_alpha`: categorical, e.g. [8, 16, 32, 64]
- `lora_dropout`: uniform, e.g. 0.0 to 0.1

Recommended fixed defaults:

- trainer: `TRL_SFT`
- PEFT enabled by default
- LoRA/QLoRA depending on VRAM and model size
- `eval_strategy: steps`
- `save_final_model: true`
- TensorBoard enabled locally
- W&B optional
- deterministic seed recorded

Optimization target:

- Primary: minimize validation loss or task-specific validation error.
- Secondary: maximize task-specific metric.
- Use multi-objective only when the dashboard and selection logic can explain tradeoffs; otherwise compute secondary metrics but select with one primary metric.

Auto tuning must include early stopping/pruning rules in the orchestration layer when available, because users may request large synthetic data counts and cloud GPU time is expensive.

## Task-Specific Data and Metrics

Classification:

- Output format: strict JSON, e.g. `{"label": "..."}`.
- Metrics: accuracy, macro F1, per-class F1, confusion matrix.
- Quality gates: label set consistency, class balance, duplicate texts with conflicting labels.

NER:

- Output format: strict JSON entities with spans when possible, e.g. `{"entities":[{"text":"...","label":"...","start":0,"end":4}]}`.
- Metrics: entity-level precision/recall/F1, exact span match, label-wise F1.
- Quality gates: valid spans, valid labels, no overlapping entities unless schema allows it.

Question answering:

- Output format: direct answer or JSON with answer and evidence depending on task.
- Metrics: exact match, token F1, semantic similarity, optional LLM judge for open-ended answers.
- Quality gates: answer grounded in context when context is provided.

Extraction:

- Output format: strict JSON matching a user-defined schema.
- Metrics: field-level exact match, JSON validity, schema validity, per-field precision/recall/F1.
- Quality gates: parseable JSON, required fields, type validation.

Ranking:

- Output format: ordered ids or pairwise preference.
- Metrics: NDCG, MRR, pairwise accuracy, Spearman/Kendall where applicable.
- Quality gates: no missing ids, no duplicate rank positions, all candidates accounted for.

Function calling:

- Output format: strict tool/function call JSON.
- Metrics: function name accuracy, argument exact match, JSON validity, schema validity.
- Quality gates: arguments match declared schema, no invented tools, valid enum values.

## Suggested Directory/Artifact Contract

Use this convention unless the repo later establishes a different one:

```text
runs/
  {run_id}/
    input/
      seed_raw.*
      task_request.json
    synthetic/
      generated.jsonl
      rejected.jsonl
      manifest.json
    data/
      train.jsonl
      validation.jsonl
      test.jsonl
      data_report.json
    configs/
      train.yaml
      tune.yaml
      eval.yaml
      infer.yaml
    training/
      checkpoints/
      logs/
      tensorboard/
    tuning/
      trials_results.csv
      trial_*/
    evaluation/
      metrics.json
      predictions.jsonl
      judge_results.jsonl
    export/
      adapter/
      merged_model/
      model_card.md
      run_card.json
```

## MVP Recommendation

Build MVP in this order:

1. Manual SFT path for one small instruct SLM with LoRA.
2. Distilabel output to Oumi `text_sft` JSONL converter.
3. Classification and extraction task adapters first, because metrics and strict JSON evaluation are easiest to stabilize.
4. Evaluation runner and dashboard ingestion.
5. Inference playground against the fine-tuned checkpoint/adapter.
6. Auto hyperparameter config with a small `oumi tune` search space.
7. Add NER, QA, ranking, and function calling adapters.
8. Add Vast.ai execution path.
9. Optional quantization/export variants.

## Non-Negotiable Engineering Rules

- Never assume task content/schema before upload; infer/validate after seed data arrives.
- Keep task config, dataset version, synthetic generation version, training config, and evaluation config tied to a single `run_id`.
- Store every generated YAML config and all final resolved hyperparameters.
- Prefer parseable structured outputs for all non-open-ended tasks.
- Do not select a best trial only from training loss; use validation loss and task metrics.
- Do not train on raw synthetic data before quality filtering.
- Keep seed data separated from synthetic data in manifests for traceability.
- Do not delete failed trial artifacts until the dashboard has recorded failure status and reason.
- Be explicit about base model licenses and whether exported artifacts include only adapters or merged weights.
- Treat Oumi quantization as optional until the feature is stable enough for the product path.

## Source Docs Read

- Oumi docs: https://www.oumi.ai/docs/en/latest/index.html
- Oumi training: https://www.oumi.ai/docs/en/latest/user_guides/train/train.html
- Oumi training methods: https://www.oumi.ai/docs/en/latest/user_guides/train/training_methods.html
- Oumi training config: https://www.oumi.ai/docs/en/latest/user_guides/train/configuration.html
- Oumi data formats: https://www.oumi.ai/docs/en/latest/resources/datasets/data_formats.html
- Oumi dataset analysis: https://www.oumi.ai/docs/en/latest/user_guides/analyze/analyze.html
- Oumi synthesis: https://www.oumi.ai/docs/en/latest/user_guides/synth.html
- Oumi hyperparameter tuning: https://www.oumi.ai/docs/en/latest/user_guides/tune.html
- Oumi evaluation: https://www.oumi.ai/docs/en/latest/user_guides/evaluate/evaluate.html
- Oumi evaluation config: https://www.oumi.ai/docs/en/latest/user_guides/evaluate/evaluation_config.html
- Oumi judge: https://www.oumi.ai/docs/en/latest/user_guides/judge/judge.html
- Oumi inference: https://www.oumi.ai/docs/en/latest/user_guides/infer/infer.html
- Oumi launch: https://www.oumi.ai/docs/en/latest/user_guides/launch/launch.html
- Oumi quantization: https://www.oumi.ai/docs/en/latest/user_guides/quantization.html
- Vast.ai quickstart: https://docs.vast.ai/documentation/get-started/quickstart
- Vast.ai pricing: https://docs.vast.ai/documentation/instances/pricing
