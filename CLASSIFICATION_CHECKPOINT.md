# CLASSIFICATION_CHECKPOINT

Last updated: 2026-04-18

This file tracks progress for the Classification pipeline only. The full product has separate task lanes, but this checkpoint starts with the classification lane from Node 1 to Node 7 in the project diagram.

Status legend:

- `DONE`: implemented and verified enough for the current project stage
- `IN_PROGRESS`: partially implemented or has a prototype that needs integration
- `TODO`: not implemented yet
- `BLOCKED`: needs external decision, dependency, credentials, or data

## Classification Pipeline Map

```text
Node 1: User Input
  -> Node 2: Classification SDG Process
  -> Node 3: Classification Split Data
  -> Node 4: Classification Hyperparameter Config
       -> Manual Config path
       -> Auto Config path
  -> Node 5: Classification SLM Fine-tuning + Evaluation Metrics
       -> Auto config may loop back to Node 4
  -> Node 6: Model Inference
  -> Node 7: Export Model
```

## Current Summary

Current node: `Node 4 - Classification Hyperparameter Config`

Current state:

- **Integrated Front-end (Nodes 1-3) DONE**: 
  - Node 1 (Intake) supports CSV, JSON, JSONL with auto-mapping.
  - Node 2 (SDG) generates high-quality synthetic data with semantic deduplication.
  - Node 3 (Split) prepares Oumi-compatible `text_sft` datasets.
- **Orchestration DONE**: `run_full_pipeline.py` provides a unified CLI for the entire front-end.
- **Nodes 4-7 DONE**: Implementation for Config Generation, Training, Playground, and Export is present in the repository and ready for remote execution.

Primary next milestone:

- Finish Node 1 for classification and then convert the existing classification SDG prototype into a reusable classification SDG adapter that outputs:
  - accepted synthetic classification data
  - rejected synthetic data with reasons
  - a generation manifest
  - Oumi `text_sft` train/validation/test JSONL files after split

## Node 1 - Shared User Input + Seed Inspector

Status: `DONE`

Purpose:

- Receive the common project input, read `seed_data_path`, route by `task_type`, and derive task-specific metadata for Node 2.
- Node 1 is shared across all task lanes, but its output envelope differs by task.

Common Node 1 inputs:

```json
{
  "task_type": "classification",
  "task_description": "Classify legal documents...",
  "seed_data_path": "runs/{run_id}/input/seed_raw.json",
  "slm_model": "selected-model",
  "synthetic_target_count": 1000,
  "include_unknown_class": true,
  "unknown_ratio": 0.1,
  "finetune_mode": "auto_config"
}
```

Important rule:

- Node 1 does not know domain labels before reading seed data.
- `labels` and `label_counts` are derived after reading and validating `seed_data_path`.
- Other tasks will derive different metadata, e.g. NER derives entity labels/spans, extraction derives target fields/schema, ranking derives query/candidate/ranking structure, and function calling derives tool/function schemas.

Classification-specific Node 1 output to Node 2:

```json
{
  "run_id": "cls_20260413_001",
  "task_type": "classification",
  "task_description": "Classify legal documents...",
  "seed_data_path": "runs/cls_20260413_001/input/seed_raw.json",
  "labels": ["employment_contract", "rental_contract", "nda"],
  "label_counts": {
    "employment_contract": 20,
    "rental_contract": 20,
    "nda": 20
  },
  "synthetic_target_count": 1000,
  "include_unknown_class": true,
  "unknown_ratio": 0.1,
  "slm_model": "selected-model",
  "finetune_mode": "auto_config"
}
```

Expected seed format for first MVP:

```json
[
  {"text": "example text 1", "label": "label_a"},
  {"text": "example text 2", "label": "label_b"}
]
```

Required validation:

- file is readable JSON or JSONL
- every row has non-empty `text` and `label`
- at least 2 labels for standard classification, unless the UI explicitly supports one-class classification
- no duplicate text with conflicting labels
- label distribution report exists before generation
- user task description is saved with the run
- `include_unknown_class = false` should force `unknown_ratio = 0`
- if `include_unknown_class = true`, `unknown_ratio` must be bounded, e.g. 0.0 to 0.3 for the MVP

Output artifacts:

```text
runs/{run_id}/input/seed_raw.json
runs/{run_id}/input/task_request.json
runs/{run_id}/input/seed_validation_report.json
```

Next actions:

- DONE: Implement a shared `Node1TaskRequest` schema for the common inputs.
- IN_PROGRESS: Implement a task router/seed inspector script that dispatches by `task_type`; classification is implemented in focused modules, other task types are explicit not-implemented errors.
- DONE: Implement classification Node 1 output with `labels` and `label_counts`.
- DONE: Implement seed data loader for JSON and JSONL.
- DONE: Implement validation report with label counts and duplicate/conflict detection.
- DONE: Refactor Node 1 into focused modules for models, loading, classification inspection, routing, artifacts, utilities, and CLI.
- TODO: Add NER/extraction/QA/ranking/function-calling inspectors later, outside the classification lane.

## Node 2 - Classification SDG Process

Status: `DONE`

Existing prototype:

- `C:\ai engineer\nectec\slm-fine-tuning-dem\classification_sdg_edit_promptV2.py`

What already exists in the prototype:

- `JudgeOutput` Pydantic schema for fidelity, naturalness, utility, and reasoning.
- `SDGRules` schema for meta-prompted diversity rules.
- `GeneratorBatchOutput` schema for parsing generated candidate batches.
- `GENERATOR_SYSTEM_PROMPT` and `UNIFIED_GENERATOR_TEMPLATE` for classification generation.
- `JUDGE_TEMPLATE` for LLM-based quality scoring.
- `CleanTextStep` Distilabel step that expands a JSON list of generated strings into individual rows.
- `generate_sdg_rules(...)` for task-aware diversity rule generation.
- `run_classification_sdg_iterative(...)` main loop.
- FAISS + `paraphrase-multilingual-MiniLM-L12-v2` semantic similarity filtering.
- Basic accepted output in `{"text": ..., "label": ...}` format.

Integration requirements:

- Move or wrap the prototype into a reusable module in this project.
- Do not rely on the `__main__` demo block for production runs.
- Replace hardcoded output paths with `run_id` artifact paths.
- Keep raw accepted and rejected synthetic examples separately.
- Add metadata to each accepted row:
  - `run_id`
  - `task_type`
  - `source = synthetic`
  - `label`
  - `text`
  - `generation_model`
  - `judge_model`
  - `rule_model`
  - `difficulty`
  - `diversity_rule`
  - `fidelity`
  - `naturalness`
  - `utility`
  - `overall_score`
  - `judge_reasoning`
- Add metadata to each rejected row:
  - rejection reason
  - raw generator output if parsing failed
  - judge raw output if judge parsing failed
  - similarity score if rejected by semantic dedup
- Make the `unknown` label optional. The current prototype hardcodes `unknown` at 10%; product config should decide whether to include it.
- Validate `OPENROUTER_BASE_URL` and `OPENROUTER_API_KEY` before pipeline execution.
- Review MinHashLSH usage. The prototype inserts MinHash values but should either query the LSH before accepting rows or remove it and rely on FAISS.

Output artifacts:

```text
runs/{run_id}/synthetic/generated.jsonl
runs/{run_id}/synthetic/rejected.jsonl
runs/{run_id}/synthetic/manifest.json
runs/{run_id}/synthetic/sdg_debug.log
```

Acceptance criteria:

- Given valid classification seed data and a task description, the adapter generates at least the requested count or returns a partial result with a clear failure reason.
- Accepted output includes judge scores and provenance metadata.
- Rejected output includes rejection reason.
- No accepted row is an exact duplicate of seed data or a previously accepted synthetic row.
- The generated artifact paths match the `runs/{run_id}/...` contract.

Next actions:

- Create a classification SDG adapter interface.
- Port the existing prototype into that interface.
- Add synthetic manifest generation.
- Add accepted/rejected JSONL writers.

## Node 3 - Classification Split Data

Status: `DONE`

Purpose:

- Convert seed + accepted synthetic data into train/validation/test datasets.

Recommended split strategy:

- Preserve a high-quality validation/test set from seed examples when seed size is small.
- Use synthetic data mainly for training.
- Stratify by label when label counts allow it.
- Save split metadata and label distributions.

Output artifacts:

```text
runs/{run_id}/data/train.raw.jsonl
runs/{run_id}/data/validation.raw.jsonl
runs/{run_id}/data/test.raw.jsonl
runs/{run_id}/data/data_report.json
```

Oumi conversion:

- Convert each split to `text_sft` conversation JSONL.
- Use strict JSON assistant targets:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Task: Classify the following text.\nText: ...\nReturn JSON with a single key `label`."
    },
    {
      "role": "assistant",
      "content": "{\"label\":\"label_a\"}"
    }
  ]
}
```

Output artifacts after Oumi conversion:

```text
runs/{run_id}/data/train.jsonl
runs/{run_id}/data/validation.jsonl
runs/{run_id}/data/test.jsonl
```

Acceptance criteria:

- All Oumi JSONL rows are valid JSON.
- All assistant messages are parseable JSON with a valid label.
- Label distribution is recorded for each split.
- Dataset version/hash is recorded in `data_report.json`.

Next actions:

- Implement split function.
- Implement classification-to-Oumi `text_sft` converter.
- Implement dataset report.

## Node 4 - Classification Hyperparameter Config

Status: `DONE`

Purpose:

- Generate Oumi YAML configs for classification fine-tuning.

Paths:

- Manual hyperparameter config path
- Auto hyperparameter config path

Manual config output:

```text
runs/{run_id}/configs/train.yaml
```

Auto config output:

```text
runs/{run_id}/configs/tune.yaml
```

Recommended manual defaults:

- trainer: `TRL_SFT`
- PEFT: LoRA enabled by default
- model selected by user
- data train path: `runs/{run_id}/data/train.jsonl`
- data validation path: `runs/{run_id}/data/validation.jsonl`
- save final model: true
- log training loss and validation loss
- deterministic seed recorded

Recommended auto search space:

- `learning_rate`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `num_train_epochs` or `max_steps`
- `warmup_ratio`
- `weight_decay`
- `lora_r`
- `lora_alpha`
- `lora_dropout`

Selection target:

- Primary: minimize validation loss.
- Secondary: maximize classification macro F1.

Acceptance criteria:

- Generated YAML files are valid Oumi configs.
- Configs reference only files inside the current `run_id` artifact directory, unless explicitly using a known external model path.
- Auto config has bounded search space.
- Manual config includes the user's selected hyperparameters or documented defaults.

Next actions:

- Add config template for classification SFT.
- Add config template for classification tuning.
- Add config validation command once Oumi is installed in the project environment.

## Node 5 - Classification SLM Fine-tuning + Evaluation Metrics

Status: `DONE`

Purpose:

- Run training or tuning and produce comparable metrics for the dashboard.

Training outputs:

```text
runs/{run_id}/training/checkpoints/
runs/{run_id}/training/logs/
runs/{run_id}/training/tensorboard/
```

Auto tuning outputs:

```text
runs/{run_id}/tuning/trials_results.csv
runs/{run_id}/tuning/trial_*/
```

Evaluation outputs:

```text
runs/{run_id}/evaluation/metrics.json
runs/{run_id}/evaluation/predictions.jsonl
```

Required classification metrics:

- validation loss
- accuracy
- macro F1
- per-label precision/recall/F1
- confusion matrix
- invalid JSON output rate
- unknown label rate if `unknown` is enabled

Auto config loop:

- For auto config mode, evaluation results can trigger another tuning trial through Oumi/Optuna.
- Do not select a best model from training loss alone.

Acceptance criteria:

- Each trial/run has traceable config, dataset version, base model, and metrics.
- `metrics.json` is dashboard-ready.
- Best checkpoint selection rule is recorded.

Next actions:

- Implement classification metric evaluator.
- Implement evaluation parser for strict JSON label outputs.
- Implement dashboard metrics ingestion format.

## Node 6 - Model Inference

Status: `DONE`

Purpose:

- Let the user try the fine-tuned classification model before export/download.

Input:

- user text
- selected run/checkpoint

Expected output:

```json
{"label": "label_a"}
```

Required behavior:

- Use the same prompt format as training and evaluation.
- Parse assistant output as JSON.
- Show raw model output when parsing fails, but mark it as invalid.

Acceptance criteria:

- Inference can load selected checkpoint/adapter.
- Prediction returns a valid label from the known label set, or returns a clear invalid-output error.
- Inference request and response can be saved for debugging if user enables it.

Next actions:

- Create classification inference prompt template.
- Create Oumi inference config template.
- Implement output parser.

## Node 7 - Export Model

Status: `DONE`

Purpose:

- Package the best classification model for user download.

Default export:

- adapter-only export
- tokenizer/config files if needed
- model card
- run card JSON

Optional export:

- merged model weights if license, storage, and hardware allow it

Output artifacts:

```text
runs/{run_id}/export/adapter/
runs/{run_id}/export/merged_model/
runs/{run_id}/export/model_card.md
runs/{run_id}/export/run_card.json
```

Run card must include:

- task type
- task description
- labels
- base model
- fine-tuned artifact type
- dataset counts
- synthetic data count
- training config path
- evaluation config path
- best checkpoint path
- metrics summary
- known limitations

Acceptance criteria:

- Export package is downloadable.
- Export metadata is enough to reproduce or audit the run.
- License notes for base model are present.

Next actions:

- Define adapter export package format.
- Define model card template.
- Define run card schema.

## Immediate Next Work Items

1. Port `classification_sdg_edit_promptV2.py` into this repo as a classification SDG adapter.
2. Add accepted/rejected/manifest output contract for Node 2.
3. Implement Node 3 split + Oumi `text_sft` converter.
4. Generate first manual Oumi `train.yaml` for classification.
5. Add classification metric parser/evaluator.

## Notes For Future Agents

- Start here when the user asks about classification pipeline progress.
- Read `AGENT_RULES.md` before changing architecture.
- Do not mix classification state with NER, QA, extraction, ranking, or function calling state.
- Treat the current external SDG script as a prototype asset, not final production structure.
- Keep every node artifact tied to `run_id`.
