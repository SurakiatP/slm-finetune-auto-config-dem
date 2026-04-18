# NODE 1 CHECKPOINT: Intake & Validation

**Last Updated:** 2026-04-18
**Status:** ✅ Completed & Tested

## 🎯 Goal
Act as the "Front-door" for the SLM Auto Config Pipeline by ingesting raw, unstructured user data and normalizing it into a standard, clean format required by downstream nodes (like Node 2 SDG).

## 🛠️ Key Components
- **Architecture Base:** Implementation in `src/slm_auto_config/node1/`.
- **Classification Mapper:** `ClassificationIntake` in `node1/classification.py`.
- **Parsing Engine:** `BaseIntake` in `node1/base.py` (Supports CSV, JSON, JSONL).
- **Factory Router:** `get_intake()` in `node1/factory.py`.
- **Schema Validation:** `NormalizedExample` and `IntakeMetadata` via Pydantic in `node1/models.py`.

## ✨ Features Verified
1. **Multi-format Support:** Seamless parsing of `.csv`, `.json`, and `.jsonl` files.
2. **Auto-Mapping Columns:** Heuristics automatically match fuzzy column names (e.g., `Document Content` -> `text`, `Doc_Class` -> `label`).
3. **Quarantine Logic:** Flawed rows (missing or whitespace-only labels/texts) are aggressively dropped without interrupting the flow.
4. **Metadata Extraction:** Pre-calculates dataset metrics (`label_distribution`, row counts) and stores them in `task_request.json` for Node 2.

## 🧪 Verification Results
- **Automated Tests:** `test_node1_intake.py` confirms successful parsing, fuzzy mapping, dropping of bad rows, and saving outputs precisely as needed.

## 🔄 Relationship to Pipeline
- **Upstream:** Receives raw User Data files.
- **Downstream:** Outputs exactly two files to the run directory (`runs/{run_id}/input/`):
  1. `seed_raw.jsonl` (Cleaned data array).
  2. `task_request.json` (Dataset stats & metrics). Node 2 (SDG) will use this metadata to establish diversity rules.

## 🚀 Next Steps
- Node 1 and Node 2 are officially operational. The connection point between Intake and Data Generation is secure.
- We are ready to proceed with assessing **Full Pipeline Orchestration** or connecting Node 1 + Node 2 sequentially!
