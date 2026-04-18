# NODE 2 CHECKPOINT: Synthetic Data Generation (SDG)

**Last Updated:** 2026-04-18
**Status:** ✅ Completed & Verified (Parity with V2 Script)

## 🎯 Goal
Generate high-quality, diverse synthetic training data using Distilabel, guided by task descriptions and small seed datasets. This node acts as the "Engine" for data expansion before the fine-tuning process.

## 🛠️ Key Components
- **SDG Process & Router:** Implementation in `src/slm_auto_config/node2/`.
- **Classification Engine:** `ClassificationSDGGenerator` in `node2/classification.py`.
- **Base Engine:** `BaseSDGEngine` in `node2/base.py` (provides core LSH + FAISS logic).
- **Factory Pattern:** `get_sdg_generator` in `node2/factory.py` for task-based routing.

## ✨ Features Verified
1. **RTC-FO Prompt Structure:** Implemented Role, Task, Context, Few-shot, and Output instructions in `GENERATOR_SYSTEM_PROMPT`.
2. **Meta-Prompting:** Automatically generates task-specific "Diversity Rules" using an LLM (e.g., Qwen 3.6 Plus).
3. **Iterative Generation Loop:** Refined logic to continue generation cycles until the `target_count` is reached with 100% valid, judged samples.
4. **Global Deduplication:**
   - **MinHashLSH (90%):** Detections for near-exact text duplicates.
   - **FAISS (85%):** Detections for semantic redundancy using `paraphrase-multilingual-MiniLM-L12-v2`.
5. **Quality Gate:** Integrated LLM-based Judge (`JUDGE_TEMPLATE`) to score Fidelity, Naturalness, and Utility.

## 🧪 Verification Results
- **Parity Test:** Successfully matched the behavior of `classification_sdg_edit_promptV2.py`.
- **Test Run:** Verified with `target_count = 10` over `mock_seed.json`.
- **Quality:** High-fidelity Thai legal text generation with distinct styles.

## ⚠️ Technical Notes & Gotchas
- **Windows Encoding:** When running on Windows, use `$env:PYTHONIOENCODING='utf-8'` to avoid `UnicodeEncodeError` when printing Thai characters or emojis to the terminal.
- **Dependencies:** Requires `datasketch`, `faiss-cpu`, and `sentence-transformers`.
- **Deduplication Strategy:** Comparison is 2-layered. First, it records text via MinHash for history, then it uses **FAISS (Semantic)** for the actual "Redundancy Reject" decision (Threshold: 0.85).

## 🔄 Relationship to Pipeline
- **Upstream:** Receives cleaned data (`seed_raw.jsonl`) and metadata from **NODE 1 (Intake)**.
- **Downstream:** Outputs raw synthetic JSON to **NODE 3 (Split Data)**. Now fully integrated via `run_full_pipeline.py`.

## ✨ Latest Enhancements
- **Flexible Loading:** Updated to support both `.json` and `.jsonl` seed files automatically.
- **Orchestration:** Chained with Node 1 and Node 3 for "One-Click" data generation.

## 🚀 Next Steps
- Expand `node2/` to support other task types (Extraction, NER) as defined in the master diagram.
- Ensure efficient batching for very large target counts.
