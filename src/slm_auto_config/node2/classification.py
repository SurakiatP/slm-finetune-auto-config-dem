import math
import random
import os
import json
import re
import logging
from typing import List, Dict, Any

from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, Step, StepInput
from distilabel.steps.tasks import TextGeneration
from distilabel.llms import OpenAILLM
from openai import OpenAI

from .base import BaseSDGEngine
from .models import JudgeOutput, SDGRules, GeneratorBatchOutput
from ..utils import load_json, load_jsonl

logger = logging.getLogger(__name__)

# ============================================================================
# UNMODIFIED LEGACY PROMPTS
# ============================================================================

GENERATOR_SYSTEM_PROMPT = """[Role]
You are a Senior Synthetic Data Engineer specializing in generating high-quality datasets for Small Language Model (SLM) fine-tuning.

[Task]
Your mission is to produce highly specific, diverse, and accurate text classifications that match a target label perfectly.

[Context]
High data fidelity is critical. The generated text must exhibit the nuances, vocabulary, and structural variations characteristic of authentic human-written content. Ensure maximum variation across multiple generated samples.

[Few-shot Guideline]
When provided with seed examples, capture their stylistic essence (tone, length, vocabulary) while ensuring all your generated outputs are 100% unique from the seeds and entirely distinct from one another.

[Output Instructions]
1. Brainstorm and generate EXACTLY 5 highly diverse candidate texts.
2. Ensure each candidate has a significantly different tone, perspective, professional terminology, length, or structure from the others.
3. Output ONLY a valid JSON object matching this exact schema:
{
  "results": [
    "diverse candidate text 1...",
    "diverse candidate text 2...",
    "diverse candidate text 3...",
    "diverse candidate text 4...",
    "diverse candidate text 5..."
  ]
}
DO NOT include any markdown formatting, <think> tags, explanations, or conversational text. Output only the pure JSON string.
"""

UNIFIED_GENERATOR_TEMPLATE = """Task Description: {{ task_description }}
Target Category/Label: '{{ label }}'

[Constraint: Diversity/Difficulty]
Rule: {{ diversity_rule }}
Difficulty: {{ difficulty }}

[Style Reference: Seed Examples (DO NOT COPY)]
{% if examples %}
{% for ex in examples %}
- {{ ex }}
{% endfor %}
{% else %}
No specific examples provided. Please generate highly creative and original examples for the '{{ label }}' category that are distinct from the general task.
{% endif %}

Please provide EXACTLY 5 unique text examples that fulfill the requirements of the '{{ label }}' label.
CRITICAL DIVERSITY CHECK: Each of the 5 examples MUST be drastically different from the others. Do not just change a few words. Create completely different scenarios, writing styles, or contexts for each one of them while strictly adhering to the Constraint and Label.

Output ONLY a raw JSON object containing an array of 5 strings under the "results" key.
DO NOT include any introductions, output candidate numbers, brainstorming, or internal evaluation. If you include any text outside the JSON structure, the entire output will be rejected."""

JUDGE_TEMPLATE = """Evaluate the following generated text for a classification task.
Task Description: {{ task_description }}
Generated Text: {{ cleaned_text }}
Target Label: {{ label }}

# Definition of 'unknown' Label:
If the Target Label is 'unknown', it means the text should be irrelevant, out-of-scope, or nonsensical relative to the Task Description provided.

# Evaluation Metrics (0.0 to 1.0):
1. Fidelity: Does the core meaning of the text a perfect match for the category '{{ label }}'?
2. Naturalness: Is the review fluent and realistic? (For 'unknown', it should just be coherent text unless it's intended to be gibberish).
3. Utility: Is this a high-quality example for training a classifier?

Output ONLY a JSON object matching this schema:
{
  "fidelity": float,
  "naturalness": float,
  "utility": float,
  "reasoning": "string"
}
Do not include any other text. Output scores MUST be a FLOAT between 0.0 and 1.0 (e.g., 0.85). DO NOT use fractions (like 9/10), strings, or negative values. Be extremely strict."""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_multiple_outputs(text: str) -> List[str]:
    if not text:
        return []
    text = re.sub(r'<(think|reasoning)>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = text.strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
        clean_text = re.sub(r"\s*```$", "", clean_text)
    try:
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = clean_text[start_idx:end_idx+1]
            validated_data = GeneratorBatchOutput.model_validate_json(json_str)
            return [res.strip() for res in validated_data.results if res.strip()]
    except Exception as e:
        logger.warning(f"Error parsing JSON from Generator: {e} | Raw text: {text[:150]}...")
        results = re.findall(r'"([^"]+)"', clean_text)
        filtered_results = [r for r in results if r != "results" and len(r.strip()) > 10]
        if filtered_results:
            return filtered_results
    return []

# ============================================================================
# DISTILABEL CUSTOM STEPS
# ============================================================================

class CleanTextStep(Step):
    @property
    def inputs(self) -> List[str]:
        return ["generated_text"]

    @property
    def outputs(self) -> List[str]:
        return ["cleaned_text"]

    def process(self, inputs: StepInput):
        exploded_outputs = []
        for item in inputs:
            cleaned_list = extract_multiple_outputs(item.get("generated_text", ""))
            if not cleaned_list:
                new_item = item.copy()
                new_item["cleaned_text"] = ""
                exploded_outputs.append(new_item)
                continue
                
            for extracted_text in cleaned_list:
                new_item = item.copy()
                new_item["cleaned_text"] = extracted_text
                exploded_outputs.append(new_item)
        yield exploded_outputs

# ============================================================================
# CLASSIFICATION STRATEGY
# ============================================================================

class ClassificationSDGGenerator(BaseSDGEngine):
    
    def generate_sdg_rules(self, client: OpenAI, model_name: str, labels: List[str]) -> SDGRules:
        logger.info(f"🚀 กำลังใช้ Meta-Prompting เพื่อสร้างกฎสำหรับงาน: {self.task_description[:50]}...")
        
        prompt = f"""You are a Senior Data Engineer. Your task is to brainstorm "Diversity Rules" for a Synthetic Data Generation pipeline.
These rules will be used to guide an LLM to generate high-quality, diverse, and realistic training data for a classification task.

[Task Description]
{self.task_description}

[Target Labels]
{", ".join(labels)}

[Requirements]
1. Diversity Rules: These rules should encourage the generator to use different perspectives, professional jargon, tones, or focus on specific nuances of the task. (e.g., 'Write from the perspective of an expert', 'Focus on edge cases like X').
2. Unknown Rules: These rules should create 'out-of-distribution' data that is NOT related to the main task but might be common noise (e.g., 'General greetings', 'Unstructured chatter about weather', 'Technical manuals for unrelated items').

Please provide a JSON object following this schema:
{{
  "diversity_rules": ["rule1", "rule2", ...],
  "unknown_diversity_rules": ["rule1", "rule2", ...]
}}
Output ONLY the JSON object.
"""
        response = client.chat.completions.create(
            model=model_name, 
            messages=[{"role": "system", "content": "You are a Senior Data Engineer. Return ONLY JSON."}, 
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        raw_json = response.choices[0].message.content
        clean_json = raw_json.strip()
        if clean_json.startswith("```"):
            clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
            clean_json = re.sub(r"\s*```$", "", clean_json)

        return SDGRules.model_validate_json(clean_json)

    def run(self, seed_data_path: str, output_path: str, api_kwargs: Dict[str, str]):
        logger.info(f"1. Loading Seed Data from {seed_data_path}...")
        if not os.path.exists(seed_data_path):
            logger.error(f"Error: Seed file not found at {seed_data_path}")
            return

        ext = os.path.splitext(seed_data_path)[1].lower()
        if ext == '.jsonl':
            seed_data = load_jsonl(seed_data_path)
        else:
            seed_data = load_json(seed_data_path)

        if not seed_data:
            logger.error(f"Error: Could not load data from {seed_data_path}")
            return

        # 2. Extract Labels and Setup Classes
        label_examples = {}
        for item in seed_data:
            lbl = item.get("label")
            txt = item.get("text")
            if lbl and txt:
                if lbl not in label_examples:
                    label_examples[lbl] = []
                label_examples[lbl].append(txt)

        unique_labels_seed = list(label_examples.keys())
        num_classes_seed = len(unique_labels_seed)
        
        if num_classes_seed == 0:
            logger.error("Error: No valid labels found in Seed Data")
            return

        target_unknown = math.ceil(self.target_count * 0.10)
        target_others = self.target_count - target_unknown
        target_per_class = math.ceil(target_others / num_classes_seed)
        
        unique_labels = unique_labels_seed + ["unknown"]
        label_examples["unknown"] = [] 

        logger.info(f"Found {num_classes_seed} labels. Target: {self.target_count} items total.")
        
        target_counts_map = {lbl: target_per_class for lbl in unique_labels_seed}
        target_counts_map["unknown"] = target_unknown
        collected_counts = {lbl: 0 for lbl in unique_labels} 
        collected_output = []

        # 3. Setup global deduplication via base engine
        seed_texts = [item.get("text") for item in seed_data if item.get("text")]
        self.inject_seed_into_index(seed_texts)

        # 4. APIs
        base_url = api_kwargs.get("base_url")
        api_key = api_kwargs.get("api_key")
        model_name = api_kwargs.get("model_name", "qwen/qwen3-235b-a22b-2507")
        rule_model_name = api_kwargs.get("rule_model_name", "qwen/qwen3.6-plus")
        judge_model_name = api_kwargs.get("judge_model_name", "openai/gpt-4o-mini")

        client = OpenAI(base_url=base_url, api_key=api_key)
        
        try:
            sdg_rules = self.generate_sdg_rules(client, rule_model_name, unique_labels)
            diversity_rules = sdg_rules.diversity_rules
            unknown_diversity_rules = sdg_rules.unknown_diversity_rules
        except Exception as e:
            logger.warning(f"⚠️ Meta-Prompting failed, using fallback: {e}")
            diversity_rules = ["Focus on specific details.", "Use formal language.", "Consider different perspectives."]
            unknown_diversity_rules = ["Generic chatter.", "Unrelated news snippets."]

        difficulties = ["easy", "medium", "hard", "complex-structure"]
        
        loop_count = 1
        
        # 5. Iterative Generation
        while sum(collected_counts.values()) < self.target_count and loop_count <= self.max_loops:
            logger.info(f"\n{'='*40}\n 🔁 Loop {loop_count}/{self.max_loops}\n{'='*40}")
            
            pipeline_inputs = []
            
            for label in unique_labels:
                needed = target_counts_map[label] - collected_counts[label]
                if needed <= 0:
                    continue 
                    
                generate_quota = max(1, math.ceil((needed / 4.0) * 1.5))
                
                for _ in range(generate_quota):
                    if label == "unknown":
                        examples = []
                        div_rule = random.choice(unknown_diversity_rules)
                    else:
                        examples = random.sample(label_examples[label], min(3, len(label_examples[label])))
                        div_rule = random.choice(diversity_rules)

                    pipeline_inputs.append({
                        "task_description": self.task_description,
                        "label": label,
                        "examples": examples,
                        "difficulty": random.choice(difficulties),
                        "diversity_rule": div_rule 
                    })

            random.shuffle(pipeline_inputs)

            if not pipeline_inputs:
                break

            with Pipeline(name=f"classification-sdg-loop-{loop_count}") as pipeline:
                loader = LoadDataFromDicts(data=pipeline_inputs, batch_size=100)
                
                generator = TextGeneration(
                    name="generate_text",
                    llm=OpenAILLM(
                        model=model_name, 
                        base_url=base_url, 
                        api_key=api_key, 
                        generation_kwargs={"max_new_tokens": 4096, "temperature": 0.7, "response_format": {"type": "json_object"}}
                    ),
                    system_prompt=GENERATOR_SYSTEM_PROMPT,
                    template=UNIFIED_GENERATOR_TEMPLATE,
                    columns=["task_description", "label", "examples", "difficulty", "diversity_rule"],
                    output_mappings={"generation": "generated_text"},
                    input_batch_size=100
                )
                
                cleaner = CleanTextStep(name="clean_text", input_batch_size=100)
                
                judge = TextGeneration(
                    name="judge_text",
                    llm=OpenAILLM(
                        model=judge_model_name, 
                        base_url=base_url, 
                        api_key=api_key, 
                        generation_kwargs={"temperature": 0.0, "response_format": {"type": "json_object"}}
                    ),
                    template=JUDGE_TEMPLATE,
                    columns=["task_description", "label", "cleaned_text"],
                    output_mappings={"generation": "judge_raw_output"},
                    input_batch_size=100
                )
                
                loader >> generator >> cleaner >> judge

            distiset = pipeline.run(use_cache=False)
            ds = distiset["default"]["train"]
            
            all_texts = [row.get("cleaned_text", "") for row in ds]
            valid_indices = [idx for idx, txt in enumerate(all_texts) if txt and txt.strip() and txt.lower() != "none"]
            valid_texts = [all_texts[idx] for idx in valid_indices]
            
            vector_map = {}
            if valid_texts:
                batch_vectors = self.embedding_model.encode(valid_texts, normalize_embeddings=True, show_progress_bar=False)
                vector_map = {idx: batch_vectors[i] for i, idx in enumerate(valid_indices)}

            added_in_loop = 0
            
            for i, row in enumerate(ds):
                lbl = row["label"]
                text = row.get("cleaned_text", "")
                
                if collected_counts[lbl] >= target_counts_map[lbl]:
                    continue
                    
                if i not in vector_map:
                    continue

                judge_raw_output = row.get("judge_raw_output", "{}")
                
                try:
                    judge_data = JudgeOutput.model_validate_json(judge_raw_output)
                    overall_score = (judge_data.fidelity * 0.4) + (judge_data.naturalness * 0.3) + (judge_data.utility * 0.3)
                    
                    if overall_score < self.threshold:
                        continue 
                except Exception:
                    continue
                    
                is_redundant, max_sim = self.is_semantically_redundant(vector_map[i])
                if is_redundant:
                    continue

                self.add_generated_to_index(text, vector_map[i])
                
                collected_output.append({"text": text, "label": lbl})
                collected_counts[lbl] += 1
                added_in_loop += 1
                
                if sum(collected_counts.values()) >= self.target_count:
                    break
                    
            logger.info(f"Loop {loop_count} finished. Added: {added_in_loop} items.")
            loop_count += 1

        # 6. Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(collected_output, f, ensure_ascii=False, indent=2)
            
        logger.info(f"SDG Completed. Saved {len(collected_output)} items to {output_path}")

