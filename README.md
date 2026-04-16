# SLM Fine-tune Auto Config (Classification Demo) 🚀

This repository contains a modular pipeline for fine-tuning Small Language Models (SLMs) on Thai legal document classification tasks. It leverages **Oumi** for the training execution and **Distilabel** for synthetic data generation.

## 🎯 Key Features
- **Auto-Config Generation**: Automatically creates Oumi-compatible YAML configs for SFT and HP-Tuning.
- **Interchangeable Models**: Optimized for `Qwen2.5-0.5B-Instruct` for fast and light inference.
- **Dynamic Labeling**: Inference and evaluation snippets are domain-agnostic; they adapt to your dataset labels automatically using fuzzy matching.
- **Interative Playground**: A Gradio-based interface to test your fine-tuned adapters in real-time.
- **Vast.ai Ready**: Includes automation scripts for syncing data and executing training on remote GPU clusters.

## 📂 Repository Overview
- `src/slm_auto_config/node3`: Data Splitting & Conversion (Standard Chat Template).
- `src/slm_auto_config/node4`: Hyperparameter & Config Generation (Oumi YAMLs).
- `src/slm_auto_config/node5`: Training Orchestration & Best-Trial Selection.
- `src/slm_auto_config/node6`: Interactive Inference Playground (Gradio).
- `src/slm_auto_config/node7`: Detailed Evaluation & PDF Reporting.

## 🚀 Getting Started
Check out **[VAST_AI_DEPLOYMENT_GUIDE.md](file:///VAST_AI_DEPLOYMENT_GUIDE.md)** for step-by-step instructions on training and deploying to the cloud.

---
*Maintained by Park - Part of the NECTEC AI Engineering workflow.*
