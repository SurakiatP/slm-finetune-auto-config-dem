import os
import torch
import shutil
import logging
import subprocess
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from .base import BaseExporter
from .generator import ModelCardGenerator, MetadataGenerator

logger = logging.getLogger(__name__)

class ClassificationExporter(BaseExporter):
    """
    Concrete exporter for classification models.
    Handles merging LoRA weights and converting to various formats.
    """
    def export(self, formats: List[str]) -> Dict[str, str]:
        results = {}
        
        # 1. Base Export: SafeTensors (Merged)
        # This is required as a source for other formats
        logger.info("Merging LoRA weights into base model...")
        merged_dir = self._export_safetensors()
        results["safetensors"] = merged_dir
        
        # 2. Optional: GGUF
        if "gguf" in formats:
            logger.info("Converting to GGUF format...")
            gguf_path = self._export_gguf(merged_dir)
            results["gguf"] = gguf_path
            
        # 3. Optional: ONNX
        if "onnx" in formats:
            logger.info("Exporting to ONNX format...")
            onnx_dir = self._export_onnx(merged_dir)
            results["onnx"] = onnx_dir

        # 4. Documentation Generation
        logger.info("Generating model card and metadata...")
        gen = ModelCardGenerator(self.run_id)
        meta = MetadataGenerator(self.run_id)
        
        gen.generate(self.export_dir, results)
        meta.generate(self.export_dir, results)
        
        return results

    def _export_safetensors(self) -> str:
        """Merges LoRA adapter with base model and saves as HF SafeTensors."""
        output_dir = f"{self.export_dir}/safetensors"
        
        # Load logic similar to Node 6
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if self.adapter_path:
            model = PeftModel.from_pretrained(model, self.adapter_path)
            model = model.merge_and_unload()
        
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        return output_dir

    def _export_gguf(self, source_dir: str) -> str:
        """
        Converts HF model to GGUF using llama.cpp scripts.
        Note: Requires llama.cpp scripts to be present.
        """
        output_file = f"{self.export_dir}/gguf/model.gguf"
        # We assume the user has cloned llama.cpp or we provide a conversion call
        # For MVP, we will generate the command string that the user can run on Vast.ai
        # because cloning llama.cpp inside this script might be too slow/complex.
        command = f"python3 conversion_scripts/convert-hf-to-gguf.py {source_dir} --outfile {output_file}"
        
        # We write this command to a helper script in the export folder
        with open(f"{self.export_dir}/gguf/convert_cmd.sh", "w") as f:
            f.write(f"#/bin/bash\n{command}\n")
            
        logger.info(f"GGUF conversion command prepared at {self.export_dir}/gguf/convert_cmd.sh")
        return output_file

    def _export_onnx(self, source_dir: str) -> str:
        """Exports the merged model to ONNX using Hugging Face Optimum."""
        output_dir = f"{self.export_dir}/onnx"
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
            # This triggers the conversion
            model = ORTModelForCausalLM.from_pretrained(source_dir, export=True)
            model.save_pretrained(output_dir)
            return output_dir
        except ImportError:
            logger.error("Optimum not installed. Skipping ONNX export.")
            return "SKIPPED (Optimum missing)"
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return f"FAILED: {str(e)}"
