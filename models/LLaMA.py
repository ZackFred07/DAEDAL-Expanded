# models/LLaMA.py
from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class LLaMAConfig:
    pretrained: str
    dtype: str = "bfloat16"  # or "float16" on V100 if OOM
    device: str = "cuda"

class LLaMA:
    def __init__(self, pretrained: str, dtype: str = "bfloat16", device: str = "cuda", **kwargs):
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        self.tok = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def generate(self, prompts: List[str], gen_kwargs: Dict[str, Any]) -> List[str]:
        # supports chat template if caller sent messages already formatted
        inputs = self.tok(prompts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k,v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        texts = self.tok.batch_decode(out, skip_special_tokens=True)
        return texts
