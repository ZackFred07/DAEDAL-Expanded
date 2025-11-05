from dllm_eval.models.huggingface import HFLM
from dllm_eval.api.registry import register_model

# Use like: --model LLaMA --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct
@register_model("LLaMA", "llama", "meta-llama")
class LLaMAHF(HFLM):
    """Alias wrapper that reuses the HF-backed LM (HFLM)."""
    pass
