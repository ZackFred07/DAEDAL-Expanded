import logging
import os
from datetime import timedelta
from typing import Dict, List, Literal, Optional, Tuple, Union, TypeVar
from huggingface_hub import PyTorchModelHubMixin
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from datasets import Dataset
from accelerate.utils import get_max_memory
from packaging import version
from tqdm import tqdm
import torch.distributed as dist
import torch.distributions as dists
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from dllm_eval.api.instance import Instance
from dllm_eval.api.model import LM, TemplateLM
from dllm_eval.api.registry import register_model
from dllm_eval.models.utils import get_dtype, configure_pad_token
from dllm_eval.models.modeling_diffullama import replace_attention_mask
from dllm_eval.utils import simple_parse_args_string


eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")


# TODO Could move this to the dllm_eval/models
class DiscreteDiffusionModel(nn.Module, PyTorchModelHubMixin):
    """
    diffusion model
    """

    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def __init__(self, model, config, tokenizer, device):
        super().__init__()
        if isinstance(
            model, str
        ):  # if use pre-trained model name from huggingface, e.g., gpt2, gpt2-medium.
            config_pt = AutoConfig.from_pretrained(model)
            self.model = AutoModelForCausalLM.from_config(config_pt)
        else:
            self.model = model
        self.config = config
        self.embed_dim = self.config.hidden_size
        self.hidden_dim = self.config.hidden_size
        if self.model.get_input_embeddings().weight.size(0) != len(tokenizer):
            self.model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=2)
        self.vocab_size = self.model.get_input_embeddings().weight.size(0)
        if getattr(self.config, "model_type", None) == "gpt2":
            self.embed_tokens = self.model.transformer.wte
            self.denoise_model = (
                self.model.transformer
            )  # use inputs_embeds instead of input_ids in forward function
            for gpt2block in self.model.transformer.h:
                gpt2block.attn.bias.fill_(True)  # remove causal mask
            self.lm_head = self.model.lm_head
            del self.denoise_model.wte
        elif getattr(self.config, "model_type", None) == "llama":
            self.embed_tokens = self.model.model.embed_tokens
            self.denoise_model = self.model.model
            self.lm_head = self.model.lm_head
            del self.denoise_model.embed_tokens
        del self.model
        self.device = device

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_embeds(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids,
        attention_mask,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        denoise the input
        """
        x_embed = self.get_embeds(input_ids)

        x = self.denoise_model(
            inputs_embeds=x_embed, attention_mask=attention_mask, return_dict=False
        )[0]

        logits = self.get_logits(x)

        return logits


"""
Code on the outside
    prefix = [tokenizer.bos_token_id] + tokenizer.encode("Today is a wonderful day,")

    src_mask = [1]*len(prefix)+[0]*(gen_len-len(prefix))
    x0 = prefix + [0]*(gen_len-len(prefix))

    inputs = {
        "input_ids": torch.tensor([x0]),
        "src_mask": torch.tensor([src_mask])
    }
    res = generate_samples(model, args, tokenizer, inputs, verbose=args.verbose)
    pred = tokenizer.decode(res.tolist()[0])
    print(pred)
"""


# TODO Zack
@torch.no_grad()
def generate(
    model: DiscreteDiffusionModel,
    prompt,  # (batch, prompt)
    tokenizer,
    steps=64,
    gen_length=128,
    remasking="low_confidence",
    mask_id=811,
    logits_temp=0.9,
    topp_temp=0.9,
    shift=True,  # Should leave this as true, essentially what we are doing is cheesing the oringal AR model to do same token prediction
):
    """
    select 1/T% tokens to denoise at each step
    """
    # model.cuda()
    model.eval()

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=prompt.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    src_mask = x != mask_id

    x_embed = model.get_embeds(x)
    seq_len = x.size(1)
    batch_size = x.size(0)
    attention_mask = get_anneal_attn_mask(
        seq_len, batch_size, dtype=x_embed.dtype, device=x.device, attn_mask_ratio=1.0
    )  # all 0

    maskable_mask = ~src_mask

    # All tokens are masked
    xt = x

    # Denoising step at t = T
    logits = model(xt, attention_mask=attention_mask)
    filter_logits = top_p_logits(
        logits / logits_temp, p=topp_temp
    )  # Logits where cumulative prob > p (the tail of the distribution) get their logits set to -inf; nucleas sampling filtering
    scores = torch.log_softmax(filter_logits, dim=-1)
    x0 = dists.Categorical(
        logits=scores
    ).sample()  # Sample from a categorical distirbution defined by scores for each toekn position; yields an initial guess for the sequence
    x0_scores = torch.gather(scores, -1, x0.unsqueeze(-1)).squeeze(
        -1
    )  # Log prob of the sampled token at each position, pulled with gather

    if shift:
        # shifts x0 to the right by one, and keep the original first token from x; used to make predictions "aligned" differently (predict token t+1 given t)
        #### deal with shift, left most token will be replaced anyway
        x0 = torch.cat([x[:, 0:1], x0[:, :-1]], dim=1)
        x0_scores = torch.cat([x0_scores[:, 0:1], x0_scores[:, :-1]], dim=1)

    #### replace output of non-[MASK] positions with xt
    x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])

    # Diffusion loop gradually fixing tokens
    for t in range(steps - 1, 0, -1):  # t from T-1 to 1
        with torch.no_grad():
            #### select rate% tokens to be still [MASK]
            p_to_x0 = 1 / (
                t + 1
            )  # Fraction of currently maskable tokens that will be "fixed" at this step

            # Wghat positions become "fixed" now
            masked_to_x0 = maskable_mask & (
                torch.rand_like(x0, dtype=torch.float) < p_to_x0
            )
            xt.masked_scatter_(
                masked_to_x0, x0[masked_to_x0]
            )  # Inserts the predicted tokens into xt at those positions, so xt becomes less masked as time goes on
            maskable_mask = maskable_mask.masked_fill(
                masked_to_x0, False
            )  # marks positions as no longer maskable as they have been predicted

            # Run for this step t
            logits = model(xt, attention_mask=attention_mask)
            filter_logits = top_p_logits(logits / logits_temp, p=topp_temp)
            scores = torch.log_softmax(filter_logits, dim=-1)
            x0 = dists.Categorical(logits=scores).sample()
            x0_scores = torch.gather(scores, -1, x0.unsqueeze(-1)).squeeze(-1)

            if shift:
                #### deal with shift, left most token will be replaced anyway
                x0 = torch.cat([x[:, 0:1], x0[:, :-1]], dim=1)
                x0_scores = torch.cat([x0_scores[:, 0:1], x0_scores[:, :-1]], dim=1)

            # replace output of non-[MASK] positions with xt
            x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])

            # This pattern continues until steps have been completed

    # Shift back
    if shift:
        x0 = x0[:, 1:]

    return x0


@register_model("diffullama")
class DiffuLLaMA(TemplateLM):

    def __init__(
        self,
        pretrained: str,
        assistant_prefix: str = "",
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 1,
        mask_id: int = 811,
        add_bos_token: Optional[bool] = True,
        prefix_token_id: Optional[int] = None,
        **kwargs,
    ):

        super().__init__()
        replace_attention_mask()

        self.device = torch.device(device)
        self.assistant_prefix = assistant_prefix
        self.batch_size = batch_size
        self.mask_id = mask_id
        self.add_bos_token = True

        self.truncation = kwargs.get("truncation", None)
        self.remasking = kwargs.get("remasking", "low_confidence")

        self.escape_until = False
        self.is_first_inference = True

        # ---- load tokenizer & model ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            trust_remote_code=True,
            use_fast=True,
        )

        self.tokenizer.chat_template = (
            "{% set system = '' %}"
            "{% for m in messages %}"
            "{% if m['role'] == 'system' %}{% set system = m['content'] | trim %}{% endif %}"
            "{% endfor %}"
            "{% for m in messages %}"
            "{% if m['role'] == 'user' %}"
            "{% if loop.first %}"
            "{{ bos_token + '[INST] ' + (system + '\\n\\n' if system else '') + (m['content']|trim) + ' [/INST]' }}"
            "{% else %}"
            "{{ '[INST] ' + (m['content']|trim) + ' [/INST]' }}"
            "{% endif %}"
            "{% elif m['role'] == 'assistant' %}"
            "{{ ' ' + (m['content']|trim) + ' ' }}"
            "{% endif %}"
            "{% endfor %}"
        )

        self.mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if self.mask_id is None:
            # fall back if needed
            self.mask_id = self.tokenizer.eos_token_id

        config = AutoConfig.from_pretrained(
            pretrained,
            _attn_implementation="eager",
            trust_remote_code=True,
        )

        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        # This assumes you registered DiffuLLaMA with AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="cuda:1",  # TODO Change back before pushing
        )

        self.model = DiscreteDiffusionModel(
            model=model, config=config, tokenizer=self.tokenizer, device="cuda"
        ).to(self.device)

        self.model.eval()

        self._max_length = int(
            kwargs.get("max_length", getattr(config, "max_position_embeddings", 2048))
        )
        self._max_batch_size = batch_size
        self.custom_prefix_token_id = prefix_token_id

    @property
    def prefix_token_id(self):
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def tok_encode(self, s: str, **kwargs) -> List[int]:
        return self.tokenizer.encode(s, add_special_tokens=False, **kwargs)

    def tok_decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        bar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Running generate_until requests",
        )
        ds_data = [{"text": req.args[0]} for req in requests]
        ds = Dataset.from_list(ds_data)
        gen_kwargs = requests[0].args[1]

        for batch in ds.iter(batch_size=int(self.batch_size)):
            contexts = batch["text"]
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                truncation=self.truncation,
            )
            prompt_length = context_enc.shape[1]
            out_full = generate(
                model=self.model,
                prompt=context_enc,
                tokenizer=self.tokenizer,
                steps=gen_kwargs.get("steps", gen_kwargs.get("gen_length", 128)),
                gen_length=gen_kwargs.get("gen_length", 128),
                remasking=gen_kwargs.get("remasking", self.remasking),
                mask_id=self.mask_id,
            )
            generated_tokens = out_full[:, prompt_length:]
            cont_toks_list = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=False
            )
            if self.rank == 0 and self.is_first_inference:
                eval_logger.info("\n--- First Batch Inference (Rank 0) ---")
                for i, (question, answer) in enumerate(zip(contexts, cont_toks_list)):
                    eval_logger.info(f"Question {i+1}: {question}")
                    eval_logger.info(f"Answer   {i+1}: {answer}\n")
                eval_logger.info("------------------------------------")
                self.is_first_inference = False
            for s in cont_toks_list:
                if not self.escape_until:
                    stop_sequences = gen_kwargs.get("until", [])
                    if stop_sequences:
                        for term in stop_sequences:
                            if len(term) > 0:
                                s = s.split(term)[0]
                res.append(s)
                bar.update(1)
        bar.close()
        return res

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        raise NotImplementedError

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if self.assistant_prefix:
            chat_templated += self.assistant_prefix
        return chat_templated

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ):
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
        )

        self.tokenizer.padding_side = old_padding_side
        return encoding["input_ids"].to(self.device), encoding["attention_mask"].to(
            self.device
        )

    def tok_decode(self, tokens, skip_special_tokens=False):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)


def get_anneal_attn_mask(seq_len, bsz, dtype, device, attn_mask_ratio):
    mask = torch.full((seq_len, seq_len), 0, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)
    causal_mask = mask.to(dtype)

    random_mask = torch.bernoulli(
        torch.full((seq_len, seq_len), 0.0, device=device) + attn_mask_ratio
    )

    anneal_mask = torch.logical_or(causal_mask, random_mask)
    expanded_mask = anneal_mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.to(dtype)

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def top_p_logits(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # import pdb; pdb.set_trace();
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits
