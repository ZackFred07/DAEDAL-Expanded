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


# TODO Zack may the force be with you on this one
@torch.no_grad()
def generate(
    model: DiscreteDiffusionModel,
    prompt,  # (batch, prompt)
    tokenizer,
    steps=64,  # TODO do we care to keep this?
    initial_gen_length=64,
    max_gen_length=2048,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    high_conf_threshold=0.90,
    low_conf_threshold=0.10,
    expansion_factor=8,
    mask_id=811,
    eos_token_id=2,
    eos_confidence_threshold=0.5,
    expand_eos_confidence_threshold=0.9,
    eos_check_tokens=32,
    logits_temp=0.9,
    topp_temp=0.9,
    shift=True,  # Should leave this as true, essentially what we are doing is cheesing the oringal AR model to do same token prediction
):
    # Helper function to calculate EOS confidence
    def _calculate_eos_confidence(
        logits, total_lengths, prompt_length, eos_check_tokens
    ):
        if eos_token_id is None:
            return torch.zeros(logits.shape[0], device=logits.device)

        # Convert Logits to probabilities
        confidences = F.softmax(logits, dim=-1)
        predicted_tokens = torch.argmax(logits, dim=-1)

        batch_eos_confidences = []
        for i in range(logits.shape[0]):
            # Go through each batch from total_lengths [i] to prompt_length -1
            eos_confs_for_avg = []
            start_scan_pos = total_lengths[i].item() - 1
            end_scan_pos = prompt_length - 1

            for pos in range(start_scan_pos, end_scan_pos, -1):
                # Collect up to eos_check_tokens positions where predicted_tokens[i,pos] == eos_token_id
                if len(eos_confs_for_avg) >= eos_check_tokens:
                    break
                # Record the EOS probability
                if predicted_tokens[i, pos] == eos_token_id:
                    eos_confs_for_avg.append(confidences[i, pos, eos_token_id].item())
            avg_conf = sum(eos_confs_for_avg) / eos_check_tokens
            batch_eos_confidences.append(avg_conf)

        # A Batch vector of those average EOS confidences
        return torch.tensor(batch_eos_confidences, device=logits.device)

    # model.cuda()
    model.eval()
    batch_size = prompt.shape[0]
    device = prompt.device
    prompt_length = prompt.shape[1]
    assert eos_token_id is not None
    gen_lengths = torch.full((batch_size,), initial_gen_length, dtype=torch.long, device=device)
    x = torch.full(
        (batch_size, prompt_length + initial_gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_length] = prompt.clone() # TODO Found that we keep reusing the bos token need to fix that
    src_mask = x != mask_id

    x_embed = model.get_embeds(x)
    seq_len = x.size(1)
    batch_size = x.size(0)
    # attention_mask = get_anneal_attn_mask(
    #     seq_len, batch_size, dtype=x_embed.dtype, device=x.device, attn_mask_ratio=1.0
    # )  # all 0

    # maskable_mask = ~src_mask

    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
        print("[Stage-1] Initial Length Adjustment")
    while True:

        total_lengths = prompt_length + gen_lengths
        max_len_pre = x.shape[1]
        # Build an attention mask up to total lengths
        # arange_tensor_pre = torch.arange(max_len_pre, device=device).expand(batch_size, -1)
        # attention_mask_pre = (arange_tensor_pre < total_lengths.unsqueeze(1)).long()
        attention_mask_pre = get_anneal_attn_mask(
            total_lengths,
            batch_size,
            dtype=x_embed.dtype,
            device=x.device,
            attn_mask_ratio=1.0,
        )
        logits_pre = model(x, attention_mask=attention_mask_pre)

        # Compute EOS for each Sequence
        batch_eos_confidences = _calculate_eos_confidence(logits_pre, total_lengths, prompt_length, eos_check_tokens)
        # Decide which sequence need mroe space
        sequences_to_expand = (batch_eos_confidences < eos_confidence_threshold) & (gen_lengths < max_gen_length)

        if not sequences_to_expand.any():
            # No sequence needs expansion
            if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                print(f"All sequences' EOS confidence reach the threshold {eos_confidence_threshold} or max length.")
            break
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            print(f"Some sequences' EOS confidence ({[round(c.item(), 4) for c in batch_eos_confidences]}) < {eos_confidence_threshold}. Expand initial length.")

        # Increase their gen_lengths by expansion factor (capped by max_gen_length)
        max_new_gen_len = gen_lengths[sequences_to_expand].max().item()
        new_gen_lengths = gen_lengths.clone()
        # Compute new generation length
        new_gen_lengths[sequences_to_expand] = torch.clamp(gen_lengths[sequences_to_expand] + expansion_factor, max=max_gen_length)
        if new_gen_lengths.max() <= gen_lengths.max():
            # Check that max length is hit and break
            if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                print(f"WARNING: Cannot expand initial length further (already at max length: {max_gen_length}).")
            break
        # Build the new tensor
        max_new_total_len = prompt_length + new_gen_lengths.max()
        new_x_tensor = torch.full((batch_size, max_new_total_len), eos_token_id, dtype=torch.long, device=device)
        for i in range(batch_size):
            # Copy the existing tokens into the front
            original_total_len = prompt_length + gen_lengths[i].item()
            new_x_tensor[i, :original_total_len] = x[i, :original_total_len]
            if sequences_to_expand[i]:
                # Where sequence was expanded, fill new positions with mask_id
                new_total_len_i = prompt_length + new_gen_lengths[i].item()
                new_x_tensor[i, original_total_len : new_total_len_i] = mask_id
        # Update
        x = new_x_tensor
        gen_lengths = new_gen_lengths

    # Extend each sequence's generation elgnth by half of eos_check_tokens capped by max_gen_length
    new_gen_lengths_with_eos = gen_lengths + int(eos_check_tokens/2)
    new_gen_lengths_with_eos = torch.clamp(new_gen_lengths_with_eos, max=max_gen_length)
    max_new_total_len = prompt_length + new_gen_lengths_with_eos.max()
    intermediate_x_tensor = torch.full((batch_size, max_new_total_len), eos_token_id, dtype=torch.long, device=device)
    for i in range(batch_size):
        # First gen_lengths[i] positions after the prompt are set to mask_id, rest is previously set eos_token_id
        intermediate_x_tensor[i, :prompt_length] = x[i, :prompt_length]
        intermediate_x_tensor[i, prompt_length:prompt_length+gen_lengths[i].item()] = mask_id
    x = intermediate_x_tensor
    gen_lengths = new_gen_lengths_with_eos

    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
        print(f"[Stage-2] Iterative Denoising and Mask Insertion")

    current_pos = torch.full((batch_size,), prompt_length, dtype=torch.long, device=device) # The starting index into the generation region for all sequences
    denoise_only_mode = torch.zeros(batch_size, dtype=torch.bool, device=device) # Sequence is at max length, true if shouldn't expand

    while (current_pos < prompt_length + gen_lengths).any():

        total_lengths = prompt_length + gen_lengths
        x_before_step = x.clone()

        for i in range(batch_size):
            if gen_lengths[i] >= max_gen_length and not denoise_only_mode[i]:
                if current_pos[i] < total_lengths[i]:
                    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                        print(f"Sequence {i} has reached the max length {max_gen_length}. Entering denoise-only mode.")
                    denoise_only_mode[i] = True

        max_len = x.shape[1]
        arange_tensor = torch.arange(max_len, device=device).expand(batch_size, -1)
        attention_mask = (arange_tensor < total_lengths.unsqueeze(1)).long()

        # TODO Forward Pass, Need to do Denoising step at t = T and the rest of the loop potentially
        # TODO Want to see what I can do with this
        xt = x
        logits = model(xt, attention_mask=attention_mask)

        filter_logits = top_p_logits(
            logits / logits_temp, p=topp_temp
        )  # Logits where cumulative prob > p (the tail of the distribution) get their logits set to -inf; nucleas sampling filtering
        confidences = torch.log_softmax(filter_logits, dim=-1)
        predicted_tokens = x0 = dists.Categorical(
            logits=confidences
        ).sample()  # Sample from a categorical distirbution defined by scores for each toekn position; yields an initial guess for the sequence
        predicted_confidences = x0_scores = torch.gather(confidences, -1, x0.unsqueeze(-1)).squeeze(
            -1
        )  # Log prob of the sampled token at each position, pulled with gather
        batch_eos_confidences = _calculate_eos_confidence(
            logits, total_lengths, prompt_length, eos_check_tokens
        )

        if shift:
            # shifts x0 to the right by one, and keep the original first token from x; used to make predictions "aligned" differently (predict token t+1 given t)
            #### deal with shift, left most token will be replaced anyway
            x0 = torch.cat([x[:, 0:1], x0[:, :-1]], dim=1)
            x0_scores = torch.cat([x0_scores[:, 0:1], x0_scores[:, :-1]], dim=1)

        currently_masked = x == mask_id

        high_conf_indices = (
            (x0_scores > high_conf_threshold)
            & (x == mask_id)
            & (x0 != mask_id)
        )

        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=device)

        # TODO Our inner diffusion loop
        for i in range(batch_size):
            if current_pos[i] >= total_lengths[i]: continue
            start_idx, end_idx = current_pos[i], min(current_pos[i] + block_length, total_lengths[i].item())

            if not high_conf_indices[i, start_idx:end_idx].any():
                # Consider items in the valid region
                valid_region = positions < total_lengths[i]
                if (high_conf_indices[i] & valid_region).any():
                    continue

                # Consider all currently masked tokens in region
                valid_fallback_mask = currently_masked[i] & valid_region
                if not valid_fallback_mask.any():
                    continue

                # Compute candidate confidences and tokens
                candidate_indices = torch.where(valid_fallback_mask)[0]
                candidate_confs = predicted_confidences[i, candidate_indices]
                candidate_tokens = predicted_tokens[i, candidate_indices]

                sorted_confs, sort_indices = torch.sort(candidate_confs, descending=True)
                best_idx_to_fill = -1
                for sorted_idx in sort_indices:
                    if candidate_tokens[sorted_idx] != mask_id:
                        best_idx_to_fill = candidate_indices[sorted_idx]; break
                if best_idx_to_fill != -1:
                    high_conf_indices[i, best_idx_to_fill] = True
                else:
                    stuck_logits = logits[i, candidate_indices]
                    # Masked tokens set to negative infinity
                    stuck_logits[:, mask_id] = -torch.inf
                    new_confidences = F.softmax(stuck_logits, dim=-1)
                    new_best_confs, new_best_tokens = torch.max(new_confidences, dim=-1)

                    # Pick the most confident non-mask token to be marked as high confidence (guarantee something gets denoised)
                    best_of_the_best_local_idx = torch.argmax(new_best_confs)
                    pos_to_fill = candidate_indices[best_of_the_best_local_idx]
                    token_to_fill = new_best_tokens[best_of_the_best_local_idx]
                    predicted_tokens[i, pos_to_fill] = token_to_fill
                    high_conf_indices[i, pos_to_fill] = True

            # Identify low confidence tokens to expands
            potential_expand_mask = (predicted_confidences < low_conf_threshold) & currently_masked & (~high_conf_indices)
            expand_indices = torch.zeros_like(x, dtype=torch.bool, device=device)
            for i in range(batch_size):
                if batch_eos_confidences[i] >= expand_eos_confidence_threshold or gen_lengths[i] >= max_gen_length: continue
                if denoise_only_mode[i] or current_pos[i] >= total_lengths[i]: continue

                # Create expansion points
                masked_candidates = torch.where(potential_expand_mask[i])[0]
                if len(masked_candidates) > 0:

                    candidate_confs = predicted_confidences[i, masked_candidates]
                    num_to_expand = min(1, len(masked_candidates))
                    if num_to_expand > 0:
                        _, lowest_conf_local_indices = torch.topk(candidate_confs, num_to_expand, largest=False)
                        indices_to_expand_global = masked_candidates[lowest_conf_local_indices]
                        expand_indices[i, indices_to_expand_global] = True

                # Apply fills
                fill_mask = high_conf_indices
                if not expand_indices.any():
                    x[fill_mask] = predicted_tokens[fill_mask]
                else:
                    x[fill_mask] = predicted_tokens[fill_mask]

                    # Calculate how much to expand
                    max_new_total_len = 0
                    temp_new_gen_lengths = gen_lengths.clone()
                    for i in range(batch_size):
                        expansion_count = expand_indices[i].sum().item()
                        if expansion_count > 0:
                            new_len = gen_lengths[i].item() + expansion_count * (
                                expansion_factor - 1
                            )
                            temp_new_gen_lengths[i] = min(new_len, max_gen_length)

                    # Compute new max total length and allocate new_x_tensor filled with EOS
                    max_new_total_len = prompt_length + temp_new_gen_lengths.max()
                    new_x_tensor = torch.full(
                        (batch_size, max_new_total_len),
                        eos_token_id,
                        device=device,
                        dtype=torch.long,
                    )
                    new_gen_lengths = torch.zeros_like(gen_lengths)

                    for i in range(batch_size):
                        if not expand_indices[i].any():
                            # Copy the old sequence as is
                            total_len = prompt_length + gen_lengths[i].item()
                            new_x_tensor[i, :total_len] = x[i, :total_len]
                            new_gen_lengths[i] = gen_lengths[i]
                            continue
                        write_ptr = prompt_length

                        new_x_tensor[i, :prompt_length] = x[
                            i, :prompt_length
                        ]  # Copy prompt
                        # Iterate over old generation region
                        for j in range(
                            prompt_length, prompt_length + gen_lengths[i].item()
                        ):
                            if write_ptr >= max_new_total_len:
                                break
                            # If marked for expansion write the expansion otherwise continue
                            if expand_indices[i, j]:
                                end_write = min(
                                    write_ptr + expansion_factor, max_new_total_len
                                )
                                new_x_tensor[i, write_ptr:end_write] = mask_id
                                write_ptr = end_write
                            else:
                                new_x_tensor[i, write_ptr] = x[i, j]
                                write_ptr += 1

                        new_gen_lengths[i] = write_ptr - prompt_length
                    # Update
                    x = new_x_tensor
                    gen_lengths = new_gen_lengths

                for i in range(batch_size):
                    total_len = total_lengths[i].item()
                    start_pos = current_pos[i].item()

                    if start_pos >= total_len:
                        continue

                    segment = x[i, start_pos:total_len]
                    masked = (segment == mask_id)

                    if masked.any():
                        # First index in this segment where we see a mask
                        first_rel_idx = torch.nonzero(masked, as_tuple=False)[0, 0].item()
                        current_pos[i] = start_pos + first_rel_idx
                    else:
                        # No masked tokens left in the valid region; we move to the end
                        current_pos[i] = total_len

                # State hasnt changed exit main loop
                if torch.equal(x, x_before_step):
                    if (
                        not (dist.is_available() and dist.is_initialized())
                        or dist.get_rank() == 0
                    ):
                        print(
                            f"WARNING: Sequence state is stagnant, forcing generation to end."
                        )
                    break

    current_pos = torch.full((batch_size,), prompt_length, dtype=torch.long, device=device) # The starting index into the generation region for all sequences
    denoise_only_mode = torch.zeros(batch_size, dtype=torch.bool, device=device) # Sequence is at max length, true if shouldn't expand

    # Shift back
    if shift:
        x0 = x0[:, 1:]

    # TODO Final output assembly
    final_outputs = []
    for i in range(batch_size):
        final_len = prompt_length + gen_lengths[i]
        final_outputs.append(x[i, :final_len])
    return final_outputs

    return x0


@register_model("diffullama_DAEDAL")
class DiffuLLaMA_DAEDAL(TemplateLM):

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
            out_list = generate(
                model=self.model,
                prompt=context_enc,
                tokenizer=self.tokenizer,
                initial_gen_length=gen_kwargs.get("initial_gen_length", 64),
                max_gen_length=gen_kwargs.get("max_gen_length", 2048),
                block_length=gen_kwargs.get("block_length", 32),
                temperature=gen_kwargs.get("temperature", 0.0),
                cfg_scale=gen_kwargs.get("cfg_scale", 0.0),
                high_conf_threshold=gen_kwargs.get("high_conf_threshold", 0.90),
                low_conf_threshold=gen_kwargs.get("low_conf_threshold", 0.10),
                expansion_factor=gen_kwargs.get("expansion_factor", 8),
                mask_id=self.mask_id,
                eos_token_id=self.eot_token_id,
                eos_confidence_threshold=gen_kwargs.get("eos_confidence_threshold", 0.5),
                expand_eos_confidence_threshold=gen_kwargs.get("expand_eos_confidence_threshold", 0.9),
                eos_check_tokens=gen_kwargs.get("eos_check_tokens", 32),
            )
            cont_toks_list = []
            for single_output in out_list:
                generated_tokens = single_output[prompt_length:]
                decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                cont_toks_list.append(decoded_text)
            if self.rank == 0 and self.is_first_inference:
                eval_logger.info("\n\n--- First Batch Inference (Rank 0) ---")
                for i, (question, answer) in enumerate(zip(contexts, cont_toks_list)):
                    eval_logger.info(f"Question {i+1}: {question}")
                    eval_logger.info(f"\nAnswer   {i+1}: {answer}\n")
                eval_logger.info("------------------------------------\n\n")
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
