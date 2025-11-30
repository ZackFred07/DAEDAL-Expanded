import logging
import os
from datetime import timedelta
from typing import Dict, List, Literal, Optional, Tuple, Union, TypeVar
import torch
import torch.nn.functional as F
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
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)
from dllm_eval.api.instance import Instance
from dllm_eval.api.model import LM, TemplateLM
from dllm_eval.api.registry import register_model
from dllm_eval.models.utils import get_dtype, configure_pad_token
from dllm_eval.models.modeling_dream import DreamModelLM


eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(
    logits,
    temperature=0.0,
    top_p=None,
    top_k=None,
    margin_confidence=False,
    neg_entropy=False,
):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


# Essentially replaces `generation_utils.DreamGenerationMixin` for the diffusion process
@torch.no_grad()
def generate(
    model,
    prompt,  # (batch, prompt)
    tokenizer,
    attention_mask,
    min_gen_length=64,
    max_gen_length=2048,
    temperature=0.0,
    high_conf_threshold=0.90,
    low_conf_threshold=0.10,
    expansion_factor=8,
    remasking="low_confidence",
    mask_token_id=151666,
    pad_token_id=151643,
    eos_token_id=151643,
    return_dict_in_generate=False,
    output_history=False,
    eps=1e-3,
    alg="origin",
    alg_temp=None,
    top_p=None,
    top_k=None,
    eos_confidence_threshold=0.5,
    expand_eos_confidence_threshold=0.9,
    eos_check_tokens=32,
    block_length=32,
):
    def _calculate_eos_confidence(
    # Helper function to calculate EOS confidence
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

    with torch.autocast(device_type="cuda"):
        assert eos_token_id is not None
        assert prompt is not None
        batch_size = prompt.shape[0]
        input_ids = prompt
        device = input_ids.device
        initial_gen_length = min_gen_length + (max_gen_length - min_gen_length) // 2
        floor_gen_lengths = torch.full(
            (batch_size,), min_gen_length, dtype=torch.long, device=device
        )
        ceiling_gen_lengths = torch.full(
            (batch_size,), max_gen_length, dtype=torch.long, device=device
        )
        min_eos = eos_confidence_threshold
        max_eos = eos_confidence_threshold + 0.1
        gen_lengths = torch.full((batch_size,), initial_gen_length, dtype=torch.long, device=device)
        prompt_length = input_ids_length = input_ids.shape[-1]
        x = torch.full(
                (batch_size, prompt_length + initial_gen_length),
                mask_token_id,
                dtype=torch.long,
                device=device,
            )
        x[:, :prompt_length] = prompt.clone()
        prompt_index = x != mask_token_id

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(
                attention_mask, (0, gen_lengths - attention_mask.shape[1]), value=1.0
            )
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        # TODO Stage 1
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            print("[Stage-1] Initial Length Adjustment")
        while True:

            total_lengths = prompt_length + gen_lengths
            max_len_pre = x.shape[1]
            # Build an attention mask up to total lengths
            arange_tensor_pre = torch.arange(max_len_pre, device=device).expand(batch_size, -1)
            attention_mask_pre = (arange_tensor_pre < total_lengths.unsqueeze(1)).long()
            if attention_mask_pre is not None and torch.any(attention_mask_pre == 0.0):
                print("That condition with the attention mask is actually true btw")
            logits_pre = model(x, attention_mask="full").logits
            # Compute EOS for each Sequence
            batch_eos_confidences = _calculate_eos_confidence(logits_pre, total_lengths, prompt_length, eos_check_tokens)
            # Decide which sequence need mroe space
            sequences_to_increase = min_eos >= batch_eos_confidences
            sequences_to_decrease = max_eos <= batch_eos_confidences
            sequences_to_search = sequences_to_increase | sequences_to_decrease
            if (floor_gen_lengths == ceiling_gen_lengths).any():
                print("floor & ceiling equaled")
            sequences_to_search = (
                ~(floor_gen_lengths >= ceiling_gen_lengths) & sequences_to_search
            )

            if not sequences_to_search.any():
                # No sequence needs further searching
                if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                    print(f"All sequences' EOS confidence reach the threshold {eos_confidence_threshold} or max length.")
                break
            if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                print(f"Some sequences' EOS confidence ({[round(c.item(), 4) for c in batch_eos_confidences]}) < {eos_confidence_threshold}. Expand initial length.")

            # Binear Search
            new_gen_lengths = gen_lengths.clone()
            floor_gen_lengths[sequences_to_increase] = new_gen_lengths[
                sequences_to_increase
            ] + 1
            ceiling_gen_lengths[sequences_to_decrease] = new_gen_lengths[
                sequences_to_decrease
            ] - 1

            new_gen_lengths[sequences_to_search] = floor_gen_lengths[
                sequences_to_search
            ] + torch.bitwise_right_shift(
                ceiling_gen_lengths[sequences_to_search]
                - floor_gen_lengths[sequences_to_search],
                1,
            )

            # Build the new tensor
            max_new_total_len = prompt_length + new_gen_lengths.max()
            new_x_tensor = torch.full(
                (batch_size, max_new_total_len),
                eos_token_id,
                dtype=torch.long,
                device=device,
            )
            for i in range(batch_size):
                # Copy the existing tokens into the front
                original_total_len = min(
                    prompt_length + gen_lengths[i].item(),
                    prompt_length + new_gen_lengths[i].item(),
                )
                new_x_tensor[i, :original_total_len] = x[i, :original_total_len]
                new_total_len_i = prompt_length + new_gen_lengths[i].item()
                if sequences_to_search[i] and new_total_len_i > original_total_len:
                    # Where sequence was expanded, fill new positions with mask_id
                    new_x_tensor[i, original_total_len:new_total_len_i] = mask_token_id
            # Update
            x = new_x_tensor
            gen_lengths = new_gen_lengths

        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            print(f"[Stage-2] Iterative Denoising and Mask Insertion")

        current_pos = torch.full(
            (batch_size,), prompt_length, dtype=torch.long, device=device
        )  # The starting index into the generation region for all sequences
        denoise_only_mode = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )  # Sequence is at max length, true if shouldn't expand

        # While our current position hasnt ended
        while (current_pos < prompt_length + gen_lengths).any():

            total_lengths = prompt_length + gen_lengths
            x_before_step = x.clone()

            for i in range(batch_size):
                if gen_lengths[i] >= max_gen_length and not denoise_only_mode[i]:
                    if current_pos[i] < total_lengths[i]:
                        if (
                            not (dist.is_available() and dist.is_initialized())
                            or dist.get_rank() == 0
                        ):
                            print(
                                f"Sequence {i} has reached the max length {max_gen_length}. Entering denoise-only mode."
                            )
                        denoise_only_mode[i] = True

            max_len = x.shape[1]
            arange_tensor = torch.arange(max_len, device=device).expand(batch_size, -1)
            attention_mask = (arange_tensor < total_lengths.unsqueeze(1)).long()

            # Forward pass
            if attention_mask is not None and torch.any(attention_mask == 0.0):
                print("That condition with the attention mask is actually true btw")
            logits = model(x, "full", tok_idx).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            # mask_logits = logits[mask_index]

            _, predicted_tokens = sample_tokens(
                logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True
            )
            confidences = F.softmax(logits, dim=-1)

            predicted_confidences = torch.gather(
                confidences, dim=-1, index=predicted_tokens.unsqueeze(-1)
            ).squeeze(-1)

            block_mask = torch.zeros_like(x, dtype=torch.bool, device=device)
            for i in range(batch_size):
                if current_pos[i] >= total_lengths[i]:
                    continue
                block_mask[
                    i,
                    current_pos[i] : min(
                        current_pos[i] + block_length, total_lengths[i].item()
                    ),
                ] = True

            batch_eos_confidences = _calculate_eos_confidence(
                logits, total_lengths, prompt_length, eos_check_tokens
            )

            currently_masked = (x == mask_token_id)

            high_conf_indices = (
                (predicted_confidences > high_conf_threshold)
                & block_mask
                & currently_masked
                & (predicted_tokens != mask_token_id)
            )

            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=device)

            for i in range(batch_size):
                if current_pos[i] >= total_lengths[i]:
                    continue
                start_idx, end_idx = current_pos[i], min(
                    current_pos[i] + block_length, total_lengths[i].item()
                )

                if not high_conf_indices[i, start_idx:end_idx].any():
                    # Cibsuder all valid_fallback_mask
                    valid_fallback_mask = block_mask[i] & currently_masked[i]
                    if not valid_fallback_mask.any():
                        continue
                    # Compute candidate confidences and tokens
                    candidate_indices = torch.where(valid_fallback_mask)[0]
                    candidate_confs = predicted_confidences[i, candidate_indices]
                    candidate_tokens = predicted_tokens[i, candidate_indices]

                    sorted_confs, sort_indices = torch.sort(
                        candidate_confs, descending=True
                    )
                    best_idx_to_fill = -1
                    for sorted_idx in sort_indices:
                        if candidate_tokens[sorted_idx] != mask_token_id:
                            best_idx_to_fill = candidate_indices[sorted_idx]
                            break
                    if best_idx_to_fill != -1:
                        high_conf_indices[i, best_idx_to_fill] = True
                    else:
                        stuck_logits = logits[i, candidate_indices]
                        # Masked tokens set to negative infinity
                        stuck_logits[:, mask_token_id] = -torch.inf
                        new_confidences = F.softmax(stuck_logits, dim=-1)
                        new_best_confs, new_best_tokens = torch.max(
                            new_confidences, dim=-1
                        )

                        # Pick the most confident non-mask token to be marked as high confidence (guarantee something gets denoised)
                        best_of_the_best_local_idx = torch.argmax(new_best_confs)
                        pos_to_fill = candidate_indices[best_of_the_best_local_idx]
                        token_to_fill = new_best_tokens[best_of_the_best_local_idx]
                        predicted_tokens[i, pos_to_fill] = token_to_fill
                        high_conf_indices[i, pos_to_fill] = True

            # Identify low confidence tokens to expands
            potential_expand_mask = (
                (predicted_confidences < low_conf_threshold)
                & currently_masked & block_mask
                & (~high_conf_indices)
            )
            expand_indices = torch.zeros_like(x, dtype=torch.bool, device=device)
            for i in range(batch_size):
                if (
                    batch_eos_confidences[i] >= expand_eos_confidence_threshold
                    or gen_lengths[i] >= max_gen_length
                ):
                    continue
                if denoise_only_mode[i] or current_pos[i] >= total_lengths[i]:
                    continue

                # Create expansion points
                masked_candidates = torch.where(potential_expand_mask[i])[0]
                if len(masked_candidates) > 0:

                    candidate_confs = predicted_confidences[i, masked_candidates]
                    num_to_expand = min(1, len(masked_candidates))
                    if num_to_expand > 0:
                        _, lowest_conf_local_indices = torch.topk(
                            candidate_confs, num_to_expand, largest=False
                        )
                        indices_to_expand_global = masked_candidates[
                            lowest_conf_local_indices
                        ]
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
                            new_x_tensor[i, write_ptr:end_write] = mask_token_id
                            write_ptr = end_write
                        else:
                            new_x_tensor[i, write_ptr] = x[i, j]
                            write_ptr += 1

                    new_gen_lengths[i] = write_ptr - prompt_length
                # Update
                x = new_x_tensor
                gen_lengths = new_gen_lengths

            # Look at all the blocks, find one with a mask and set current position to that location
            for i in range(batch_size):
                total_len = prompt_length + gen_lengths[i]
                while current_pos[i] < total_len:
                    start_check = current_pos[i]
                    end_check = min(start_check + block_length, total_len.item())
                    if start_check == end_check:
                        break
                    if not (x[i, start_check:end_check] == mask_token_id).any():
                        current_pos[i] = start_check + block_length
                    else:
                        break
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

        # Final output assembly
        final_outputs = []
        for i in range(batch_size):
            final_len = prompt_length + gen_lengths[i]
            final_outputs.append(x[i, :final_len])
        return final_outputs


@register_model("Dream_LDAEDAL")
class Dream_LDAEDAL(TemplateLM):
    AUTO_MODEL_CLASS = transformers.AutoModel
    _DEFAULT_MAX_LENGTH = 20480

    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        backend: Literal["default", "causal", "seq2seq"] = "causal",
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = True,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = False,
        escape_until: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        parallelize: Optional[bool] = False,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        mc_num: int = 1024,
        remasking: str = "low_confidence",
        mask_id: int = 126336,
        is_check_greedy: bool = True,
        assistant_prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mc_num = mc_num
        self.mask_id = mask_id
        self.remasking = remasking
        self.pretrained = pretrained
        self.is_check_greedy = is_check_greedy
        self.assistant_prefix = assistant_prefix
        self.add_bos_token = add_bos_token
        self.escape_until = escape_until
        if not isinstance(pretrained, str):
            eval_logger.warning(
                "`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not launch via accelerate or use `parallelize=True` if passing an existing model this way."
            )
            assert (
                not parallelize
            ), "`parallelize=True` is not compatible with passing pre-initialized model to `pretrained`"
            self._model = pretrained
            self._device = self._model.device
            self._config = self._model.config
            gpus = 0

        else:
            assert isinstance(device, str)
            assert isinstance(pretrained, str)
            assert isinstance(batch_size, (int, str))
            gpus = torch.cuda.device_count()
            accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
            accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
            if accelerator.num_processes > 1:
                self.accelerator = accelerator
            if "npu" in accelerator.device.type:
                gpus = torch.npu.device_count()
            if not (parallelize or accelerator.num_processes > 1):
                device_list = set(
                    ["cuda", "cpu"]
                    + [f"cuda:{i}" for i in range(gpus)]
                    + ["mps", "mps:0"]
                    + [f"npu:{i}" for i in range(gpus)]
                )
                if device and device in device_list:
                    self._device = torch.device(device)
                    eval_logger.info(f"Using device '{device}'")
                    if device in ("mps", "mps:0") and version.parse(
                        torch.__version__
                    ) < version.parse("2.1"):
                        raise RuntimeError(
                            f"mps requires torch >= 2.1. You have {torch.__version__}"
                        )
                else:
                    eval_logger.info("Device not specified")
                    eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                    self._device = (
                        torch.device("cuda")
                        if torch.cuda.is_available()
                        else torch.device("cpu")
                    )
            else:
                if device != "cuda":
                    eval_logger.info(
                        f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                    )
                self._device = (
                    self.accelerator.device
                    if hasattr(self, "accelerator")
                    else torch.device(device)
                )
            revision = str(revision)
            revision = revision + ("/" + subfolder if subfolder is not None else "")
            self._get_config(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
                gguf_file=gguf_file,
            )
        self._get_backend(
            config=self.config, backend=backend, trust_remote_code=trust_remote_code
        )
        self._create_tokenizer(
            pretrained,
            tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
            gguf_file=gguf_file,
            add_bos_token=add_bos_token,
        )
        if isinstance(pretrained, str):
            self._create_model(
                pretrained=pretrained,
                revision=revision,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                parallelize=parallelize,
                gpus=gpus,
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                peft=peft,
                delta=delta,
                autogptq=autogptq,
                gptqmodel=gptqmodel,
                gguf_file=gguf_file,
                **kwargs,
            )
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()
        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)
        self.add_bos_token = add_bos_token
        if "gemma" in getattr(self.config, "model_type", ""):
            self.add_bos_token = True
            eval_logger.info(
                f"Model type is '{self.config.model_type}', part of the Gemma family--a BOS token will be used as Gemma underperforms without it."
            )
        self._max_length = max_length
        self.pretrained = pretrained
        self.delta = delta
        self.peft = peft
        self.revision = revision
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)
        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            if gpus > 1:
                if hasattr(self, "accelerator") and self.accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > self.accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {self.accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{self.accelerator.device}")
                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    self._rank = 0
                    self._world_size = 1
            else:
                self._rank = 0
                self._world_size = 1
        else:
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )
        self.is_first_inference = True

    @property
    def rank(self):
        if hasattr(self, "_rank"):
            return self._rank
        if hasattr(self, "accelerator"):
            return self.accelerator.local_process_index
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def world_size(self):
        if hasattr(self, "_world_size"):
            return self._world_size
        if hasattr(self, "accelerator"):
            return self.accelerator.num_processes
        return int(os.environ.get("WORLD_SIZE", 1))

    def _get_accelerate_args(
        self,
        parallelize: Optional[bool] = None,
        device_map: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        gpus: Optional[int] = None,
    ) -> dict:
        num_local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        if parallelize is None and gpus is not None and gpus > 1:
            parallelize = True
        args = {}
        if parallelize:
            max_memory_all_gpus = get_max_memory()
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            max_memory_per_gpu_map = (
                {
                    device_idx: max_memory_per_gpu
                    for device_idx in range(len(max_memory_all_gpus))
                }
                if max_memory_per_gpu is not None
                else {k: v for k, v in max_memory_all_gpus.items()}
            )
            if hasattr(self, "accelerator"):
                max_memory_per_gpu_map = {
                    k: v
                    for k, v in max_memory_all_gpus.items()
                    if k % num_local_processes
                    == self.accelerator.process_index % num_local_processes
                }
            args["max_memory"] = max_memory_per_gpu_map
            args["device_map"] = "auto"
            args["offload_folder"] = offload_folder
            if max_cpu_memory is not None:
                args["max_memory"]["cpu"] = max_cpu_memory
            eval_logger.info(
                f"Model parallel set to True. Max memory per GPU: {args['max_memory']}, Device map: {args['device_map']}"
            )
        else:
            args["device_map"] = {"": str(self.device)}
            eval_logger.info(
                f"Model parallel set to False. Device map: {args['device_map']}"
            )
        return args

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length > 1e10:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _get_backend(
        self,
        config: Union[transformers.PretrainedConfig, transformers.AutoConfig],
        backend: Literal["default", "causal", "seq2seq"] = "default",
        trust_remote_code: Optional[bool] = False,
    ) -> None:
        assert backend in ["default", "causal", "seq2seq"]
        if backend != "default":
            self.backend = backend
            eval_logger.info(
                f"Overrode HF model backend type, and using type '{self.backend}'"
            )
        else:
            if (
                getattr(config, "model_type")
                in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
            ):
                self.backend = "seq2seq"
            elif (
                getattr(self.config, "model_type") in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            ):
                self.backend = "causal"
            else:
                eval_logger.warning(
                    "HF model type is neither CausalLM nor Seq2SeqLM. Assuming CausalLM."
                )
                self.backend = "causal"

    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
        gguf_file: Optional[str] = None,
    ) -> None:
        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat",
        trust_remote_code: Optional[bool] = False,
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        **kwargs,
    ) -> None:
        if autogptq or gptqmodel:
            raise NotImplementedError(
                "Quantization options are not implemented for this custom class."
            )
        model_dtype = get_dtype(dtype)
        eval_logger.info(f"Loading model with dtype: {model_dtype}")
        model_kwargs = kwargs if kwargs else {}
        if not parallelize:
            model_kwargs.update(
                self._get_accelerate_args(
                    parallelize=parallelize,
                    gpus=gpus,
                    max_memory_per_gpu=max_memory_per_gpu,
                    max_cpu_memory=max_cpu_memory,
                    offload_folder=offload_folder,
                )
            )
        self._model = DreamModelLM.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        if peft:
            from peft import PeftModel

            eval_logger.info(f"Loading PEFT model from {peft}")
            self._model = PeftModel.from_pretrained(
                self._model, peft, torch_dtype=model_dtype
            )
        if not parallelize:
            self._model = self._model.to(self.device)
        self._model.eval()

    def _create_tokenizer(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ],
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        gguf_file: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
    ) -> None:
        kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "use_fast": use_fast_tokenizer,
        }
        if add_bos_token:
            kwargs["add_bos_token"] = True
        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer, **kwargs
                )
            else:
                self.tokenizer = tokenizer
        else:
            model_name = (
                pretrained if isinstance(pretrained, str) else self.model.name_or_path
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, **kwargs
            )

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        special_tokens_kwargs = {}
        if add_special_tokens is None:
            if self.backend == "causal":
                special_tokens_kwargs["add_special_tokens"] = self.add_bos_token
        else:
            special_tokens_kwargs["add_special_tokens"] = add_special_tokens
        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        add_special_tokens = (
            {"add_special_tokens": self.add_bos_token}
            if self.backend == "causal"
            else {}
        )
        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len and encoding["input_ids"].size(1) > left_truncate_len:
            eval_logger.warning(
                f"Left-truncating from {encoding['input_ids'].size(1)} to {left_truncate_len} tokens."
            )
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side
        return encoding["input_ids"].to(self.device), encoding["attention_mask"].to(
            self.device
        )

    def tok_decode(self, tokens, skip_special_tokens=False):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps, attn_mask=None, labels=None):
        with torch.no_grad():
            if self.backend == "seq2seq":
                return self.model(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits
            else:
                return self.model(inps, attention_mask=attn_mask).logits

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        raise NotImplementedError

    def loglikelihood(self, requests):
        raise NotImplementedError

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
                attention_mask=attn_masks,
                initial_gen_length=gen_kwargs.get("initial_gen_length", 64),
                max_gen_length=gen_kwargs.get("max_gen_length", 2048),
                temperature=gen_kwargs.get("temperature", 0.0),
                high_conf_threshold=gen_kwargs.get("high_conf_threshold", 0.90),
                low_conf_threshold=gen_kwargs.get("low_conf_threshold", 0.10),
                expansion_factor=gen_kwargs.get("expansion_factor", 8),
                remasking=gen_kwargs.get("remasking", self.remasking),
                eos_confidence_threshold=gen_kwargs.get(
                    "eos_confidence_threshold", 0.5
                ),
                expand_eos_confidence_threshold=gen_kwargs.get(
                    "expand_eos_confidence_threshold", 0.9
                ),
                eos_check_tokens=gen_kwargs.get("eos_check_tokens", 32),
                mask_token_id=151666,
                pad_token_id=151643,
                eos_token_id=151643,
            )
            cont_toks_list = []
            for single_output in out_list:
                generated_tokens = single_output[prompt_length:]
                decoded_text = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=False
                )
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
