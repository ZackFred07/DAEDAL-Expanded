# DiffuLLaMA

## High-level idea

This is **DiffuLLaMA’s discrete diffusion sampling loop** in “random keep” mode:

* You start with some input tokens `x` and a mask saying which positions are “source” (fixed) vs “maskable” (to be generated).
* At the beginning, **all maskable positions are `[MASK]`**.
* At each diffusion step `t = T, T-1, ..., 1`, the model:

  * Predicts a full denoised sequence `x0` (one token per position).
  * **Randomly chooses a fraction of remaining `[MASK]` positions** and permanently fills them with `x0` tokens.
  * Keeps the rest as `[MASK]` to be refined in later steps.
* By the end, all maskable tokens have been filled in.


is describing that per-step **fraction of mask positions that get “committed”** to `x0`.

---

## Inputs and setup

```python
def generate_samples(model, diff_args, tokenizer, inputs, verbose=False):
    model.eval()
    print("*** Start sampling, random keep...")
```

* `model`: the DiffuLLaMA model.
* `diff_args`: holds sampling hyperparams: `logits_temp`, `topp_temp`, `diffusion_steps`, `shift` (bool).
* `tokenizer`: must provide `mask_token_id`.
* `inputs`: dict with:

  * `"input_ids"`: the token ids (prompt + maybe initial text).
  * optional `"src_mask"`: True where tokens are **source / fixed**, False where the model is allowed to change them.

```python
    logits_temp = diff_args.logits_temp
    topp_temp = diff_args.topp_temp

    x = inputs["input_ids"].to(model.device)
    if "src_mask" not in inputs:
        src_mask = torch.zeros_like(x, dtype=torch.bool).to(model.device)
    else:
        src_mask = inputs["src_mask"].bool().to(model.device)
```

* `x`: original input tokens.
* `src_mask`: if not given, everything is maskable (`False` everywhere → no fixed positions).
* If given, `src_mask == True` marks **fixed (source) tokens**.

```python
    x_embed = model.get_embeds(x)
    seq_len = x.size(1)
    batch_size = x.size(0)
    attention_mask = get_anneal_attn_mask(
        seq_len, batch_size,
        dtype=x_embed.dtype,
        device=x.device,
        attn_mask_ratio=1.0
    )  # all 0
```

* `x_embed` is only used to infer dtype/device for the attention mask.
* `attention_mask` is some custom mask (here effectively “all zeros” → no attention restriction).

```python
    init_maskable_mask = maskable_mask = ~src_mask
```

* `maskable_mask`: `True` at positions the model is **allowed to overwrite**.
* Source tokens (`src_mask == True`) are never changed.

---

## Initial step: all maskable positions are `[MASK]`

```python
    # first forward, all position except src is [M]
    xt = x.masked_fill(maskable_mask, tokenizer.mask_token_id)
```

* `xt` is the **current noisy sequence** (`x_t` in diffusion notation).
* All maskable positions are replaced with `[MASK]`.
* Source positions keep their original tokens.

```python
    if verbose:
        print(f"t=T(in):", tokenizer.decode(xt.tolist()[0]))
```

Debug print of the input at time `t = T`.

---

## First prediction (`t = T`)

```python
    logits = model(xt, attention_mask=attention_mask)
    filter_logits = top_p_logits(logits/logits_temp, p=topp_temp)
    scores = torch.log_softmax(filter_logits, dim=-1)
```

* Run the model on fully masked input (except src).
* Apply **temperature** (`logits / logits_temp`) and **top-p / nucleus sampling** via `top_p_logits`.
* Convert to log probabilities `scores`.

```python
    x0 = dists.Categorical(logits=scores).sample()
    x0_scores = torch.gather(scores, -1, x0.unsqueeze(-1)).squeeze(-1)
```

* Sample a **candidate denoised token** at every position → `x0`.
* `x0_scores` is just the log prob of the sampled token at each position (not really used later, may be for debugging/analysis).

```python
    if diff_args.shift:
        #### deal with shift, left most token will be replaced anyway
        x0 = torch.cat([x[:,0:1], x0[:, :-1]], dim=1)
        x0_scores = torch.cat([x0_scores[:,0:1], x0_scores[:, :-1]], dim=1)
```

* If `shift` is enabled, they **align predictions with the original input** by shifting `x0` right by one:

  * First position copied from original input `x[:, 0:1]`.
  * Rest of tokens shifted.

```python
    #### replace output of non-[MASK] positions with xt
    x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
```

* `x0` contained predictions for *every* position.
* Here they **overwrite only the maskable positions** with those predictions.
* Non-maskable (source) positions use the tokens from `xt` (which are original `x` there).

```python
    if verbose:
        print(f"t=T(out):", tokenizer.decode(x0.tolist()[0]))
```

So after this, `x0` is the **model’s full predicted clean sequence** given a fully masked starting point.

---

## Main diffusion loop: t = T-1, ..., 1

```python
    for t in range(diff_args.diffusion_steps-1, 0, -1): # t from T-1 to 1
        with torch.no_grad():
            #### select rate% tokens to be still [MASK]
            p_to_x0 = 1/(t+1)
```

* Backward loop over diffusion steps.
* At step `t`, **each still-maskable position** will be turned into a “committed” token with probability `p_to_x0 = 1/(t+1)`.

```python
            masked_to_x0 = maskable_mask & (torch.rand_like(x0, dtype=torch.float) < p_to_x0)
```

* `masked_to_x0` is a boolean mask:

  * Only True where position is still maskable.
  * Bernoulli sample with probability `p_to_x0` for those positions.

**Interpretation:** at each step, a random subset of remaining `[MASK]` tokens is now **locked in** to the current `x0` prediction.

```python
            xt.masked_scatter_(masked_to_x0, x0[masked_to_x0])
            maskable_mask = maskable_mask.masked_fill(masked_to_x0, False)
```

* For those chosen positions:

  * The current noised sequence `xt` (which previously had `[MASK]` there) is updated to contain the predicted tokens.
  * Those positions are now **removed from `maskable_mask`** (no longer allowed to change in future steps).

```python
            if verbose:
                print(f"t={t}(in):", tokenizer.decode(xt.tolist()[0]))
```

Debug: the input `xt` to the model at this step.

---

### Re-predict `x0` from partially denoised `xt`

```python
            logits = model(xt, attention_mask=attention_mask)
            filter_logits = top_p_logits(logits/logits_temp, p=topp_temp)
            scores = torch.log_softmax(filter_logits, dim=-1)
            x0 = dists.Categorical(logits=scores).sample()
            x0_scores = torch.gather(scores, -1, x0.unsqueeze(-1)).squeeze(-1)
```

* The model sees a sequence where some positions are already tokens, some are still `[MASK]`.
* It predicts a **new full candidate denoised sequence** `x0`.

```python
            if diff_args.shift:
                #### deal with shift, left most token will be replaced anyway
                x0 = torch.cat([x[:,0:1], x0[:, :-1]], dim=1)
                x0_scores = torch.cat([x0_scores[:,0:1], x0_scores[:, :-1]], dim=1)
```

* Same shift trick as before.

```python
            # replace output of non-[MASK] positions with xt
            x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
            if verbose:
                print(f"t={t}(out):", tokenizer.decode(x0.tolist()[0]))
```

* `x0` is updated such that:

  * For positions already committed (no longer maskable), it effectively uses `xt` (i.e., the committed tokens).
  * For still maskable positions, it uses the new predicted tokens.
* So `x0` is always a **full candidate clean sequence consistent with all commitments so far**.

This repeats until `t = 1`. Over time, `maskable_mask` shrinks to all False (all positions committed).

---

## Final shift and return

```python
    if diff_args.shift:
        x0 = x0[:,1:]

    return x0
```

* If `shift` was used, they drop the first token so the output aligns with expectation (e.g., removing some dummy or BOS-like token).
* `x0` is the final generated token IDs.

---

## Intuition in one sentence

This function is a **mask-based diffusion sampler**:
start with `[MASK]` in all generatable positions, and at each step randomly “lock in” some subset of tokens from the model’s current prediction, while the model keeps refining the remaining `[MASK]` tokens until everything is filled.

If you want, next I can sketch this as a tiny toy example with a 1-D sequence (like `A [MASK] [MASK] B`) and walk through the steps `T → T-1 → ... → 1` with concrete numbers.

# LLaDA

## Big-picture behavior

* Input: `prompt` tokens, shape `(batch, prompt_len)`.
* It appends `gen_length` masked slots after the prompt and then **generates tokens into those slots**.
* Generation is done in **blocks** of size `block_length` (e.g., 32 tokens at a time).
* For each block:

  * Run `steps_per_block` refinement steps.
  * At each step:

    * Get logits for all positions.
    * Decide a *confidence* for each masked position in the current block.
    * **Commit a certain number of the highest-confidence masked tokens** (turn them from `[MASK]` into actual tokens).
* Optional: **classifier-free guidance** (`cfg_scale`) and randomness (`temperature` + Gumbel noise).
* `remasking` controls how “confidence” is defined (true low-confidence schedule vs random).

---

## Setup and initial sequence

```python
def generate(
    model,
    prompt,  # (batch, prompt)
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    with torch.autocast(device_type="cuda"):
```

* Mixed precision on CUDA for speed/memory.

```python
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=prompt.device
        )
        x[:, : prompt.shape[1]] = prompt.clone()
```

* `x` is the full working sequence: `[prompt | gen_area]`.
* Shape: `(batch_size, prompt_len + gen_length)`.
* Initially:

  * Prompt positions contain the real tokens.
  * All positions after the prompt are `[MASK]` (`mask_id`).

```python
        prompt_index = x != mask_id
```

* `prompt_index` is `True` where the tokens are **initially non-mask** (i.e., the prompt).
* This does *not* change later; it’s used to construct the unconditional input for CFG.

---

## Block scheduling

```python
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
```

* The generation region is split into `num_blocks` contiguous segments of length `block_length`.
* Total `steps` are divided across blocks → `steps_per_block`.

---

## Loop over blocks

```python
        for num_block in tqdm(
            range(num_blocks),
            disable=(dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)
        ):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length
```

* For block `num_block`, indices `[start_idx:end_idx]` in `x` are the current block’s positions.

```python
            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
```

* `block_mask_index`: which positions in this block are still `[MASK]` right now.
* `get_num_transfer_tokens` returns a tensor of shape `(batch_size, steps_per_block)`:

  * `num_transfer_tokens[j, i]` = **how many tokens to commit** in block `num_block` for sample `j` at step `i`.
  * Usually designed so that over all steps the block’s masks eventually all get filled.

---

## Inner diffusion loop: per-step refinement

```python
            for i in range(steps_per_block):
                mask_index = x == mask_id
```

* `mask_index`: global mask over entire sequence (prompt + all blocks).
* True where `x` is still `[MASK]`.

### Classifier-free guidance (optional)

```python
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits
```

* If `cfg_scale > 0`:

  * `x`: **conditional** input (with prompt).
  * `un_x`: **unconditional** input where prompt positions are masked out (`mask_id`).
  * Concatenate batch: `[x; un_x]`, forward once for efficiency.
  * Split logits into `logits` (cond) and `un_logits` (uncond).
  * Combine via a guidance formula:

    * Essentially boosts directions where conditional differs from unconditional.
* If `cfg_scale == 0`, just use `model(x)`.

---

### Sampling raw token guesses (`x0`)

```python
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
```

* `add_gumbel_noise` adds Gumbel noise scaled by `temperature`:

  * `temperature = 0` → greedy (no noise).
  * `temperature > 0` → sampling via Gumbel-Max.
* `x0` is the current **candidate token** at each position.

---

### Confidence scores / remasking strategy

```python
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)
```

* `x0_p` is a “confidence” scalar per position:

  * `"low_confidence"`: actual probability of the selected token under the softmax.

    * High-prob tokens → high confidence.
  * `"random"`: uniform random in `[0,1]` (so tokens are chosen to be committed in a random order).

```python
                x0_p[:, end_idx:] = -np.inf
```

* **Important**: For this step / block, any position **after** the current block (`end_idx:`) has confidence `-∞`.
* This ensures we will never commit tokens outside the current block in this block’s loop.

---

### Respect existing tokens vs masks

```python
                x0 = torch.where(mask_index, x0, x)
```

* For positions that are still `[MASK]`, keep the predicted `x0`.
* For positions already filled (not `[MASK]`), **override x0 with the existing token** in `x`.

  * So previously committed tokens are never changed.

```python
                confidence = torch.where(
                    mask_index,
                    x0_p,
                    torch.tensor(-np.inf, device=x0.device)
                )
```

* `confidence` is valid only at mask positions; all non-masked tokens get `-∞` confidence.
* Combined with the earlier `x0_p[:, end_idx:] = -np.inf`, this means:

  * Only **currently masked positions within the current block** can be chosen this step.

---

### Commit the top-k tokens for this step

```python
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_indices] = x0[j, select_indices]
```

Per batch element `j`:

* Look up how many tokens to commit this step: `num_tokens`.
* Select the `num_tokens` positions with **highest confidence** for this sample:

  * Because of the masked `-∞`, these positions are:

    * in the current block,
    * currently `[MASK]`.
* Then **write those tokens into `x`**:

  * `x[j, select_indices] = x0[j, select_indices]`
  * These positions are no longer `[MASK]` in `x`, so on the next step:

    * `mask_index` will be False there.
    * Their `confidence` will be `-∞`.
    * They won’t be changed again.

After finishing `steps_per_block` steps, that block should have all its masks filled (by construction of `get_num_transfer_tokens`).

---

## Return

```python
        return x
```

* `x` now contains:

  * The original prompt in the first `prompt_len` positions.
  * The generated tokens in the next `gen_length` positions (no `[MASK]` left if all blocks fully filled).

---

## Intuitive summary

* Think of `x` as a long sequence with a prompt and a masked tail.
* LLaDA fills the tail **block by block**:

  1. For the current block, run the model multiple times.
  2. At each step, pick a subset of still-mask tokens in that block:

     * Highest-confidence predictions if `remasking="low_confidence"`.
     * Random positions if `remasking="random"`.
  3. Turn them from `[MASK]` into actual tokens, and never touch them again.
* Classifier-free guidance (`cfg_scale`) biases predictions towards versions that strongly depend on the prompt.
* `temperature` controls how much randomness is injected via Gumbel noise.

If you want, I can sketch a tiny toy run with, say, `prompt_len=3`, `gen_length=4`, `block_length=2` and walk step-by-step through how tokens get committed.

