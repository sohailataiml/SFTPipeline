# End-to-End SFT Pipeline — PEFT + QLoRA on Hugging Face

A single, runnable Google Colab notebook that demonstrates the complete supervised fine-tuning
lifecycle for an open LLM: from raw dataset through 4-bit quantization and LoRA adapters to a
before/after comparison of model behaviour.

**Deliverable:** [`SFT_QLoRA_Pipeline.ipynb`](SFT_QLoRA_Pipeline.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sohailataiml/SFTPipeline/blob/main/SFT_QLoRA_Pipeline.ipynb)

---

## Quick start

1. Click the **Open in Colab** badge above (or use the link in *Running it in Colab* below).
2. **Runtime → Change runtime type → GPU.** An **L4** or **A100** is recommended (bfloat16 support).
   A T4 works but is slower and falls back to fp16.
3. **Runtime → Run all.**

No Hugging Face token, no license acceptance, no manual configuration. Expected wall-clock on an L4:
roughly **8–15 minutes**, most of it in the training cell.

---

## Running it in Colab

### Open it

```
https://colab.research.google.com/github/sohailataiml/SFTPipeline/blob/main/SFT_QLoRA_Pipeline.ipynb
```

That URL loads this repo's notebook directly — no download, no upload, no sign-in to GitHub. It opens
a scratch copy; **Colab will not write back to this repo** unless you explicitly choose
*File → Save a copy in GitHub*.

Alternatives, if you prefer: in Colab use *File → Open notebook → GitHub* and paste
`sohailataiml/SFTPipeline`; or download `SFT_QLoRA_Pipeline.ipynb` and use *File → Upload notebook*.

### Attach a GPU (required)

*Runtime → Change runtime type → Hardware accelerator → **GPU***, then pick the type:

| GPU | Works? | Notes |
| --- | --- | --- |
| **L4** | Best default | bfloat16; the Colab Pro standard allocation |
| **A100** | Best performance | bfloat16; use if you have compute units to spend |
| **T4** | Works, slower | No bfloat16 — the notebook auto-detects this and falls back to fp16, and prints a warning |
| CPU / TPU | **No** | Section 1.2 stops with a clear message. QLoRA needs bitsandbytes CUDA kernels. |

You do not need to change any code to switch GPU — precision is detected at runtime.

### Run it

*Runtime → Run all.* Timings on a bfloat16 GPU:

| Stage | Sections | Time |
| --- | --- | --- |
| pip install | 1.1 | 1–3 min |
| Pre-flight, dataset, formatting | 2.1–4.2 | under 1 min |
| Base model download + load in 4-bit | 5 | ~1 min |
| Baseline generations | 6 | under 1 min |
| **SFT training (250 steps)** | 8.2 | **~2:54 (measured)** |
| Save, reload, post-training generations | 10–12 | 1–2 min |

Measured on a real run: `train_runtime` 175.9 s, 22.7 samples/s, **peak GPU memory 4.24 GB**. The
memory headroom is why `gradient_checkpointing=False` is safe here — a 0.5B model in 4-bit with r=16
adapters does not come close to filling an L4's 24 GB.

### What success looks like

- §1.2 prints your GPU name and `Mixed precision : bfloat16`
- §8.1 prints **8,798,208 trainable (1.75%)** — the LoRA adapters against a frozen 4-bit base
- §8.2 shows a live loss table that trends downward (measured: train 0.111 → 0.039,
  eval 0.114 → 0.063, mean token accuracy 0.967 → 0.982)
- §12 renders the base-vs-fine-tuned comparison table
- §13 ends with `All 13 checks passed.`
- §15 prints the run report

If §13 raises, it names exactly which check failed.

### If something goes wrong

| Symptom | Fix |
| --- | --- |
| Colab shows a **"Restart runtime"** button after the install cell | Click it, then *Runtime → Run all* again. The install cell is idempotent and the second pass is cached. |
| `ImportError` / `ModuleNotFoundError` in section 1.2 | Same fix: *Runtime → Restart session*, then *Run all*. This happens when a package was already imported before being upgraded. |
| `RuntimeError: NO CUDA GPU DETECTED` | No GPU attached — see *Attach a GPU* above. |
| Pre-flight raises on the model or dataset | The Hub was unreachable or a name is wrong. Re-run the cell; it is a pure network check. |
| `CUDA out of memory` | Unlikely at 0.5B/4-bit, but: lower `PER_DEVICE_BS` to 4 and raise `GRAD_ACCUM_STEPS` to 4 in section 2, then *Run all*. |
| Runtime disconnects mid-training | Colab reclaims idle sessions. Re-run from the top; nothing is corrupted. To keep the adapter across disconnects, use section 10.1. |

### Keeping the trained adapter

`./sft_adapter` lives only inside the Colab session and disappears when the runtime is recycled.
Section **10.1** is an opt-in cell for keeping it — set `DOWNLOAD_LOCALLY = True` to pull a zip to your
machine, or `PUSH_TO_HUB = True` (plus `HUB_MODEL_ID`) to push the few-MB adapter to the Hub. Both are
off by default so *Run all* works without a token.

---

## Pipeline

```text
Training Dataset
        ↓
Dataset Cleaning / Formatting
        ↓
Chat / Instruction Template
        ↓
Tokenizer
        ↓
Pretrained Base LLM
        ↓
4-bit Quantization
        ↓
PEFT / LoRA Adapters
        ↓
Supervised Fine-Tuning
        ↓
Save Adapter / Checkpoint
        ↓
Reload Base Model + Adapter
        ↓
Inference
        ↓
Compare Base vs Fine-Tuned Model
```

---

## What was chosen, and why

| Choice | Value | Rationale |
| --- | --- | --- |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` | Ungated (no token needed), small enough to train in minutes on any Colab GPU, ships an official ChatML chat template, and uses standard `q_proj`/`v_proj`-style module names that PEFT targets cleanly. |
| Dataset | `b-mc2/sql-create-context` | Natural-language question + `CREATE TABLE` schema → SQL. Ungated, CC-BY-4.0, short sequences, and a task whose correct *output format* is unambiguous — which makes the before/after difference visible rather than a matter of taste. |
| Method | QLoRA (4-bit NF4 base + LoRA r=16) | The assignment's target technique; 8,798,208 of 502,830,976 parameters trainable (1.75%), measured on a real run. |
| Trainer | TRL `SFTTrainer` | Current supported SFT API; handles chat templating and completion-only loss masking. |

The reference material for this assignment used Llama 2. That was not carried over: Llama 2 is gated
(requiring a token and license acceptance, which breaks *Run All* on a fresh account) and is large
enough to make a Colab demonstration fragile. The pipeline itself is model-agnostic — swapping
`MODEL_NAME` to `Qwen/Qwen2.5-1.5B-Instruct` is a one-line change.

---

## Pinned versions

```
transformers==5.15.0
trl==1.10.0
peft==0.20.0
datasets==5.0.1
accelerate==1.14.0
bitsandbytes==0.50.1
```

PyTorch is **deliberately not pinned** — the notebook uses whatever CUDA build Colab already ships.
Pinning it would force a multi-GB re-download and risk a CUDA/driver mismatch, and none of the
packages above require a torch newer than `>=2.4`.

These are current APIs, and they differ from older Llama-2 QLoRA tutorials in several places that
would otherwise raise `TypeError` at runtime:

| Old (deprecated / removed) | Current |
| --- | --- |
| `from_pretrained(..., torch_dtype=...)` | `from_pretrained(..., dtype=...)` |
| `SFTConfig(max_seq_length=...)` | `SFTConfig(max_length=...)` |
| `TrainingArguments(warmup_ratio=0.1)` | `warmup_steps=0.1` (a float < 1 means "ratio") |
| `SFTTrainer(tokenizer=...)` | `SFTTrainer(processing_class=...)` |
| `evaluation_strategy=` | `eval_strategy=` |

---

## Notebook map

| Section | Contents |
| --- | --- |
| 1 | Install pinned packages; environment report; hard failure if no CUDA GPU |
| 2 | All configuration in one cell (`MODEL_NAME`, LoRA + training hyperparameters) |
| 3 | Load dataset, show raw examples, clean, split train/eval |
| 4 | Convert to conversational prompt/completion format; render one fully formatted example |
| 5 | Load base model with 4-bit NF4 quantization |
| 6 | **Baseline inference** on held-out prompts; responses stored |
| 7 | `LoraConfig` with explanation of each parameter |
| 8 | `SFTConfig` + `SFTTrainer`; parameter budget (total / trainable / %); training |
| 9 | Training metrics table and loss curve |
| 10 | Save the LoRA adapter to `./sft_adapter`; list generated files and sizes |
| 11 | Tear down the trainer, free GPU, reload base + adapter **from disk** |
| 12 | **Post-training inference** on the same prompts; side-by-side comparison table |
| 13 | Sanity checks (13 assertions covering GPU use, data, adapters, artifacts, reload) |
| 14 | Conceptual architecture diagram |
| 15 | Final summary and run report |

---

## Reading the training loss honestly

The loss starts around 0.11 rather than the 1–3 typical of an SFT run, and mean token accuracy is
already 96.7% at step 25. That is not the model being unusually good — it is `completion_only_loss`
doing its job. Loss is computed on roughly 16 highly-patterned SQL tokens with the schema already in
context, and teacher-forced next-token prediction on `SELECT COUNT(*) FROM head WHERE age > 56` is an
easy problem. **A low loss here is not evidence that the model emits valid SQL when generating
freely** — section 12 is the only part of this notebook that tests that.

Eval loss also plateaus around step 150 (0.0641 → 0.0630 over the final 100 steps) while training
loss keeps halving, which is the onset of overfitting. Two epochs is kept because it makes the loss
curve in section 9 more legible; `NUM_TRAIN_EPOCHS = 1` reaches a comparable eval loss in half the
time if you would rather have the compute back.

---

## Notes on the results

The notebook ships with **cleared outputs**. Every loss value, parameter count and model response is
produced live when you run it — nothing in the file is a pasted or illustrative result.

What this run demonstrates is a change in **behaviour**, not in knowledge. The base model already
"knows" SQL; what it has not learned is that in this task the expected response is a bare query rather
than a chat answer with a preamble and an explanation. With the shipped settings (2,000 examples x 2 epochs at an effective batch of 16, i.e. 250 steps),
the adapter shifts the output distribution toward the shape of the training answers. Query
correctness on hard joins is a separate axis that would need substantially more data and compute —
the format shift is what a run this small demonstrates honestly.

---

## Reusing the trained adapter

`./sft_adapter` is the entire output of the training run — a few megabytes, because the frozen 4-bit
base never changed and can be re-fetched from the Hub by name.

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="auto")
model = PeftModel.from_pretrained(base, "./sft_adapter")
```

---

## Provenance: which Hugging Face skills this was built and audited against

This notebook was audited against the **official Hugging Face skills repository**
([huggingface/skills](https://github.com/huggingface/skills)) at commit
`020194918dc4a27d5a5d9a154b6b56cc2bd21364` — the exact commit the official Claude Code plugin
marketplace pins for the `huggingface-skills` plugin.

These skills are **installed** into `~/.claude/skills/` and invocable. They were installed via the
[skills.sh](https://www.skills.sh) registry, pinned explicitly to the official `huggingface/skills`
repository:

```bash
npx skills add huggingface/skills --global --agent claude-code --skill <name> --yes
```

The installed copies were then diffed against the audited clone at commit `0201949`: **all six are
byte-identical**, so the audit below was not invalidated by the install.

> Pin the source repo explicitly. A search for "huggingface" on skills.sh returns 100 results across
> 32 different owners, and several third parties publish skills under names identical to the official
> ones — `trl-training`, `hf-cli` and `huggingface-accelerate` each appear from more than one source.
> Skills are instructions an agent executes, so `huggingface/skills` should be named, not inferred.

Skills installed and used:

| Skill | Used for |
| --- | --- |
| `huggingface-llm-trainer` | SFT/TRL training patterns, reliability principles, dataset validation, eval-dataset requirement, hardware and precision guidance, ephemeral-environment warning |
| `huggingface-llm-trainer/scripts/dataset_inspector.py` | Executed against `b-mc2/sql-create-context`; reported `[SFT] NEEDS MAPPING`, which section 4 performs |
| `huggingface-llm-trainer/scripts/train_sft_example.py` | Reference LoRA config (`r=16`, `alpha=32`, `dropout=0.05`, `bias="none"`, `task_type="CAUSAL_LM"`) |
| `trl-training` | `SFTTrainer`/`SFTConfig` usage, LoRA learning rate (2e-4), packing guidance, general TRL best practices |
| `huggingface-datasets` | Datasets Server `/is-valid` endpoint used by the pre-flight cell |
| `huggingface-trackio` | Monitoring option evaluated for section 8 (see deviations below) |
| `hf-cli` | Hub operations |

The skills.sh install surfaced a third-party **Snyk "Med Risk"** rating on five of the six
(`huggingface-trackio` rated Low). Socket reported 0 alerts and the generative scan reported Safe for
all six. The rating is not explained by the registry; the contents are the official Hugging Face
repository at a pinned commit, and were read directly before use.

### Deliberate deviations from the skill guidance

Both are documented inline in the notebook rather than applied silently.

| Skill guidance | What this notebook does | Why |
| --- | --- | --- |
| "Always include Trackio" for monitoring | `report_to="none"`; metrics read from `trainer.state.log_history` | Trackio needs an HF token and a Trackio Space. The skill's target is Hugging Face Jobs, where training runs detached and logs are the only window in; here the trainer prints metrics straight into the cell. |
| "Always enable Hub push — environment is ephemeral" | Optional, opt-in cell (section 10.1), off by default | The warning applies to Colab too, but making it mandatory would require a token and break tokenless *Run all*. The cell documents both Hub push and local download. |
| `trl-training`: `--packing` fixes slow training on short sequences | `packing=False` | Our sequences *are* short (median 91 tokens), so the flag applies. But in TRL 1.x the default `bfd` packing strategy forces padding-free mode, which is only reliable with a FlashAttention backend. `huggingface-llm-trainer`'s Reliability Principle 2 ("Prioritize Reliability Over Performance") breaks the tie — there is no speed problem to solve in a run this short. Documented inline in section 8. |

The skills contain **no `BitsAndBytesConfig` guidance for the TRL path** — their only 4-bit
references are via Unsloth's `FastLanguageModel`. The QLoRA quantization setup here therefore follows
the Transformers/PEFT documentation and the QLoRA paper (NF4, double quantization, bf16 compute), and
is outside what the skills cover.
