# End-to-End SFT Pipeline — PEFT + QLoRA on Hugging Face

A single, runnable Google Colab notebook that demonstrates the complete supervised fine-tuning
lifecycle for an open LLM: from raw dataset through 4-bit quantization and LoRA adapters to a
before/after comparison of model behaviour.

**Deliverable:** [`SFT_QLoRA_Pipeline.ipynb`](SFT_QLoRA_Pipeline.ipynb)

---

## Quick start

1. Open the notebook in Google Colab.
2. **Runtime → Change runtime type → GPU.** An **L4** or **A100** is recommended (bfloat16 support).
   A T4 works but is slower and falls back to fp16.
3. **Runtime → Run all.**

No Hugging Face token, no license acceptance, no manual configuration. Expected wall-clock on an L4:
roughly **8–15 minutes**, most of it in the training cell.

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
| Method | QLoRA (4-bit NF4 base + LoRA r=16) | The assignment's target technique; ~1% of parameters trainable. |
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
| 13 | Sanity checks (12 assertions covering GPU use, data, adapters, artifacts, reload) |
| 14 | Conceptual architecture diagram |
| 15 | Final summary and run report |

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
