"""QLoRA fine-tune — training example.

Runs on a single GPU (a 16 GB T4 is enough for the teaching model below).
Larger models need a fresh memory budget for weights, activations and optimizer state.

    pip install -r requirements.txt
    python train_qlora.py

"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = Path(__file__).resolve().parent / "adapter"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Single source of truth for every dtype below: bf16 needs compute capability >= 8.0
# (Ampere+). A T4 is sm_75 and must train in fp16.
use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16   # 4-bit compute + autocast
model_dtype = compute_dtype



def build_dataset(n=800):
    """Dolly rows -> TRL's conversational prompt/completion format.

    SFTTrainer serialises each example with the tokenizer's own chat template (the same
    template used at inference) and, with completion_only_loss=True, computes the loss on
    the assistant turn only (prompt tokens are labelled -100).
    """
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    if not 1 <= n <= len(ds):
        raise ValueError(f"num-examples must be between 1 and {len(ds)}")
    ds = ds.select(range(n))

    def to_prompt_completion(ex):
        ctx = ex.get("context", "")
        prompt = ex["instruction"] if not ctx else f"{ex['instruction']}\n\nContext:\n{ctx}"
        return {
            "prompt":     [{"role": "user",      "content": prompt}],
            "completion": [{"role": "assistant", "content": ex["response"]}],
        }

    return ds.map(to_prompt_completion, remove_columns=ds.column_names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-examples", type=int, default=800)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Use 1 for a short GPU check; -1 trains one epoch")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    options = parser.parse_args()
    if options.max_steps == 0 or options.max_steps < -1:
        parser.error("max-steps must be -1 or a positive integer")
    if device != "cuda":
        raise RuntimeError("This QLoRA lab requires an NVIDIA CUDA GPU.")
    torch.manual_seed(0)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=bnb,
        dtype=model_dtype,  # T4 uses fp16; supported newer GPUs use bf16
        device_map={"": torch.cuda.current_device()},  # single-GPU training; no CPU offload
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # LoRA on the attention + MLP projections; lora_alpha / r = 2 scales the update B @ A.
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # 1 epoch, LR 2e-5, cosine scheduler + 10 warmup steps.
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        peft_config=lora,
        train_dataset=build_dataset(options.num_examples),
        args=SFTConfig(
            output_dir=str(options.output_dir),
            num_train_epochs=1,
            max_steps=options.max_steps,
            seed=0,
            optim="adamw_torch",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            bf16=use_bf16,
            fp16=(device == "cuda" and not use_bf16),
            logging_steps=10,
            completion_only_loss=True,   # loss on the assistant turn only
            max_length=1024,
            report_to="none",
        ),
    )

    # TRL 1.12 casts quantized trainable parameters to bf16. Use fp32 adapter
    # master weights on T4 so fp16 AMP can safely unscale gradients.
    if not use_bf16:
        for p in trainer.model.parameters():
            if p.requires_grad and p.dtype == torch.bfloat16:
                p.data = p.data.to(torch.float32)
    trainable_dtypes = sorted({str(p.dtype) for p in trainer.model.parameters() if p.requires_grad})
    print("trainable parameter dtypes:", trainable_dtypes)
    assert use_bf16 or "torch.bfloat16" not in trainable_dtypes, (
        "bfloat16 trainable parameters on a GPU whose GradScaler cannot unscale them"
    )

    trainer.train()

    # print_trainable_parameters() lives on the PEFT model, which SFTTrainer wraps.
    trainer.model.print_trainable_parameters()
    trainer.model.save_pretrained(options.output_dir)
    tok.save_pretrained(options.output_dir)
    print("Saved LoRA adapter to", options.output_dir, sorted(p.name for p in options.output_dir.iterdir()))


if __name__ == "__main__":
    main()
