import os

import torch
from datasets import load_dataset

# from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    #    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# temporary
os.environ["PYTHONUTF8"] = "1"

dataset = load_dataset(
    "json",
    data_files="../data/instruct/zig_instruct_data.jsonl",
    split="train",
)

model_name = "Qwen/CodeQwen1.5-7B-Chat"

# Jst comment this since we has good gpu
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,  # Fix: was string "float16"
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# drop this for quality & not lora
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=16,       # Scaled down to 2*r, more standard
#     lora_dropout=0.1,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
#     task_type="CAUSAL_LM",
# )

# model = get_peft_model(model, lora_config)
# Sanity check — print trainable parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(
    f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)"
)


def format_prompt(examples):
    prompts = []
    for instruction, code in zip(examples["instruction"], examples["code"]):
        prompt = f"""### Instruction:
{instruction}

### Response:
```zig
{code}
```"""
        prompts.append(prompt)
    return {"text": prompts}


dataset = dataset.map(format_prompt, batched=True)

OUTPUT_DIR = "../model/"

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",  # Fix: tell SFT which column to use
    max_seq_length=2048,  # Fix: set explicitly, don't rely on default
    num_train_epochs=3,
    per_device_train_batch_size=16,  # We should push this up (2 to 16)
    gradient_accumulation_steps=1,  # not really needed to be high (4 to 1)
    gradient_checkpointing=True,  # saves memory during backprop
    learning_rate=1e-4,  # push higher to 1e-4
    # fp16=True,
    bf16=True,
    optim="adamw_torch_fused",
    dataloader_pin_memory=False,
    save_steps=500,
    save_total_limit=2,
    logging_steps=10,
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,  # Fix: 'tokenizer' kwarg is deprecated in TRL
    train_dataset=dataset,
    args=sft_config,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Training complete! Saved to {OUTPUT_DIR}")
