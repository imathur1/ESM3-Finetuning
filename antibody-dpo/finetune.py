import os
import torch
import wandb

from config import create_config
from datasets import load_dataset
from huggingface_hub import login
from esm.models.esm3 import ESM3
from trl import CPOConfig
from utils import ESMCPOTrainer, ESMDataCollator
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from datetime import datetime
import torch.nn as nn

# DDP is not working for some reason (cuda internal error)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


os.environ["WANDB_PROJECT"] = "antibody-dpo"
# Set wandb to resume the previous run
os.environ["WANDB_RESUME"] = "allow"
os.environ["WANDB_RUN_ID"] = "t6sepoi0"  # Run ID corresponding to the checkpoint
timestamp = "20250419-224509"
checkpoint = 50064

# login()
model = ESM3.from_pretrained("esm3-open").to("cuda")
def get_model_size_in_gb(model):
    model_size_bytes = sum(param.nelement() * param.element_size() for param in model.parameters())
    return model_size_bytes / (1024 ** 3)


# Check if multiple GPUs are available
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     # Wrap the model with DataParallel
#     model = nn.DataParallel(model)

config = create_config()

dataset = load_dataset("csv", data_files={"train": config.train_data, "eval": config.eval_data})
# split_datasets = dataset["data"].train_test_split(test_size=0.1)
# train_dataset = split_datasets["train"]
# test_dataset = split_datasets["test"]

# Freeze all params except sequence track
for name, param in model.named_parameters():
    if name in [
        "encoder.sequence_embed.weight", 
        "output_heads.sequence_head.0.weight", 
        "output_heads.sequence_head.0.bias",
        "output_heads.sequence_head.2.weight",
        "output_heads.sequence_head.2.bias",
        "output_heads.sequence_head.3.weight",
        "output_heads.sequence_head.3.bias"
    ]:
        param.requires_grad = True
    else:
        param.requires_grad = False

config = CPOConfig(
    learning_rate=config.learning_rate,
    per_device_train_batch_size=config.batch_size,
    loss_type=config.loss_type,
    cpo_alpha=config.alpha,
    beta=config.beta,
    save_strategy="steps",
    save_steps=0.1,
    save_safetensors=False,
    output_dir=f"weights/{timestamp}",
    remove_unused_columns=False,
    generate_during_eval=True,
    eval_strategy="steps",
    eval_steps=0.1,
    run_name=timestamp
)

# Set resume_from_checkpoint to the latest checkpoint
resume_from_checkpoint = f"weights/{timestamp}/checkpoint-{checkpoint}"

trainer = ESMCPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    data_collator=ESMDataCollator(),
    processing_class=EsmSequenceTokenizer()
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")
def print_gpu_memory(device="cuda:0"):
    allocated = torch.cuda.memory_allocated(device) / 1024**2  # in MB
    reserved = torch.cuda.memory_reserved(device) / 1024**2   # in MB
    print(f"GPU memory on {device}:")
    print(f"  Allocated: {allocated:.2f} MB")
    print(f"  Reserved : {reserved:.2f} MB")

print_gpu_memory()
trainer.train(resume_from_checkpoint=resume_from_checkpoint)