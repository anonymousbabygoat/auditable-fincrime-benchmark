# -*- coding: utf-8 -*-
"""AML SFT — LoRA fine-tune of Qwen2.5-7B-Instruct on the scaled matched-twin data.
Uses ONLY transformers + peft + accelerate (the SG-LegalCite stack) — NO trl, NO datasets — so the container's
PyTorch/transformer_engine is left untouched (avoids the torch-2.9 shadowing that breaks the build). bf16 on A100
(fp16 fallback), LoRA, gradient checkpointing."""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '600')

import json, torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model

MODEL  = os.environ.get('BASE_MODEL', 'Qwen/Qwen2.5-7B-Instruct')
TRAIN  = os.environ.get('TRAIN_FILE', 'aml_sft_train.jsonl')
VAL    = os.environ.get('VAL_FILE',   'aml_sft_val.jsonl')
OUT    = os.environ.get('OUT_DIR',    'aml_qwen7b_lora')
MAXSEQ = int(os.environ.get('MAX_SEQ',  '1536'))
BSZ    = int(os.environ.get('BSZ',      '4'))
GA     = int(os.environ.get('GRAD_ACCUM','4'))
EPOCHS = float(os.environ.get('EPOCHS', '3'))
LR     = float(os.environ.get('LR',     '2e-4'))

assert torch.cuda.is_available(), "no CUDA"
bf16 = torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if bf16 else torch.float16
print(f"GPU: {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB "
      f"| dtype={'bf16' if bf16 else 'fp16'} | maxseq={MAXSEQ} bsz={BSZ}x{GA} epochs={EPOCHS}")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype, trust_remote_code=True,
                                             attn_implementation='eager')
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
model.enable_input_require_grads()

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
                  target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

def load(path):
    out = []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line: continue
        msgs = json.loads(line)['messages']
        try:
            text = tok.apply_chat_template(msgs, tokenize=False)
        except Exception:
            # models whose chat template has no 'system' role (Gemma-2, some Mistral):
            # fold the system message into the first user turn, then retry.
            folded, sys_txt = [], ""
            for m in msgs:
                if m["role"] == "system":
                    sys_txt = m["content"]; continue
                if m["role"] == "user" and sys_txt:
                    folded.append({"role": "user", "content": sys_txt + "\n\n" + m["content"]}); sys_txt = ""
                else:
                    folded.append(m)
            text = tok.apply_chat_template(folded, tokenize=False)
        ids = tok(text, truncation=True, max_length=MAXSEQ)['input_ids']
        out.append({'input_ids': ids, 'attention_mask': [1]*len(ids)})
    return out

class DS(torch.utils.data.Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return self.d[i]

train_ds, val_ds = DS(load(TRAIN)), DS(load(VAL))
print(f"train={len(train_ds)}  val={len(val_ds)}")
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

args = TrainingArguments(
    output_dir=OUT, num_train_epochs=EPOCHS,
    per_device_train_batch_size=BSZ, per_device_eval_batch_size=BSZ, gradient_accumulation_steps=GA,
    learning_rate=LR, lr_scheduler_type='cosine', warmup_ratio=0.03, weight_decay=0.0,
    logging_steps=10, save_strategy='epoch', eval_strategy='epoch',
    bf16=bf16, fp16=(not bf16), gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    optim='adamw_torch', report_to='none', save_total_limit=1,
    load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=collator, tokenizer=tok)
trainer.train()
best = os.path.join(OUT, 'best_lora')
model.save_pretrained(best); tok.save_pretrained(best)
print("saved LoRA ->", best)
