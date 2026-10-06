# -*- coding: utf-8 -*-
"""AML two-way held-out eval: base Qwen2.5-7B-Instruct (+optional LoRA) on the 120 ORIGINAL seed cases,
each tagged split=seen (90 trained typologies) / unseen (30 held-out). violation + benign_twin per seed.
Uses the EXACT SFT training prompt so the LoRA is tested in-distribution. Writes an answers CSV that the
Claude/Kimi grader consumes (columns: mid,typology_id,family,split,instance,gold_decision,decision,passed,answer).
Self-contained: 120 seeds embedded (base64). Set USE_LORA=1 + LORA_DIR to eval the LoRA; USE_LORA=0 for base."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import re, csv, json, base64, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SEEDS = json.load(open(os.environ.get("EVAL_SEEDS", "seeds_aml.json"), encoding="utf-8"))
MODEL   = os.environ.get('BASE_MODEL', 'Qwen/Qwen2.5-7B-Instruct')
LORA    = os.environ.get('LORA_DIR', 'aml90_qwen7b_lora/best_lora')
USE_LORA= os.environ.get('USE_LORA','1')=='1'
OUT     = os.environ.get('OUT_CSV', 'aml90_2way_LORA_answers.csv' if USE_LORA else 'aml90_2way_BASE_answers.csv')
MAXNEW  = int(os.environ.get('MAX_NEW','512'))

# EXACT SFT training prompt (must match build_aml90_sft.py so the LoRA sees the format it learned)
SYSTEM = ("You are an experienced AML/CFT analyst at a bank regulated in Singapore. You are given a customer's "
          "profile and transaction record and must decide whether the activity should be REPORTED to compliance as "
          "suspicious (for a possible STR), or is legitimate and can PROCEED. Judge by the substance and PURPOSE of "
          "the AML rules (MAS Notice 626; CDSA; FATF standards), not merely whether a rule is technically satisfied "
          "on its face. When you report, give a complete, auditable justification.")
INSTR = ("\n\nDecide whether this should be REPORTED to compliance as suspicious, or can PROCEED as legitimate.\n"
         "Answer in EXACTLY these lines, nothing before them:\n"
         "DECISION: REPORT or PROCEED\n"
         "TYPOLOGY: the specific laundering typology (or NONE if legitimate)\n"
         "MECHANISM: the specific technical move the customer is using\n"
         "LOOPHOLE: which rule/threshold/exemption is met on its face and how its purpose is defeated (cite MAS 626 / CDSA)\n"
         "DISTINGUISHING: the specific fact(s) that separate this from a genuine, legitimate look-alike")

def parse_decision(text):
    tl=(text or "").lower()
    m=re.search(r'decision\s*:\s*(report|proceed)', tl)
    if m: return m.group(1).upper()
    rep=any(w in tl for w in ("report","flag","suspicious","escalate","file a str","file an str"," sar"))
    proc=any(w in tl for w in ("proceed","legitimate","not suspicious","no action","clear"))
    if rep and not proc: return "REPORT"
    if proc and not rep: return "PROCEED"
    return "UNCLEAR"

# build eval items: violation (gold REPORT) + benign_twin (gold PROCEED) per seed
ITEMS=[]
for s in SEEDS:
    ITEMS.append({"typology_id":s["id"],"family":s["family"],"split":s["split"],
                  "instance":"violation","gold":"REPORT","story":s["vstory"]})
    ITEMS.append({"typology_id":s["id"],"family":s["family"],"split":s["split"],
                  "instance":"benign_twin","gold":"PROCEED","story":s["tstory"]})

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"eval: {len(ITEMS)} items ({sum(1 for i in ITEMS if i['split']=='seen')} seen + {sum(1 for i in ITEMS if i['split']=='unseen')} unseen) | LoRA={USE_LORA}")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype, trust_remote_code=True, device_map="auto")
if USE_LORA:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, LORA); print("loaded LoRA:", LORA)
model.eval()

rows=[]
for i,it in enumerate(ITEMS,1):
    msgs=[{"role":"system","content":SYSTEM},{"role":"user","content":it["story"].strip()+INSTR}]
    try:
        prompt=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        # no 'system' role (Gemma-2, some Mistral): fold system into the user turn
        prompt=tok.apply_chat_template(
            [{"role":"user","content":SYSTEM+"\n\n"+it["story"].strip()+INSTR}],
            tokenize=False, add_generation_prompt=True)
    enc=tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out=model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    ans=tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    dec=parse_decision(ans)
    passed="PASS" if dec==it["gold"] else "FAIL"
    rows.append({"mid":f'{it["typology_id"]}-{it["instance"]}',"typology_id":it["typology_id"],
                 "family":it["family"],"split":it["split"],"instance":it["instance"],
                 "gold_decision":it["gold"],"decision":dec,"passed":passed,"answer":ans})
    if i%20==0 or i==len(ITEMS): print(f"  {i}/{len(ITEMS)}")
    with open(OUT,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# quick deterministic summary (detection / false-pos by split)
def rate(inst,gold,split=None):
    sub=[r for r in rows if r["instance"]==inst and (split is None or r["split"]==split)]
    hit=sum(1 for r in sub if r["decision"]==gold)
    return f"{hit}/{len(sub)}"
print("\n== decision-only (pre-grading) ==")
for sp in ("seen","unseen"):
    print(f"  [{sp}] detection(REPORT on violation)={rate('violation','REPORT',sp)}  "
          f"false-pos(REPORT on twin)={rate('benign_twin','REPORT',sp)}")
print("wrote ->", OUT)
