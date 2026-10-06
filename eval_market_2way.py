# -*- coding: utf-8 -*-
"""MARKET two-way held-out eval: base model (+optional LoRA) on the 100 ORIGINAL seed cases, each tagged
split=seen (75 trained typologies) / unseen (25 held-out). violation + benign_twin per seed. Uses the EXACT
market SFT prompt. Writes answers CSV (cols: mid,typology_id,family,split,instance,gold_decision,decision,passed,answer).
Self-contained: 100 market seeds embedded (base64). USE_LORA=1 + LORA_DIR for LoRA; USE_LORA=0 for base."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import re, csv, json, base64, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SEEDS = json.load(open(os.environ.get("EVAL_SEEDS", "seeds_market.json"), encoding="utf-8"))

MODEL   = os.environ.get('BASE_MODEL', 'Qwen/Qwen2.5-7B-Instruct')
LORA    = os.environ.get('LORA_DIR', 'market75_qwen7b_lora/best_lora')
USE_LORA= os.environ.get('USE_LORA','1')=='1'
OUT     = os.environ.get('OUT_CSV', 'market75_2way_LORA_answers.csv' if USE_LORA else 'market75_2way_BASE_answers.csv')
MAXNEW  = int(os.environ.get('MAX_NEW','512'))

SYSTEM = ("You are an experienced market surveillance analyst at a firm regulated in Singapore. You are given a "
          "trading and order record and must decide whether the activity should be REPORTED to surveillance as market "
          "abuse (for escalation), or is legitimate and can be CLEARED. Judge by the substance and PURPOSE of the "
          "market-abuse rules (SFA Part XII; EU MAR; IOSCO), not merely whether a rule is technically satisfied on its "
          "face. When you report, give a complete, auditable justification.")
INSTR = ("\n\nDecide whether this should be REPORTED to surveillance as market abuse, or can be CLEARED as legitimate.\n"
         "Answer in EXACTLY these lines, nothing before them:\n"
         "DECISION: REPORT or CLEAR\n"
         "TYPOLOGY: the specific market-abuse typology (or NONE if legitimate)\n"
         "MECHANISM: the specific abusive move the trader is using\n"
         "BREACH: which statutory provision is breached and why (cite SFA s.197/198/201 / MAR Art 8/12/15 / IOSCO)\n"
         "DISTINGUISHING: the specific fact(s) that separate this from a genuine, legitimate look-alike")

def parse_decision(text):
    tl=(text or "").lower()
    m=re.search(r'decision\s*:\s*(report|clear)', tl)
    if m: return m.group(1).upper()
    rep=any(w in tl for w in ("report","flag","escalate","suspicious","manipulat","abuse"))
    clr=any(w in tl for w in ("clear","legitimate","not abuse","no action","bona fide"))
    if rep and not clr: return "REPORT"
    if clr and not rep: return "CLEAR"
    return "UNCLEAR"

ITEMS=[]
for s in SEEDS:
    ITEMS.append({"typology_id":s["id"],"family":s["family"],"split":s["split"],
                  "instance":"violation","gold":"REPORT","story":s["vstory"]})
    ITEMS.append({"typology_id":s["id"],"family":s["family"],"split":s["split"],
                  "instance":"benign_twin","gold":"CLEAR","story":s["tstory"]})

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print("eval:", len(ITEMS), "items (", sum(1 for i in ITEMS if i["split"]=="seen"), "seen +",
      sum(1 for i in ITEMS if i["split"]=="unseen"), "unseen) | LoRA=", USE_LORA)
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
        prompt=tok.apply_chat_template(
            [{"role":"user","content":SYSTEM+"\n\n"+it["story"].strip()+INSTR}],
            tokenize=False, add_generation_prompt=True)
    enc=tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out=model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
    ans=tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    dec=parse_decision(ans)
    passed="PASS" if dec==it["gold"] else "FAIL"
    rows.append({"mid":it["typology_id"]+"-"+it["instance"],"typology_id":it["typology_id"],
                 "family":it["family"],"split":it["split"],"instance":it["instance"],
                 "gold_decision":it["gold"],"decision":dec,"passed":passed,"answer":ans})
    if i%20==0 or i==len(ITEMS): print("  ", i, "/", len(ITEMS))
    with open(OUT,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def rate(inst,gold,split=None):
    sub=[r for r in rows if r["instance"]==inst and (split is None or r["split"]==split)]
    hit=sum(1 for r in sub if r["decision"]==gold)
    return str(hit)+"/"+str(len(sub))
print("\n== decision-only (pre-grading) ==")
for sp in ("seen","unseen"):
    print("  ["+sp+"] detection(REPORT on violation)="+rate("violation","REPORT",sp)+
          "  false-pos(REPORT on twin)="+rate("benign_twin","REPORT",sp))
print("wrote ->", OUT)
