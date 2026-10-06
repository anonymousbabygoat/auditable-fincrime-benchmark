#!/usr/bin/env python3
"""Per-element grader for the market-abuse two-way (base vs LoRA) eval, across all fine-tuned models.
One judge-model call per reasoning element (mechanism, breach, distinguishing); the decision is checked
directly; a scenario is fully correct only when all three elements pass and the decision is right.

Incremental: grades only models not already present in market75_sweep_results.csv (set SB_FORCE to
re-grade). Mirrors grade_aml_2way.py; the judge logic (judge_elem) is identical.

Usage:
  export SB_API_KEY=...            # judge key (not stored)
  export SB_JUDGE_BASE=https://api.anthropic.com/v1/   SB_JUDGE_MODEL=claude-sonnet-5
  export SB_RUBRICS=grade_rubrics_market.json   SB_DIR=.   SB_SCOPE=unseen   # or all
  export SB_FORCE=qwen7b,qwen3b    # optional: force re-grade these models
  # reads market75_2way_<model>_{BASE,LORA}_answers.csv from SB_DIR for each model below
  python3 grade_market_2way.py        # add --smoke to self-test the judge
"""

import os, re, sys, csv, time, json, base64
csv.field_size_limit(10**7)
RUBRICS = json.load(open(os.environ.get("SB_RUBRICS", "grade_rubrics_market.json"), encoding="utf-8"))
SCOPE = os.environ.get("SB_SCOPE", "unseen").lower()

TAGS = [
 ("qwen7b",   "market75_2way_qwen7b_BASE_answers.csv",   "market75_2way_qwen7b_LORA_answers.csv"),
 ("qwen3b",   "market75_2way_qwen3b_BASE_answers.csv",   "market75_2way_qwen3b_LORA_answers.csv"),
 ("gemma2b",  "market75_2way_gemma2b_BASE_answers.csv",  "market75_2way_gemma2b_LORA_answers.csv"),
 ("gemma9b",  "market75_2way_gemma9b_BASE_answers.csv",  "market75_2way_gemma9b_LORA_answers.csv"),
 ("llama8b",  "market75_2way_llama8b_BASE_answers.csv",  "market75_2way_llama8b_LORA_answers.csv"),
 ("llama3b",  "market75_2way_llama3b_BASE_answers.csv",  "market75_2way_llama3b_LORA_answers.csv"),
 ("mistral7b","market75_2way_mistral7b_BASE_answers.csv","market75_2way_mistral7b_LORA_answers.csv"),
]

JSYS = ("You are a strict Singapore market-abuse examiner. You are shown the correct rubric and an analyst's full answer, "
        "and asked about ONE element only. Judge only that element; ignore whether the other elements are present. A vague "
        "or generic statement does not earn it.")
LABELS = {
 "mech": "MECHANISM - the specific abusive trading/order move (not a generic label like 'spoofing')",
 "breach": "BREACH - the specific statutory provision breached and WHY its purpose is defeated (cite SFA s.197/198/201 / MAR Art 8/12/15 / IOSCO)",
 "distinguishing": "DISTINGUISHING - what a genuine legitimate participant would show here that this one does NOT (the contrast with the innocent look-alike)",
}
def four_points(r): return r.split("ANSWER KEY")[0].strip()

# --- judge_elem: VERBATIM from the original working grader (OpenAI-compat client) ---
def judge_elem(client, model, rubric, ek, answer):
    if not (answer or "").strip(): return 0
    msg = (f"RUBRIC for this case (the correct answer):\n{four_points(rubric)}\n\n"
           f"The analyst's FULL answer:\n\"{answer[:3000]}\"\n\n"
           f"Focus ONLY on the {LABELS[ek]} element. Does the answer correctly convey THAT one element as required by "
           f"the rubric? Ignore the other elements entirely. Output ONLY 1 or 0.")
    for tokkw in ("max_tokens", "max_completion_tokens"):
        try:
            r = client.chat.completions.create(model=model, messages=[{"role":"system","content":JSYS},{"role":"user","content":msg}], **{tokkw: 2000})
            ds = re.findall(r'[01]', r.choices[0].message.content or ""); return int(ds[-1]) if ds else 0
        except Exception as e:
            if any(w in str(e).lower() for w in ("max_tokens","max_completion","unsupported")): continue
            time.sleep(2)
            try:
                r = client.chat.completions.create(model=model, messages=[{"role":"system","content":JSYS},{"role":"user","content":msg}], max_completion_tokens=2000)
                ds = re.findall(r'[01]', r.choices[0].message.content or ""); return int(ds[-1]) if ds else 0
            except Exception: return 0
    return 0

def main():
    base = os.environ.get("SB_JUDGE_BASE", "https://api.anthropic.com/v1/")
    key = os.environ.get("SB_API_KEY", ""); jmodel = os.environ.get("SB_JUDGE_MODEL", "claude-sonnet-5")
    desk = os.environ.get("SB_DIR", "."); force = os.environ.get("SB_FORCE", "")
    if not key: print("Set SB_API_KEY."); return
    from openai import OpenAI
    client = OpenAI(base_url=base, api_key=key, timeout=90.0, max_retries=4)
    if "--smoke" in sys.argv:
        rub = ("A correct answer must:\n1. MECHANISM - layering: posting large one-sided orders the account does not mean "
               "to execute to move the price, then trading the other way.\n2. BREACH - SFA s.197 false trading: the orders "
               "create a false appearance of demand; their purpose is defeated.\n3. DISTINGUISHING - a genuine liquidity "
               "provider quotes both sides and lets fills stand.\n4. DECISION - report.")
        good = judge_elem(client, jmodel, rub, "breach", "DECISION: REPORT. The ladder creates a false appearance of buying demand to push the offer up, then the account sells into it - SFA s.197 false trading and s.201 manipulative device.")
        weak = judge_elem(client, jmodel, rub, "breach", "DECISION: REPORT. This looks manipulative.")
        print(f"SMOKE strong (expect 1): {good}"); print(f"SMOKE weak (expect 0): {weak}")
        if good==0 and weak==0:
            print("!! judge returned all zeros - API call is failing (rate limit / key / model). Do NOT run the full grade until this shows 1 and 0.")
        return
    out = os.path.join(desk, "market75_sweep_results.csv")
    cols = ["model","arm","n","fully","detection","false_pos","mechanism","breach","distinguishing"]
    existing = []; done = set()
    if os.path.exists(out):
        existing = list(csv.DictReader(open(out)))
        done = {r["model"] for r in existing}
    forced = set(t.strip() for t in force.split(",") if t.strip())
    done -= forced
    keep = (lambda r: True) if SCOPE == "all" else (lambda r: r["split"] == "unseen")
    print(f"Judge={jmodel} SCOPE={SCOPE}. Already graded (skipped): {sorted(done) or 'none'}\n")
    order = [t[0] for t in TAGS]
    new_rows = []
    print(f"  {'model':10} {'arm':5} {'FULLY':>6} {'detect':>8} {'falsepos':>9} {'mech':>5} {'brch':>5} {'dist':>5}")
    for tag, bfn, lfn in TAGS:
        if tag in done:
            print(f"  {tag:10} (cached - already in results, skipping)"); continue
        for arm, fn in (("base", bfn), ("lora", lfn)):
            path = os.path.join(desk, fn)
            if not os.path.exists(path): continue
            rows = list(csv.DictReader(open(path)))
            V = [r for r in rows if r["instance"] == "violation" and keep(r)]
            T = [r for r in rows if r["instance"] == "benign_twin" and keep(r)]
            if not V: continue
            m = b = ds = fu = 0
            for r in V:
                rub = RUBRICS.get(r["typology_id"], "")
                sm = judge_elem(client, jmodel, rub, "mech", r.get("answer",""))
                sb = judge_elem(client, jmodel, rub, "breach", r.get("answer",""))
                sd = judge_elem(client, jmodel, rub, "distinguishing", r.get("answer",""))
                dec = 1 if r.get("passed") == "PASS" else 0
                m += sm; b += sb; ds += sd; fu += 1 if (sm and sb and sd and dec) else 0
                print(f"\r  grading {tag}/{arm} ...        ", end="", flush=True)
            n = len(V); det = sum(1 for r in V if r["decision"] == "REPORT"); fp = sum(1 for r in T if r["decision"] == "REPORT")
            row = {"model":tag, "arm":arm, "n":n, "fully":f"{100*fu//n}%", "detection":f"{det}/{n}", "false_pos":f"{fp}/{len(T)}",
                   "mechanism":f"{100*m//n}%", "breach":f"{100*b//n}%", "distinguishing":f"{100*ds//n}%"}
            new_rows.append(row)
            print(f"\r  {tag:10} {arm:5} {row['fully']:>6} {row['detection']:>8} {row['false_pos']:>9} {row['mechanism']:>5} {row['breach']:>5} {row['distinguishing']:>5}")
    merged = [r for r in existing if r["model"] not in forced] + new_rows
    rank = {t: i for i, t in enumerate(order)}
    merged.sort(key=lambda r: (rank.get(r["model"], 99), 0 if r["arm"] == "base" else 1))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in merged: w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nSAVED -> {out}  ({len(new_rows)} new row(s) graded this run)")

if __name__ == "__main__": main()
