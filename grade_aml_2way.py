#!/usr/bin/env python3
"""Per-element grader for the AML two-way (base vs LoRA) eval.
One judge-model call per reasoning element (mechanism, loophole, distinguishing); the decision is
checked directly; a scenario is fully correct only when all three elements pass and the decision is right.

Usage:
  export SB_API_KEY=...            # judge key (not stored)
  export SB_JUDGE_BASE=https://api.anthropic.com/v1/   SB_JUDGE_MODEL=claude-sonnet-5
  export SB_RUBRICS=apo_rubrics_aml.json   SB_DIR=.    SB_SCOPE=unseen   # or all
  # reads aml90_2way_BASE_answers.csv and aml90_2way_LORA_answers.csv from SB_DIR
  python3 grade_aml_2way.py           # add --smoke to self-test the judge
"""

import os, re, sys, csv, time, json, base64
RUBRICS = json.load(open(os.environ.get("SB_RUBRICS", "grade_rubrics_aml.json"), encoding="utf-8"))
SCOPE = os.environ.get("SB_SCOPE","unseen").lower()
JSYS = ("You are a strict Singapore AML examiner. You are shown the correct rubric and an analyst's full answer, and "
        "asked about ONE element only. Judge only that element; ignore whether the other elements are present. A vague "
        "or generic statement does not earn it.")
LABELS = {
 "mech": "MECHANISM - the specific laundering method / technical move (not a generic label like 'structuring')",
 "loophole": "LOOPHOLE - a determinate rule/threshold/exemption/definition/timing/document satisfied on its face, but its PURPOSE defeated",
 "distinguishing": "DISTINGUISHING - what a genuine legitimate customer would show here that this customer does NOT (the contrast with the innocent look-alike)",
}
def four_points(rubric): return rubric.split("ANSWER KEY")[0].strip()

# --- judge_elem: VERBATIM from the original working grader (OpenAI-compat client) ---
def judge_elem(client, model, rubric, elem_key, answer):
    if not (answer or "").strip(): return 0
    msg = (f"RUBRIC for this case (the correct answer):\n{four_points(rubric)}\n\n"
           f"The analyst's FULL answer:\n\"{answer[:3000]}\"\n\n"
           f"Focus ONLY on the {LABELS[elem_key]} element. Does the answer correctly convey THAT one element as "
           f"required by the rubric? Ignore the other elements entirely. Output ONLY 1 or 0.")
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
    desk = os.environ.get("SB_DIR", ".")
    if not key: print("Set SB_API_KEY."); return
    from openai import OpenAI
    client = OpenAI(base_url=base, api_key=key, timeout=90.0, max_retries=4)
    # --- SMOKE self-check first (exactly like the original) ---
    if "--smoke" in sys.argv:
        rub = ("A correct answer must:\n1. MECHANISM - sub-threshold structuring of cash.\n2. LOOPHOLE - each deposit is "
               "under the S$20,000 CDD trigger on its face but its purpose (aggregate scrutiny) is defeated by splitting.\n"
               "3. DISTINGUISHING - a genuine cash business banks its takings in one lot.\n4. DECISION - report.")
        good = judge_elem(client, jmodel, rub, "loophole", "DECISION: REPORT. Each deposit stays just under the S$20,000 CDD trigger, but the purpose - aggregate scrutiny - is defeated by deliberately splitting the cash across days; a real business would bank in one lot.")
        weak = judge_elem(client, jmodel, rub, "loophole", "DECISION: REPORT. This looks suspicious.")
        print(f"SMOKE strong (expect 1): {good}"); print(f"SMOKE weak (expect 0): {weak}")
        if good==0 and weak==0:
            print("!! judge returned all zeros - API call is failing (rate limit / key / model). Do NOT run the full grade until this shows 1 and 0.")
        return
    arms={"base":"aml90_2way_BASE_answers.csv","lora":"aml90_2way_LORA_answers.csv"}
    keep=(lambda r: True) if SCOPE=="all" else (lambda r: r["split"]=="unseen")
    print(f"UNIFORM FAIR judge = {jmodel}. One call per element, whole answer, no position bias. SCOPE={SCOPE}\n")
    SUM=[]
    for arm,fn in arms.items():
        path=os.path.join(desk,fn)
        if not os.path.exists(path): print(f"[{arm}] missing {fn} - run the cluster eval first."); continue
        rows=list(csv.DictReader(open(path)))
        V=[r for r in rows if r["instance"]=="violation" and keep(r)]
        T=[r for r in rows if r["instance"]=="benign_twin" and keep(r)]
        for i,r in enumerate(V,1):
            rub=RUBRICS.get(r["typology_id"],"")
            s={ek:judge_elem(client,jmodel,rub,ek,r.get("answer","")) for ek in ("mech","loophole","distinguishing")}
            dec=1 if r.get("passed")=="PASS" else 0
            r["_m"],r["_l"],r["_d"]=s["mech"],s["loophole"],s["distinguishing"]
            r["_fully"]=1 if (s["mech"] and s["loophole"] and s["distinguishing"] and dec) else 0
            print(f"\r  [{arm}] grading {i}/{len(V)}   ",end="",flush=True); time.sleep(0.02)
        print()
        def grp(fam=None):
            vs=[r for r in V if (fam is None or r["family"]==fam)]; ts=[r for r in T if (fam is None or r["family"]==fam)]
            n=len(vs) or 1
            det=sum(1 for r in vs if r["decision"]=="REPORT"); fp=sum(1 for r in ts if r["decision"]=="REPORT")
            return {"n":len(vs),"detection":f"{det}/{len(vs)}","false_pos":f"{fp}/{len(ts)}",
                    "mechanism":f"{100*sum(r['_m'] for r in vs)//n}%","loophole":f"{100*sum(r['_l'] for r in vs)//n}%",
                    "distinguishing":f"{100*sum(r['_d'] for r in vs)//n}%","fully":f"{100*sum(r['_fully'] for r in vs)//n}%"}
        lab="unseen" if SCOPE!="all" else "ALL"
        g=grp(); g.update({"arm":arm,"scope":lab}); SUM.append(g)
        print(f"  {arm:5} {lab:7} n={g['n']:3}  detect={g['detection']:>7}  falsepos={g['false_pos']:>6}  "
              f"mech={g['mechanism']:>4}  loop={g['loophole']:>4}  dist={g['distinguishing']:>4}  FULLY={g['fully']:>4}")
        for fam in sorted(set(r["family"] for r in V)):
            g=grp(fam); g.update({"arm":arm,"scope":f"{lab}::{fam}"}); SUM.append(g)
            print(f"       [{fam[:34]:34}] n={g['n']:2}  FULLY={g['fully']:>4}  (mech={g['mechanism']} loop={g['loophole']} dist={g['distinguishing']})")
        print()
    out=os.path.join(desk,"aml90_2way_eval_results.csv")
    cols=["arm","scope","n","detection","false_pos","mechanism","loophole","distinguishing","fully"]
    with open(out,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for r in SUM: w.writerow({k:r.get(k,"") for k in cols})
    print(f"SAVED -> {out}")
if __name__=="__main__": main()
