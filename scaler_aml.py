#!/usr/bin/env python3
"""AML data scaler: few-shot generation of matched violation/innocent-twin pairs
from expert-authored seeds, with automatic checks (schema completeness, give-away-word
guard, near-duplicate guard vs seeds >0.85 and vs sibling variants >0.75).

Usage:
  export SC_API_KEY=...            # teacher-model API key (not stored)
  export SC_BASE=https://api.anthropic.com/v1/
  export SC_MODEL=claude-sonnet-5
  export SC_SEEDS=seeds_aml.json   # expert seed pairs (90 training typologies)
  export SC_A_PER=56               # variants per typology
  export SC_OUT=AML90_scaled_FULL.csv
  python3 scaler_aml.py            # generate; set SC_REDO_FLAGGED=1 to re-fix flagged rows
"""

import os, re, sys, csv, json, base64, time, difflib
# AML DATA SCALER (few-shot generation; generator = claude-sonnet-5). Reads the 90 TRAIN typologies (embedded)
# and, for each, few-shots Sonnet to produce NEW matched violation/twin pairs in the SAME schema. Axis A only.
# Auto-checks: schema completeness + label-leak guard on STORY bodies + near-duplicate flag vs the sealed seeds.
SEEDS = json.load(open(os.environ.get("SC_SEEDS", "seeds_aml.json"), encoding="utf-8"))
SEEDS = [s for s in SEEDS if s.get("split","seen")=="seen"]  # never generate held-out typologies

SYSTEM = ("You are a senior Singapore AML/CFT methodology expert building TRAINING scenarios for a benchmark. "
          "Each scenario is a matched pair: a genuinely suspicious 'violation' and an innocent 'look-alike twin' "
          "that shares the SAME surface profile but is legitimate. The twin must be a HARD look-alike - not "
          "obviously innocent - so that only careful reasoning separates them. Ground everything in realistic "
          "Singapore facts (MAS Notice 626, PSN01/PSN02, CDD thresholds, SGD amounts, plausible businesses). Never "
          "copy the example verbatim; change names, amounts, dates, instruments, sector and counterparties.")

LEAK_WORDS = ("launder","suspicious","suspicion","innocent","look-alike","lookalike","illicit","evasion",
              "evade","conceal","red flag","typology","legitimate look","str ","sar ","the twin","violation",
              "smurf","structuring scheme")

FORMAT = ("CRITICAL - NO LABEL LEAKAGE: the two STORY bodies must be STRICTLY FACTUAL and NEUTRAL. They must NOT "
    "contain any word that reveals whether the case is suspicious or clean - never use words like 'laundering', "
    "'suspicious', 'innocent', 'look-alike', 'illicit', 'conceal', 'evade', 'red flag', 'typology', 'legitimate', "
    "'structuring', or any meta-description. A reader of the story alone must be UNABLE to tell which is which - "
    "only the RUBRIC (never shown to the model under test) may name the abuse. Put judgement words ONLY in rubrics.\n\n"
    "Mirror the example's structure: PROFILE (customer/entity facts on file), ACTIVITY (dated transaction entries), "
    "and a neutral quoted CUSTOMER REQUEST. The twin must share the same surface profile but be genuinely legitimate.\n\n"
    "Output EXACTLY in this format, nothing before or after - use these literal header lines. Inside <...> put only "
    "neutral facts (parties, amounts, dates, instruments, documents, requests):\n"
    "FAMILY: <the family>\n"
    "TYPOLOGY: <short specific typology name>\n"
    "VIOLATION_STORY:\nPROFILE: <...>\nACTIVITY: <dated entries>\nCUSTOMER REQUEST: <a neutral quoted request>\n"
    "VIOLATION_RUBRIC:\nA correct answer must:\n1. MECHANISM - <the specific laundering method / technical move>.\n"
    "2. LOOPHOLE - <which rule/threshold/exemption is met on its face and how its PURPOSE is defeated; cite MAS 626 / CDSA where apt>.\n"
    "3. DISTINGUISHING - <the specific facts that separate this from the innocent look-alike>.\n4. DECISION - report; escalate to compliance.\n"
    "TWIN_STORY:\nPROFILE: <same surface profile, but genuinely legitimate facts>\nACTIVITY: <dated entries>\nCUSTOMER REQUEST: <a neutral quoted request>\n"
    "TWIN_RUBRIC:\nA correct CLEAR must:\n1. LEGITIMATE ACTIVITY - <name the bona fide activity>.\n"
    "2. WHY NOT ABUSE - <why the same surface facts are legitimate; name the control NOT defeated>.\n"
    "3. EXCULPATORY - <the specific exculpatory facts and the TRAP that resembles the abuse>.\n4. DECISION - proceed; document and clear.\n"
    "END")

def example_block(s):
    return (f"FAMILY: {s['family']}\nTYPOLOGY: {s['typology']}\n"
            f"VIOLATION_STORY:\n{s['vstory']}\nVIOLATION_RUBRIC:\n{s['vrubric']}\n"
            f"TWIN_STORY:\n{s['tstory']}\nTWIN_RUBRIC:\n{s['trubric']}\nEND")

DIVAXES=["a different industry sector or business type","a different customer archetype (individual / sole-prop / SME / large corporate / trust / SPV)",
         "a different instrument or product (cash / wire / cheque / card / prepaid / crypto / insurance / trade finance)",
         "a different SGD amount band (small four-figure / mid five-figure / large six-figure)",
         "a different counterparty geography and channel (branch / online / remittance / correspondent)",
         "a different sector and set of documents on file","a fresh set of names, entities and dates unlike common examples"]

def build_user(job):
    s=job["seed"]
    hint=DIVAXES[job["idx"] % len(DIVAXES)]
    return ("Here is ONE validated example pair. Generate ONE NEW pair with the SAME typology and underlying "
            "laundering mechanism, but COMPLETELY different surface details (names, amounts, dates, instrument, "
            "sector, counterparties). Keep the twin a HARD, realistic look-alike of the same difficulty.\n"
            f"FOR DIVERSITY, deliberately anchor this variant to {hint}, and make it clearly distinct from other "
            "variants of this same typology (different names, sizes, sectors, documents).\n\n"
            f"=== VALIDATED EXAMPLE (typology: {s['typology']}) ===\n{example_block(s)}\n\n"
            "=== NOW GENERATE ONE NEW PAIR ===\n" + FORMAT)

def _create(client, model, system, user, toks):
    tokkw=os.environ.get("SC_TOKEN_PARAM","max_tokens")
    return client.chat.completions.create(model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":user}], **{tokkw:toks})

def _complete(t):
    return bool(re.search(r"TWIN_RUBRIC", t or "", re.I)) and bool(re.search(r"\bEND\b", t or ""))

def ask(client, model, system, user):
    toks=int(os.environ.get("SC_MAXTOK","8000")); cap=int(os.environ.get("SC_MAXTOK_CAP","16000"))
    best=""; last=""
    for attempt in range(8):
        try:
            r=_create(client,model,system,user,toks); ch=r.choices[0]
            txt=(ch.message.content or "").strip(); fin=getattr(ch,"finish_reason","") or ""
            if len(txt)>len(best): best=txt
            truncated=(fin=="length") or (txt and not _complete(txt))
            if txt and not truncated: return txt
            if not txt: last="[ERROR] empty content"
            if toks>=cap: return best or last
            toks=min(int(toks*1.6),cap); time.sleep(2)
        except Exception as e:
            s=str(e); sl=s.lower()
            if ("max_tokens" in s or "max_completion" in s or "unsupported" in sl) and os.environ.get("SC_TOKEN_PARAM")!="max_completion_tokens":
                os.environ["SC_TOKEN_PARAM"]="max_completion_tokens"; continue
            last="[ERROR] "+type(e).__name__+": "+s[:120]
            time.sleep(min(60,5*(2**attempt)) if any(w in sl for w in ("429","rate","overload","529","503","500","timeout","502")) else 3)
    return best or last

FIELDS=["mid","seed_id","family","axis","typology","violation_story","violation_rubric","twin_story","twin_rubric",
        "generator","dup_ratio","flags","daniel_validation"]

def parse(text):
    def grab(a,b):
        m=re.search(re.escape(a)+r"\s*(.*?)\s*(?="+b+")", text, re.S|re.I)
        return (m.group(1).strip() if m else "")
    fam=grab("FAMILY:", r"TYPOLOGY:"); typ=grab("TYPOLOGY:", r"VIOLATION_STORY:")
    vs=grab("VIOLATION_STORY:", r"VIOLATION_RUBRIC:"); vr=grab("VIOLATION_RUBRIC:", r"TWIN_STORY:")
    ts=grab("TWIN_STORY:", r"TWIN_RUBRIC:"); tr=grab("TWIN_RUBRIC:", r"END\b|$")
    return fam,typ,vs,vr,ts,tr

def dup_ratio(vs):
    best=0.0
    for s in SEEDS:
        sm=difflib.SequenceMatcher(None, vs, s["vstory"] or "")
        if sm.real_quick_ratio()<best: continue
        best=max(best, sm.ratio())
    return round(best,2)

def jobs():
    if os.environ.get("SC_SMOKE")=="1":
        idxs=[0, len(SEEDS)//2, len(SEEDS)-1]
        return [{"axis":"A","seed":SEEDS[i],"idx":0} for i in idxs]
    a_per=int(os.environ.get("SC_A_PER","0")); out=[]
    for s in SEEDS:
        for i in range(a_per): out.append({"axis":"A","seed":s,"idx":i})
    lim=int(os.environ.get("SC_LIMIT","0"))
    return out[:lim] if lim else out

def intra_dup(vs, others):
    if not vs or not others: return 0.0
    best=0.0
    for o in others:
        sm=difflib.SequenceMatcher(None, vs, o or "")
        if sm.real_quick_ratio()<best: continue
        best=max(best, sm.ratio())
    return round(best,2)

def compute_flags(ans, vs, vr, ts, tr, intra_others=None):
    flags=[]
    if ans.startswith("[ERROR]"): flags.append("ERROR:"+ans[:60])
    for nm,val in [("vstory",vs),("vrubric",vr),("tstory",ts),("trubric",tr)]:
        if len(val)<40: flags.append("short_"+nm)
    for nm,body in [("vstory",vs),("tstory",ts)]:
        hit=[w for w in LEAK_WORDS if w in (body or "").lower()]
        if hit: flags.append(f"leak_{nm}:{hit[0].strip()}")
    dr=dup_ratio(vs) if vs else 1.0
    if dr>float(os.environ.get("SC_DUP","0.85")): flags.append(f"dup?{dr}")
    idr=intra_dup(vs, intra_others) if intra_others else 0.0     # vs prior variants of SAME typology
    if idr>float(os.environ.get("SC_INTRADUP","0.75")): flags.append(f"intradup?{idr}")
    return flags, dr

def gen_row(client, model, job, intra_others=None):
    ans=ask(client, model, SYSTEM, build_user(job))
    fam,typ,vs,vr,ts,tr=parse(ans)
    flags,dr=compute_flags(ans,vs,vr,ts,tr,intra_others)
    mid=f"{job['seed']['id']}-A{job['idx']+1}"
    row={"mid":mid,"seed_id":job["seed"]["id"],"family":fam or job["seed"]["family"],"axis":job["axis"],
         "typology":typ or job["seed"]["typology"],"violation_story":vs,"violation_rubric":vr,
         "twin_story":ts,"twin_rubric":tr,"generator":model,"dup_ratio":dr,"flags":";".join(flags),
         "daniel_validation":""}
    return row, flags

def clean_flagged(client, model, out):
    if not os.path.exists(out): sys.exit("No CSV to clean at "+out)
    existing=list(csv.DictReader(open(out))); order=[r["mid"] for r in existing]
    by_mid={r["mid"]:r for r in existing}
    a_per=int(os.environ.get("SC_A_PER","10"))
    alljobs={f"{s['id']}-A{i+1}":{"axis":"A","seed":s,"idx":i} for s in SEEDS for i in range(a_per)}
    flagged=[m for m in order if str(by_mid[m].get("flags","")).strip()]
    for m in alljobs:
        if m not in by_mid: flagged.append(m); order.append(m)
    flagged=sorted(set(flagged), key=lambda m:(m.split("-")[0], m))
    print(f"[clean] {len(flagged)} flagged/missing rows to fix\n")
    if not flagged: print("Nothing to fix."); return
    tries=int(os.environ.get("SC_TRIES","4"))
    for m in flagged:
        job=alljobs[m]; best=None; best_flags=None
        sid=job["seed"]["id"]
        others=[by_mid[x]["violation_story"] for x in order if x in by_mid and by_mid[x].get("seed_id")==sid and x!=m]
        for t in range(tries):
            row,flags=gen_row(client,model,job,intra_others=others)
            if not flags:
                by_mid[m]=row; best=None; print(f"{m:12} fixed on try {t+1}"); break
            if best is None or len(flags)<len(best_flags): best,best_flags=row,flags
            print(f"{m:12} try {t+1} still: {';'.join(flags)}")
        else:
            by_mid[m]=best; print(f"{m:12} kept best: {';'.join(best_flags)}")
        rows=[by_mid[x] for x in order if x in by_mid]
        with open(out,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    still=sum(1 for x in order if str(by_mid[x].get("flags","")).strip())
    print(f"\n[clean] done. {still} row(s) still flagged.")

def main():
    base=os.environ.get("SC_BASE","https://api.anthropic.com/v1/"); key=os.environ.get("SC_API_KEY","")
    model=os.environ.get("SC_MODEL","claude-sonnet-5"); outdir=os.environ.get("OUTDIR", os.path.expanduser("~/Desktop"))
    out=os.path.join(outdir, os.environ.get("SC_OUT","AML90_scaled_FULL.csv"))
    if not key: sys.exit("set SC_API_KEY")
    from openai import OpenAI
    client=OpenAI(base_url=base, api_key=key, timeout=180.0, max_retries=4)
    if os.environ.get("SC_REDO_FLAGGED")=="1": clean_flagged(client, model, out); return
    J=jobs()
    if not J: sys.exit("No jobs - set SC_A_PER (and SC_LIMIT for a smoke).")
    done={}; seen={}
    if os.environ.get("SC_RESUME")=="1" and os.path.exists(out):
        for r in csv.DictReader(open(out)):
            if r.get("violation_story") and not str(r.get("flags","")).startswith("ERROR"):
                done[r["mid"]]=r; seen.setdefault(r["seed_id"],[]).append(r["violation_story"])
        print(f"[resume] {len(done)} good rows kept.")
    rows=[]; smoke=os.environ.get("SC_LIMIT") or os.environ.get("SC_SMOKE")
    print(f"AML SCALER (90 train typologies; generator={model}): {len(J)} pairs to generate\n")
    for n,job in enumerate(J,1):
        mid=f"{job['seed']['id']}-A{job['idx']+1}"; sid=job["seed"]["id"]
        if mid in done: rows.append(done[mid]); print(f"{mid:12} (kept)"); continue
        row,flags=gen_row(client,model,job,intra_others=seen.get(sid)); vs,ts,dr=row["violation_story"],row["twin_story"],row["dup_ratio"]
        rows.append(row)
        if vs: seen.setdefault(sid,[]).append(vs)
        print(f"{mid:12} {job['axis']}  dup={dr}  {('FLAGS:'+';'.join(flags)) if flags else 'ok'}")
        if smoke:
            print("   VIOLATION:", (vs or "(empty)")[:180].replace("\n"," "))
            print("   TWIN     :", (ts or "(empty)")[:180].replace("\n"," "))
        with open(out,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
        time.sleep(float(os.environ.get("SC_SLEEP","0.3")))
    bad=sum(1 for r in rows if r.get("flags"))
    print(f"\nwrote {len(rows)} -> {out}  ({bad} flagged - re-run CLEAN to fix)")
    sn=int(os.environ.get("SC_SAMPLE","0"))
    if sn>0:
        good=[r for r in rows if not r.get("flags")]; step=max(1,len(good)//sn); samp=good[::step][:sn]
        sp=os.path.join(outdir, os.environ.get("SC_SAMPLE_OUT","AML90_scaled_DANIEL_sample20.csv"))
        with open(sp,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(samp)
        print(f"[sample] wrote {len(samp)} rows for Daniel spot-check -> {sp}")

if __name__=="__main__": main()
