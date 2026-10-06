# DSPy APO skeleton - Act 1: automatic prompt optimization for auditable AML explanations (frozen frontier model).
# Self-contained: the matched-twin benchmark + rubrics are embedded. Optimizes the prompt with MIPROv2,
# compares to the unoptimized baseline on a held-out test split, and PRINTS the discovered instruction
# (the key question: does the optimizer rediscover "reason about the innocent look-alike first"?).
#
# Requires:  pip install -U dspy-ai openai
# Run:       set the env vars below (or export OPENAI_API_KEY), then  python3 dspy_apo_aml.py
import os, re, json, base64, random

# ================= CONFIG =================
TASK_MODEL  = os.environ.get("APO_TASK_MODEL", "openai/gpt-5.4")   # the FROZEN model whose prompt we optimize
TASK_BASE   = os.environ.get("APO_TASK_BASE", "")                   # OpenAI-compat base URL (e.g. DeepSeek); blank = default
TASK_KEY    = os.environ.get("OPENAI_API_KEY") or os.environ.get("APO_TASK_KEY", "")
JUDGE_BASE  = os.environ.get("APO_JUDGE_BASE", "https://api.openai.com/v1")  # cheap grader for the metric loop
JUDGE_MODEL = os.environ.get("APO_JUDGE_MODEL", "gpt-5-mini")      # USE A CHEAP MODEL here (loop calls it a lot)
JUDGE_KEY   = os.environ.get("APO_JUDGE_KEY") or TASK_KEY
USE_EXPL    = os.environ.get("APO_USE_EXPLANATION", "1") == "1"    # set 0 for a cheap decision-only first run
AUTO        = os.environ.get("APO_AUTO", "light")                  # MIPROv2 preset: light | medium | heavy
random.seed(13)

BUNDLE = json.load(open(os.environ.get("APO_BUNDLE", "apo_bundle_aml.json"), encoding="utf-8"))
RUBRICS = json.load(open(os.environ.get("APO_RUBRICS", "apo_rubrics_aml.json"), encoding="utf-8"))

# ================= DATA =================
def case_text(it):
    tr = it.get("tool_returns", {})
    prof = tr.get("get_account_profile", "(not provided)")
    act  = tr.get("get_account_activity", "(not provided)")
    req  = " / ".join(it.get("messages", []))
    return f"CUSTOMER PROFILE: {prof}\n\nACCOUNT ACTIVITY: {act}\n\nCUSTOMER REQUEST: {req}"

import dspy
examples = []
for it in BUNDLE["items"]:
    gold = "REPORT" if it["instance"] == "violation" else "PROCEED"
    ex = dspy.Example(
        case=case_text(it),
        gold_decision=gold,
        instance=it["instance"],
        typology_id=it["typology_id"],
        rubric=RUBRICS.get(it["typology_id"], ""),
    ).with_inputs("case")
    examples.append(ex)

random.shuffle(examples)
n = len(examples)
train = examples[: int(0.5 * n)]
val   = examples[int(0.5 * n): int(0.7 * n)]
test  = examples[int(0.7 * n):]
print(f"examples={n}  train={len(train)}  val={len(val)}  test={len(test)}   (100 guilty + 100 innocent twins)")

# ================= SIGNATURE + PROGRAM =================
class AuditableAML(dspy.Signature):
    """Decide whether a bank customer's activity should be REPORTED to compliance as suspicious money
    laundering, or can PROCEED. Judge by the substance and PURPOSE of the AML rules, not merely whether a
    rule is technically satisfied on its face. Give a complete, auditable justification."""
    case          = dspy.InputField(desc="the customer profile, account activity, and request")
    decision      = dspy.OutputField(desc="REPORT or PROCEED")
    typology      = dspy.OutputField(desc="the specific laundering method, or NONE if legitimate")
    mechanism     = dspy.OutputField(desc="the specific technical move the customer is using")
    loophole      = dspy.OutputField(desc="which rule/threshold/exemption is met on its face but whose PURPOSE is defeated")
    distinguishing = dspy.OutputField(desc="what a genuine, legitimate customer would show here that this customer does NOT")

# MIPROv2 optimizes this program's instruction (+ few-shot demos). Swap to dspy.ChainOfThought for a CoT variant.
program = dspy.Predict(AuditableAML)
if os.environ.get("APO_WEAK_SEED","0") == "1":
    try:
        program.signature = program.signature.with_instructions(
            "Decide whether the customer activity should be REPORTED to compliance as suspicious money laundering or can PROCEED, and briefly justify.")
        print("[weak seed] starting from a minimal instruction so APO has room to climb.")
    except Exception as _e:
        print("weak-seed override failed:", _e)


# ================= JUDGE (cheap, used inside the metric) =================
from openai import OpenAI
_judge = OpenAI(base_url=JUDGE_BASE, api_key=JUDGE_KEY)
JSYS = ("You are a strict Singapore AML examiner. Given the correct rubric and an analyst's answer, judge ONE "
        "element only. A vague or generic statement does NOT earn it. Output ONLY 1 or 0.")
LAB = {"mech": "MECHANISM - the specific laundering method / technical move",
       "loophole": "LOOPHOLE - a rule/threshold/exemption met on its face but whose PURPOSE is defeated",
       "distinguishing": "DISTINGUISHING - what a genuine customer would show that this one does NOT"}
def four_points(r): return (r or "").split("ANSWER KEY")[0].strip()
def judge_elem(rubric, key, answer):
    if not (answer or "").strip(): return 0
    msg = (f"RUBRIC (correct answer):\n{four_points(rubric)}\n\nANALYST ANSWER:\n\"{answer[:1500]}\"\n\n"
           f"Focus ONLY on the {LAB[key]} element. Does the answer correctly convey THAT element? Output ONLY 1 or 0.")
    try:
        r = _judge.chat.completions.create(model=JUDGE_MODEL,
            messages=[{"role": "system", "content": JSYS}, {"role": "user", "content": msg}], max_tokens=2000)
        ds = re.findall(r"[01]", r.choices[0].message.content or ""); return int(ds[-1]) if ds else 0
    except Exception as e:
        if not _JUDGE_ERR[0]:
            _JUDGE_ERR[0]=True
            print(f"\n[JUDGE ERROR - explanation grades will be 0 until fixed] {type(e).__name__}: {str(e)[:160]}\n")
        return 0
_JUDGE_ERR=[False]

def parse_decision(t):
    tl = (t or "").lower()
    rep  = any(w in tl for w in ("report", "flag", "suspicious", "escalat", "str", "sar"))
    proc = any(w in tl for w in ("proceed", "clear", "legitimate", "not suspicious", "no action"))
    if rep and not proc: return "REPORT"
    if proc and not rep: return "PROCEED"
    return "REPORT" if rep else ("PROCEED" if proc else "UNCLEAR")

# ================= METRIC (composite: decision + explanation, false-positive = worst) =================
def metric(example, pred, trace=None):
    dec  = parse_decision(getattr(pred, "decision", "") or "")
    # innocent twin: correct = PROCEED; a false positive (wrongly REPORT) is the primary failure -> 0
    if example.instance == "benign_twin":
        return 1.0 if dec == "PROCEED" else 0.0
    # guilty: must REPORT, else the crime was missed
    if dec != "REPORT":
        return 0.0
    if not USE_EXPL:
        return 1.0
    ans = (f"MECHANISM: {getattr(pred,'mechanism','')}  LOOPHOLE: {getattr(pred,'loophole','')}  "
           f"DISTINGUISHING: {getattr(pred,'distinguishing','')}")
    m = judge_elem(example.rubric, "mech", ans)
    l = judge_elem(example.rubric, "loophole", ans)
    d = judge_elem(example.rubric, "distinguishing", ans)
    if os.environ.get("APO_METRIC","strict") == "strict":
        return 1.0 if (m and l and d) else 0.0   # FULLY-CORRECT: all 3 explanation elements (headroom)
    return 0.4 + 0.6 * ((m + l + d) / 3.0)        # composite (softer)

# ================= OPTIMIZE =================
_lm_kw = dict(api_key=TASK_KEY, max_tokens=8000, temperature=0.0)
if TASK_BASE: _lm_kw["api_base"] = TASK_BASE
dspy.configure(lm=dspy.LM(TASK_MODEL, **_lm_kw))
from dspy.teleprompt import MIPROv2
print(f"\nOptimizing with MIPROv2 (auto={AUTO}) on {TASK_MODEL}; loop-judge={JUDGE_MODEL}; explanation-in-metric={USE_EXPL}")
_mipro_kw = dict(metric=metric, auto=AUTO, num_threads=8)
if os.environ.get("APO_NO_DEMOS","0") == "1":
    _mipro_kw.update(max_bootstrapped_demos=0, max_labeled_demos=0)   # instruction-only: NO worked examples
    print("[instruction-only] few-shot demos disabled - APO may improve ONLY by rewording the instruction.")
tp = MIPROv2(**_mipro_kw)
optimized = tp.compile(program, trainset=train, valset=val, requires_permission_to_run=False)

# ================= EVALUATE (held-out test) =================
from dspy.evaluate import Evaluate
ev = Evaluate(devset=test, metric=metric, num_threads=8, display_progress=True)
print("\n=== BASELINE (unoptimized prompt) on held-out test ===")
base_score = ev(program)
print("\n=== OPTIMIZED (MIPROv2) on held-out test ===")
opt_score = ev(optimized)
print(f"\nBASELINE test score : {base_score}")
print(f"OPTIMIZED test score: {opt_score}")

# ================= THE MONEY OUTPUT: the discovered instruction =================
print("\n================= OPTIMIZED PROMPT (does it rediscover 'innocent-first'?) =================")
try:
    for p in optimized.predictors():
        print(p.signature.instructions)
        print("---- few-shot demos:", len(getattr(p, "demos", []) or []))
except Exception as e:
    print("(inspect optimized program manually):", e)
optimized.save("dspy_apo_aml_optimized.json")
print("\nSaved optimized program -> ~/Desktop/dspy_apo_aml_optimized.json")
print("\nNEXT: compare this optimized score against your hand-designed Contrast/IHF, and read the instruction above")
print("      to see whether the optimizer independently discovered 'consider the legitimate look-alike first'.")

# ================= 6-COLUMN REPORT (full 100 guilty + 100 twins, same columns as the ablation table) =================
if os.environ.get("APO_REPORT", "0") == "1":
    def eval_arm(prog):
        Vrep = Vunc = Trep = Tunc = 0
        mech = loop = dist = full = 0
        nV = nT = 0
        for ex in examples:
            try:
                pred = prog(case=ex.case)
            except Exception:
                pred = None
            dec = parse_decision(getattr(pred, "decision", "") if pred else "")
            if ex.instance == "violation":
                nV += 1
                if dec == "REPORT": Vrep += 1
                elif dec == "UNCLEAR": Vunc += 1
                ans = ((f"MECHANISM: {getattr(pred,'mechanism','')}  LOOPHOLE: {getattr(pred,'loophole','')}  "
                        f"DISTINGUISHING: {getattr(pred,'distinguishing','')}") if pred else "")
                m = judge_elem(ex.rubric, "mech", ans); l = judge_elem(ex.rubric, "loophole", ans); d = judge_elem(ex.rubric, "distinguishing", ans)
                mech += m; loop += l; dist += d
                if dec == "REPORT" and m and l and d: full += 1
            else:
                nT += 1
                if dec == "REPORT": Trep += 1
                elif dec == "UNCLEAR": Tunc += 1
        pc = lambda x: f"{round(100*x/max(nV,1))}%"
        print(f"    Detection {Vrep}/{nV}" + (f" ({Vunc} empty)" if Vunc else "")
              + f" | False-Pos {Trep}/{nT}" + (f" ({Tunc} empty)" if Tunc else "")
              + f" | Mechanism {pc(mech)} | Loophole {pc(loop)} | Distinguishing {pc(dist)} | Fully {pc(full)}")
    print("\n================= 6-COLUMN REPORT (full 100 guilty + 100 twins, same grader/columns as ablation) =================")
    print("  WEAK-SEED (APO baseline):"); eval_arm(program)
    print("  APO-OPTIMIZED:"); eval_arm(optimized)
    print("  (Detection & False-Pos are counts /100; Mechanism/Loophole/Distinguishing/Fully are % of the 100 guilty)")
