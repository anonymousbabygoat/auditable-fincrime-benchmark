# DSPy APO - Act 1: automatic prompt optimization for auditable MARKET-ABUSE explanations (frozen model).
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
THREADS     = int(os.environ.get("APO_THREADS", "8"))              # parallel threads; lower to stay under provider concurrency caps
random.seed(13)

BUNDLE = json.load(open(os.environ.get("APO_BUNDLE", "apo_bundle_market.json"), encoding="utf-8"))
RUBRICS = json.load(open(os.environ.get("APO_RUBRICS", "apo_rubrics_market.json"), encoding="utf-8"))

# ================= DATA =================
def case_text(it):
    return it.get("scenario", "")

import dspy
examples = []
for it in BUNDLE["items"]:
    gold = "REPORT" if it["instance"] == "violation" else "CLEAR"
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
class AuditableMarketAbuse(dspy.Signature):
    """Decide whether a trading account's activity should be REPORTED to trade surveillance / compliance as
    possible market abuse (manipulation or insider dealing), or can be CLEARED as legitimate trading. Judge by
    the substance and PURPOSE of the market-conduct rules, not merely whether trading looks ordinary. Give a
    complete, auditable justification."""
    case          = dspy.InputField(desc="the client mandate and the (unlabelled) trading blotter")
    decision      = dspy.OutputField(desc="REPORT or CLEAR")
    typology      = dspy.OutputField(desc="the specific market-abuse method, or NONE if legitimate")
    mechanism     = dspy.OutputField(desc="the specific trading move the account is using")
    breach        = dspy.OutputField(desc="which market-conduct rule is engaged: how the conduct creates a false/misleading appearance, or misuses MNPI / evades a disclosure duty")
    distinguishing = dspy.OutputField(desc="what a genuine, legitimate trader would show here that this account does NOT")

# MIPROv2 optimizes this program's instruction (+ few-shot demos). Swap to dspy.ChainOfThought for a CoT variant.
program = dspy.Predict(AuditableMarketAbuse)
if os.environ.get("APO_WEAK_SEED","0") == "1":
    try:
        program.signature = program.signature.with_instructions(
            "Decide whether the trading account's activity should be REPORTED to trade surveillance as possible market abuse or can be CLEARED, and briefly justify.")
        print("[weak seed] starting from a minimal instruction so APO has room to climb.")
    except Exception as _e:
        print("weak-seed override failed:", _e)


# ================= JUDGE (cheap, used inside the metric) =================
from openai import OpenAI
_judge = OpenAI(base_url=JUDGE_BASE, api_key=JUDGE_KEY)
JSYS = ("You are a strict Singapore market-surveillance examiner. Given the correct rubric and an analyst's "
        "answer, judge ONE element only. A vague or generic statement does NOT earn it. Output ONLY 1 or 0.")
LAB = {"mech": "MECHANISM - the specific market-abuse method (spoofing, layering, wash trading, marking the close), not a generic label",
       "breach": "BREACH - why the conduct breaches market-conduct rules: it creates a FALSE or MISLEADING appearance of the market, OR misuses material non-public information / evades a disclosure obligation",
       "distinguishing": "DISTINGUISHING - what a legitimate look-alike (registered market maker, hedger, arbitrageur, genuine benchmarked order) would show that this account does NOT"}
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
    rep = any(w in tl for w in ("report", "flag", "suspicious", "escalat", "restrict", "str", "sar"))
    clr = any(w in tl for w in ("clear", "proceed", "legitimate", "not suspicious", "no action", "no concern"))
    if rep and not clr: return "REPORT"
    if clr and not rep: return "CLEAR"
    return "REPORT" if rep else ("CLEAR" if clr else "UNCLEAR")

# ================= METRIC (composite: decision + explanation, false-positive = worst) =================
def metric(example, pred, trace=None):
    dec  = parse_decision(getattr(pred, "decision", "") or "")
    # innocent twin: correct = CLEAR; a false positive (wrongly REPORT) is the primary failure -> 0
    if example.instance == "benign_twin":
        return 1.0 if dec == "CLEAR" else 0.0
    # violation: must REPORT, else the abuse was missed
    if dec != "REPORT":
        return 0.0
    if not USE_EXPL:
        return 1.0
    ans = (f"MECHANISM: {getattr(pred,'mechanism','')}  BREACH: {getattr(pred,'breach','')}  "
           f"DISTINGUISHING: {getattr(pred,'distinguishing','')}")
    m = judge_elem(example.rubric, "mech", ans)
    b = judge_elem(example.rubric, "breach", ans)
    d = judge_elem(example.rubric, "distinguishing", ans)
    if os.environ.get("APO_METRIC","strict") == "strict":
        return 1.0 if (m and b and d) else 0.0   # FULLY-CORRECT: all 3 explanation elements (headroom)
    return 0.4 + 0.6 * ((m + b + d) / 3.0)        # composite (softer)

# ================= OPTIMIZE =================
_lm_kw = dict(api_key=TASK_KEY, max_tokens=8000, temperature=0.0)
if TASK_BASE: _lm_kw["api_base"] = TASK_BASE
dspy.configure(lm=dspy.LM(TASK_MODEL, **_lm_kw))
from dspy.teleprompt import MIPROv2
print(f"\nOptimizing with MIPROv2 (auto={AUTO}) on {TASK_MODEL}; loop-judge={JUDGE_MODEL}; explanation-in-metric={USE_EXPL}")
_mipro_kw = dict(metric=metric, auto=AUTO, num_threads=THREADS)
if os.environ.get("APO_NO_DEMOS","0") == "1":
    _mipro_kw.update(max_bootstrapped_demos=0, max_labeled_demos=0)   # instruction-only: NO worked examples
    print("[instruction-only] few-shot demos disabled - APO may improve ONLY by rewording the instruction.")
tp = MIPROv2(**_mipro_kw)
optimized = tp.compile(program, trainset=train, valset=val, requires_permission_to_run=False)

# ================= EVALUATE (held-out test) =================
optimized.save("dspy_apo_market_optimized.json")
print("\nSaved optimized program EARLY -> ~/Desktop/dspy_apo_market_optimized.json")
from dspy.evaluate import Evaluate
ev = Evaluate(devset=test, metric=metric, num_threads=THREADS, display_progress=True)
print("\n=== BASELINE (unoptimized prompt) on held-out test ===")
try:
    base_score = ev(program)
    print("\n=== OPTIMIZED (MIPROv2) on held-out test ===")
    opt_score = ev(optimized)
    print(f"\nBASELINE test score : {base_score}")
    print(f"OPTIMIZED test score: {opt_score}")
except Exception as _e:
    print(f"\n[held-out eval interrupted: {type(_e).__name__}: {str(_e)[:120]}]")
    print("(optimized program was already saved above; lower APO_THREADS and re-run for clean held-out numbers)")

# ================= THE MONEY OUTPUT: the discovered instruction =================
print("\n================= OPTIMIZED PROMPT (does it rediscover 'innocent-first'?) =================")
try:
    for p in optimized.predictors():
        print(p.signature.instructions)
        print("---- few-shot demos:", len(getattr(p, "demos", []) or []))
except Exception as e:
    print("(inspect optimized program manually):", e)
optimized.save("dspy_apo_market_optimized.json")
print("\nSaved optimized program -> ~/Desktop/dspy_apo_market_optimized.json")
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
                ans = ((f"MECHANISM: {getattr(pred,'mechanism','')}  BREACH: {getattr(pred,'breach','')}  "
                        f"DISTINGUISHING: {getattr(pred,'distinguishing','')}") if pred else "")
                m = judge_elem(ex.rubric, "mech", ans); l = judge_elem(ex.rubric, "breach", ans); d = judge_elem(ex.rubric, "distinguishing", ans)
                mech += m; loop += l; dist += d
                if dec == "REPORT" and m and l and d: full += 1
            else:
                nT += 1
                if dec == "REPORT": Trep += 1
                elif dec == "UNCLEAR": Tunc += 1
        pc = lambda x: f"{round(100*x/max(nV,1))}%"
        print(f"    Detection {Vrep}/{nV}" + (f" ({Vunc} empty)" if Vunc else "")
              + f" | False-Pos {Trep}/{nT}" + (f" ({Tunc} empty)" if Tunc else "")
              + f" | Mechanism {pc(mech)} | Breach {pc(loop)} | Distinguishing {pc(dist)} | Fully {pc(full)}")
    print("\n================= 6-COLUMN REPORT (full 100 guilty + 100 twins, same grader/columns as ablation) =================")
    print("  WEAK-SEED (APO baseline):"); eval_arm(program)
    print("  APO-OPTIMIZED:"); eval_arm(optimized)
    print("  (Detection & False-Pos are counts /100; Mechanism/Breach/Distinguishing/Fully are % of the 100 guilty)")
