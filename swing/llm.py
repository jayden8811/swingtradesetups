"""Three LLM judgment layers on top of the rules (docs/STRATEGY.md).

L1  four specialist analysts per candidate (bounded +/-10 total)
L2  bear -> bull -> referee adversarial review (penalty 0..-15)
L3  chief judge, 3 independent runs, majority vote (+/-5, pick or NO TRADE)

LLMs never set prices. Their output is schema-constrained and every evidence item must cite a data field."""
import base64
import io
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config as C

VETO_REASONS = ["none", "binary_event_in_window", "offering_or_dilution",
                "fraud_delisting_going_concern", "merger_arb"]


def enabled():
    return bool(os.environ.get("ANTHROPIC_API_KEY")) and os.environ.get("SWING_LLM", "on") != "off"


_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(max_retries=4)
    return _client


def _obj(props, required=None):
    return {"type": "object", "properties": props, "required": required or list(props), "additionalProperties": False}


S_STR, S_INT, S_NUM, S_BOOL = {"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "boolean"}

L1_SCHEMA = _obj({
    "stance": {"type": "integer", "enum": [-2, -1, 0, 1, 2]},
    "confidence": S_NUM,
    "evidence": {"type": "array", "items": _obj({"claim": S_STR, "source_field": S_STR})},
    "red_flags": {"type": "array", "items": _obj({"flag": S_STR, "severity": {"type": "string", "enum": ["low", "med", "high"]}})},
    "hard_veto": S_BOOL,
    "veto_reason": {"type": "string", "enum": VETO_REASONS},
    "summary": S_STR,
})
BEAR_SCHEMA = _obj({"objections": {"type": "array", "items": _obj(
    {"objection": S_STR, "data": S_STR, "severity": {"type": "integer", "enum": [1, 2, 3]}})}})
BULL_SCHEMA = _obj({"responses": {"type": "array", "items": _obj({"objection_index": S_INT, "response": S_STR})}})
REF_SCHEMA = _obj({"rulings": {"type": "array", "items": _obj({
    "objection_index": S_INT, "verdict": {"type": "string", "enum": ["refuted", "partially_refuted", "stands"]},
    "severity": {"type": "integer", "enum": [1, 2, 3]}, "note": S_STR})}})
JUDGE_SCHEMA = _obj({
    "consistency_issues": {"type": "array", "items": _obj({"symbol": S_STR, "issue": S_STR})},
    "adjustments": {"type": "array", "items": _obj({"symbol": S_STR, "adjustment": S_INT, "reason": S_STR})},
    "picks": {"type": "array", "items": S_STR},
    "no_trade_reason": S_STR,
    "cards": {"type": "array", "items": _obj({
        "symbol": S_STR, "thesis": S_STR, "key_points": {"type": "array", "items": S_STR},
        "risks": {"type": "array", "items": S_STR}, "invalidation": S_STR})},
})

COMMON = ("You are part of a swing-trading research system for long stock positions held 3-30 trading days. "
          "Deterministic code has already found the setup and computed every number (entry, stop, targets, "
          "probabilities). You judge only what rules cannot. Use ONLY the data provided; never invent numbers. "
          "Cite the JSON field (dotted path, e.g. plan.rr or fundamentals.rev_yoy, or 'chart' / 'news') behind "
          "every evidence item. Be balanced: argue from evidence, not optimism. Keep text terse.")

AGENTS = {
    "technical": ("Technical analyst. Study the chart image and price fields. Is this a genuine, constructive "
                  "setup or a sloppy/broken one (wide and loose base, failed breakouts, heavy overhead supply, "
                  "climactic extension)? Is the stop below a logical level and is T1 reachable?",
                  ["symbol", "setup", "setup_metrics", "plan", "factors", "price_context"]),
    "fundamental": ("Fundamental & earnings analyst. Judge the quality of growth and the latest earnings: "
                    "beat quality, guidance direction, margins, leverage, dilution risk. Is the next earnings "
                    "date inside the holding window?",
                    ["symbol", "setup", "fundamentals", "plan"]),
    "news": ("News & catalyst analyst. From headlines: is there a real catalyst or is the move news-less? "
             "Any binary event inside ~30 trading days (FDA date, trial, ruling, vote, offering, merger)? "
             "Is the news already priced in? Veto only for the listed veto reasons.",
             ["symbol", "setup", "news", "fundamentals.next_earnings", "plan"]),
    "positioning": ("Positioning & macro analyst. Does the sector, market regime and positioning (insiders, "
                    "short interest) help or fight this long? Are insider buys meaningful vs. their size?",
                    ["symbol", "setup", "regime", "fundamentals.sector", "fundamentals.insider_recent",
                     "fundamentals.insider_buyers", "fundamentals.short_pct_float", "fundamentals.short_ratio_days",
                     "factors"]),
}


def _call(system, content, schema, effort="medium", max_tokens=16000):
    kwargs = dict(model=C.LLM_MODEL, max_tokens=max_tokens, system=system,
                  messages=[{"role": "user", "content": content}],
                  output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}})
    client = _get_client()
    try:
        resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    except Exception as e:  # fallbacks unsupported on this platform/account: plain request
        if "fallback" not in str(e).lower():
            raise
        resp = client.messages.create(**kwargs)
    if resp.stop_reason in ("refusal", "max_tokens"):
        return None
    text = next((b.text for b in resp.content if b.type == "text"), None)
    return json.loads(text) if text else None


def _pick(d, path):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _subset(digest, keys):
    out = {}
    for k in keys:
        v = _pick(digest, k)
        if v is not None:
            out[k] = v
    return out


def _valid_evidence(ev, view):
    roots = {k.split(".")[0] for k in view} | {"chart", "news"}
    return [e for e in ev if e.get("source_field", "").replace("[", ".").split(".")[0] in roots]


def chart_png(df, plan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.tail(250)
    fig, (ax, av) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(d.index, d["close"], lw=1.2, color="black", label="close")
    for n, col in ((21, "tab:blue"), (50, "tab:orange"), (200, "tab:red")):
        ax.plot(d.index, df["close"].rolling(n).mean().tail(250), lw=0.8, color=col, label=f"MA{n}")
    for k, col in (("entry", "green"), ("stop", "red"), ("t1", "purple")):
        ax.axhline(plan[k], ls="--", lw=0.8, color=col, label=k)
    ax.legend(loc="upper left", fontsize=7)
    av.bar(d.index, d["volume"], color="gray", width=1)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    return base64.standard_b64encode(buf.getvalue()).decode()


# ---------------- Layer 1 ----------------

def _l1_one(cand, agent):
    task, keys = AGENTS[agent]
    view = _subset(cand["digest"], keys)
    content = []
    if agent == "technical" and cand.get("chart"):
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": cand["chart"]}})
    content.append({"type": "text", "text": f"{task}\n\nDATA:\n{json.dumps(view, default=str)}\n\n"
                    "Return stance -2..+2 for going long now, confidence 0-1, cited evidence, red flags, "
                    "and hard_veto only for: " + ", ".join(VETO_REASONS[1:]) + "."})
    out = _call(COMMON, content, L1_SCHEMA, effort="medium")
    if not out:
        return agent, None
    out["evidence"] = _valid_evidence(out.get("evidence", []), view)
    out["confidence"] = float(np.clip(out.get("confidence", 0), 0, 1))
    if out.get("veto_reason") == "none":
        out["hard_veto"] = False
    return agent, out


def layer1(cands, progress=None):
    jobs = [(c, a) for c in cands for a in AGENTS]
    with ThreadPoolExecutor(C.LLM_WORKERS) as ex:
        results = list(ex.map(lambda j: (j[0]["symbol"], *_l1_one(*j)), jobs))
    for c in cands:
        c["l1"] = {a: r for s, a, r in results if s == c["symbol"]}
        adj = sum(np.clip(r["stance"] * r["confidence"] * 2.5, -5, 5)
                  for r in c["l1"].values() if r and r.get("evidence"))
        c["l1_adj"] = float(np.clip(adj, -10, 10))
        vetoes = [f"{a}: {r['veto_reason']}" for a, r in c["l1"].items() if r and r.get("hard_veto")]
        c["veto"] = vetoes or None
    if progress:
        progress("L1 specialists done")


# ---------------- Layer 2 ----------------

def _l2_one(cand):
    data = json.dumps({"digest": cand["digest"], "layer1": cand["l1"]}, default=str)
    bear = _call(COMMON, "You are the BEAR. Give the 3 strongest falsifiable objections to buying this now, "
                 f"each tied to specific data, with severity 1-3.\n\n{data}", BEAR_SCHEMA)
    if not bear or not bear.get("objections"):
        return None
    obj = bear["objections"][:3]
    numbered = json.dumps(list(enumerate(obj)))
    bull = _call(COMMON, "You are the BULL. Answer each numbered objection using only the data. Concede "
                 f"when the data supports the objection.\n\nOBJECTIONS: {numbered}\n\n{data}", BULL_SCHEMA)
    ref = _call(COMMON, "You are a neutral REFEREE. For each numbered objection decide refuted / "
                "partially_refuted / stands based on the data, and set its severity 1-3.\n\n"
                f"OBJECTIONS: {numbered}\nBULL: {json.dumps(bull)}\n\n{data}", REF_SCHEMA, effort="high")
    if not ref:
        return None
    weight = {"stands": 3, "partially_refuted": 1.5, "refuted": 0}
    pen = -sum(r["severity"] * weight[r["verdict"]] for r in ref["rulings"])
    return {"objections": obj, "bull": bull, "rulings": ref["rulings"], "penalty": float(max(pen, -15))}


def layer2(cands, progress=None):
    with ThreadPoolExecutor(C.LLM_WORKERS) as ex:
        res = list(ex.map(_l2_one, cands))
    for c, r in zip(cands, res):
        c["l2"] = r
        c["l2_penalty"] = r["penalty"] if r else 0.0
    if progress:
        progress("L2 bull vs bear done")


# ---------------- Layer 3 ----------------

def layer3(cands, regime, max_picks, progress=None):
    packet = [{"symbol": c["symbol"], "score_so_far": round(c["score_pre_l3"], 1), "digest": c["digest"],
               "layer1_summaries": {a: (r or {}).get("summary") for a, r in c["l1"].items()},
               "layer2": c.get("l2")} for c in cands]
    prompt = (f"You are the CHIEF JUDGE / portfolio manager. Market regime: {json.dumps(regime, default=str)}. "
              f"You may pick at most {max_picks} (fewer or none is fine; at most one per sector). "
              "1) Check each trade plan for internal consistency (stop below support, T1 below obvious "
              "resistance, timing plausible for its ATR) and list issues. 2) Compare candidates side by side and "
              "give each an integer adjustment -5..+5. 3) Pick the best, or none with a no_trade_reason. "
              "4) For each pick write a card: thesis, 3 key data points, risks (from layer2), invalidation.\n\n"
              f"CANDIDATES:\n{json.dumps(packet, default=str)}")
    with ThreadPoolExecutor(C.L3_RUNS) as ex:
        runs = [r for r in ex.map(lambda _: _call(COMMON, prompt, JUDGE_SCHEMA, effort="high"), range(C.L3_RUNS)) if r]
    votes = Counter(s for r in runs for s in set(r.get("picks", [])[:max_picks]))
    need = 2 if len(runs) >= 2 else 1
    for c in cands:
        adjs = [a["adjustment"] for r in runs for a in r.get("adjustments", []) if a["symbol"] == c["symbol"]]
        c["l3_adj"] = float(np.clip(np.mean(adjs), -5, 5)) if adjs else 0.0
        c["l3_votes"] = votes.get(c["symbol"], 0)
        c["l3_pick"] = c["l3_votes"] >= need
        c["l3_issues"] = [i["issue"] for r in runs for i in r.get("consistency_issues", []) if i["symbol"] == c["symbol"]]
        c["card"] = next((card for r in runs for card in r.get("cards", []) if card["symbol"] == c["symbol"]), None)
    reasons = [r.get("no_trade_reason") for r in runs if r.get("no_trade_reason")]
    if progress:
        progress("L3 chief judge done")
    return {"runs": len(runs), "votes": dict(votes), "no_trade_reason": reasons[0] if reasons else None}
