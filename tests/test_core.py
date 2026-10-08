import json

import numpy as np
import pandas as pd
import pytest

from swing import config as C, llm, plan as P, scoring as S
from swing.indicators import atr, lin, rsi, trend_r2, windows
from swing.setups import features


def ohlc(close, spread=1.0, vol=1e6):
    c = pd.Series(close, index=pd.bdate_range("2020-01-01", periods=len(close)), dtype=float)
    return pd.DataFrame({"open": c, "high": c + spread / 2, "low": c - spread / 2, "close": c, "volume": vol})


def test_atr_constant_range():
    df = ohlc(np.full(50, 100.0), spread=2.0)
    assert atr(df).iloc[-1] == pytest.approx(2.0)


def test_rsi_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert rsi(up).iloc[-1] == pytest.approx(100, abs=1e-6) or np.isnan(rsi(up).iloc[-1])
    zig = pd.Series([1, 2] * 40, dtype=float)
    assert 40 < rsi(zig).iloc[-1] < 60


def test_trend_r2_perfect_line():
    s = pd.Series(np.exp(np.linspace(0, 1, 100)))
    assert trend_r2(s).iloc[-1] == pytest.approx(1.0, abs=1e-6)
    assert trend_r2(s[::-1].reset_index(drop=True)).iloc[-1] == pytest.approx(-1.0, abs=1e-6)


def test_windows_exclude_current_bar():
    w = windows(np.arange(10), 3)
    assert np.isnan(w[2]).all()
    assert list(w[5]) == [2, 3, 4]


def test_lin_both_directions():
    assert lin(0.5, 0, 1) == 50
    assert lin(0.1, 0.3, 0.1) == 100
    assert lin(5, 0, 1) == 100


def test_features_no_lookahead():
    rng = np.random.default_rng(0)
    close = 50 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 400)))
    df = ohlc(close)
    full = features(df, df["close"])
    cut = features(df.iloc[:300], df["close"].iloc[:300])
    cols = [c for c in full.columns if c != "C_gap_date"]
    pd.testing.assert_frame_equal(full[cols].iloc[:300], cut[cols], check_dtype=False)


def _plan(entry=100, stop=95, t1=110):
    return {"entry": entry, "stop": stop, "t1": t1, "R": entry - stop, "cost_r": 0.0, "rr": (t1 - entry) / (entry - stop)}


def test_simulate_win_loss_timeout_and_gap():
    o = np.array([100, 100, 104, 108, 109]); h = o + 3; l = o - 1; c = o
    assert P.simulate(o, h, l, c, 0, _plan(), max_hold=4)["outcome"] == "win"
    o2 = np.array([100, 100, 90, 90, 90]); assert P.simulate(o2, o2 + 1, o2 - 1, o2, 0, _plan(), 4)["r"] == pytest.approx(-2.0)
    flat = np.full(6, 100.0)
    res = P.simulate(flat, flat + 1, flat - 1, flat, 0, _plan(), max_hold=5)
    assert res["outcome"] == "timeout" and res["r"] == pytest.approx(0)
    both = np.array([100, 100]); assert P.simulate(both, both + 20, both - 20, both, 0, _plan(), 1)["outcome"] == "loss"


def test_probability_shrinks_to_baseline():
    rnd = {"n": 1000, "wins": 200, "win_rate": 0.2, "baseline": 0.25, "timeouts": 300, "timeout_r": 0.1}
    stats = {"cells": {"RANDOM": rnd, "A|80-100": {"n": 4, "wins": 4, "timeouts": 0, "timeout_r": 0, "days_median": 8, "days_iqr": [5, 12]}}}
    p = S.probability(stats, "A", "80-100", "Risk-On", rr=2.0)
    assert p["baseline"] == pytest.approx(1 / 3 * 0.8)    # random entry under the same rules
    assert p["baseline"] < p["p_win"] < 0.45              # 4/4 wins must not read as ~100%
    assert p["days_median"] == 8
    assert S.probability(None, "A", "80-100", "Risk-On", 3.0)["p_win"] == pytest.approx(0.25)


def test_expected_value():
    prob = {"p_win": 0.4, "p_timeout": 0.1, "timeout_r": 0.0}
    assert S.expected_value(prob, {"rr": 2.5, "cost_r": 0.05}) == pytest.approx(0.4 * 2.5 - 0.5 - 0.05)


class FakeResp:
    def __init__(self, payload):
        self.stop_reason = "end_turn"
        self.content = [type("B", (), {"type": "text", "text": json.dumps(payload)})()]


class FakeClient:
    """Answers each LLM layer with a canned, schema-shaped response."""
    def __init__(self):
        self.beta = self
        self.messages = self

    def create(self, **kw):
        schema = kw["output_config"]["format"]["schema"]["properties"]
        if "stance" in schema:
            return FakeResp({"stance": 2, "confidence": 1.0, "summary": "ok", "hard_veto": False, "veto_reason": "none",
                             "red_flags": [], "evidence": [{"claim": "x", "source_field": "plan.rr"},
                                                           {"claim": "made up", "source_field": "imaginary.field"}]})
        if "objections" in schema:
            return FakeResp({"objections": [{"objection": "extended", "data": "plan", "severity": 2}] * 3})
        if "responses" in schema:
            return FakeResp({"responses": []})
        if "rulings" in schema:
            return FakeResp({"rulings": [{"objection_index": i, "verdict": "stands", "severity": 2, "note": ""} for i in range(3)]})
        return FakeResp({"consistency_issues": [], "adjustments": [{"symbol": "XYZ", "adjustment": 9, "reason": ""}],
                         "picks": ["XYZ"], "no_trade_reason": "", "cards": [{"symbol": "XYZ", "thesis": "t", "key_points": ["a"], "risks": ["r"], "invalidation": "i"}]})


def test_llm_layers_are_bounded(monkeypatch):
    monkeypatch.setattr(llm, "_client", FakeClient())
    cand = {"symbol": "XYZ", "digest": {"symbol": "XYZ", "plan": {"rr": 2.5}, "setup": "A"}}
    llm.layer1([cand])
    assert cand["l1_adj"] == 10                        # 4 agents x +5 clipped to +10
    assert all(e["source_field"] != "imaginary.field" for r in cand["l1"].values() for e in r["evidence"])
    assert cand["l1"]["positioning"]["evidence"] == []   # plan.* is not in its data view
    llm.layer2([cand])
    assert cand["l2_penalty"] == -15                   # 3 x severity 2 x 3 = -18 clipped
    cand["score_pre_l3"] = 70
    llm.layer3([cand], {"state": "Risk-On"}, 1)
    assert cand["l3_adj"] == 5 and cand["l3_pick"] and cand["card"]["thesis"] == "t"


def test_dip_trades_one_position_per_stock():
    from swing.dips import _one_at_a_time
    t = [{"entry_date": "2026-01-02", "exit_date": "2026-01-06"}, None,
         {"entry_date": "2026-01-05", "exit_date": "2026-01-08"},     # overlaps -> skipped
         {"entry_date": "2026-01-07", "exit_date": "2026-01-09"}]
    assert [x["entry_date"] for x in _one_at_a_time(t)] == ["2026-01-02", "2026-01-07"]


def test_heatmap_rating_caps_and_floors():
    from swing.heatmap import rating
    base = {"D_rsi2": 50, "D_dip": 0.01, "close": 100, "sma200": 90, "D": False}
    assert rating({**base, "D_uptrend": False}, 99)[0] <= 45
    assert rating({**base, "D_uptrend": True, "D": True}, 1)[0] >= 75
