"""python -m swing [serve|scan|backtest|dips|research|active]"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(prog="swing")
    ap.add_argument("cmd", choices=["serve", "scan", "backtest", "dips", "research", "active"], nargs="?", default="serve")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    if a.cmd == "active":
        from . import active
        r = active.run()
        print(f"{'':20}{'full CAGR':>10}{'Sharpe':>8}{'maxDD':>8} | {'IS CAGR':>8} | {'OOS CAGR':>9}{'Sharpe':>8}{'maxDD':>8} | expo  trades/yr win  avg")
        for n, m in r["strategies"].items():
            f, i, o = m["full"], m["in_sample"], m["out_sample"]
            print(f"{n:20}{f['cagr']:>+10.1%}{f['sharpe']:>8.2f}{f['max_dd']:>8.0%} | {i['cagr']:>+8.1%} | {o['cagr']:>+9.1%}{o['sharpe']:>8.2f}{o['max_dd']:>8.0%} | "
                  f"{m['exposure']:.0%} {m['trades_per_year']:6.0f} " + (f"{m['win_rate']:.0%} {m['avg_trade']:+.2%}" if m['win_rate'] is not None else "") + (f"  passes={m['passes']} {m['checks']}" if "passes" in m else ""))
        print("applied:", r["applied"])
    elif a.cmd == "research":
        from . import momentum
        r = momentum.run()
        for n, m in r["strategies"].items():
            print(f"{n:16} CAGR {m['cagr']:+.1%}  Sharpe {m['sharpe']:.2f}  maxDD {m['max_dd']:.1%}  "
                  f"beat SPY 36m {m['win_36m']:.0%}  years {m['years_beat_spy']}/{m['n_years']}" + (f"  passes={m['passes']}" if "passes" in m else ""))
        print("applied:", r["applied"], "| live:", r["live"]["strategy"], len(r["live"]["holdings"]), "holdings")
    elif a.cmd == "dips":
        from . import dips
        dips.run()
    elif a.cmd == "backtest":
        from . import backtest
        backtest.run()
    elif a.cmd == "scan":
        from . import pipeline
        r = pipeline.scan()
        print(json.dumps({k: r[k] for k in ("asof", "regime", "universe", "n_setups")}, indent=1))
        for p in r["picks"]:
            print(f"🎯 {p['symbol']} {p['setup_name']} entry {p['entry']} stop {p['stop']} "
                  f"T1 {p['t1']} P(win) {p['p_win']:.0%} EV {p['ev_r']:+.2f}R ~{p['days']}d")
        if not r["picks"]:
            print("💤 no trade today")
    else:
        from . import server
        server.serve(a.port)


if __name__ == "__main__":
    main()
