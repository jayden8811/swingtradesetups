"""python -m swing [serve|scan|backtest]"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(prog="swing")
    ap.add_argument("cmd", choices=["serve", "scan", "backtest"], nargs="?", default="serve")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    if a.cmd == "backtest":
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
