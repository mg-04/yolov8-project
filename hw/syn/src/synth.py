#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Synthesise every sentry_check variant and tabulate area.

Out-of-context (-noiopad): I/O buffers would otherwise dominate a block this
small and are not part of its cost once it sits inside a datapath.

The two questions this answers:

  1. What does an 8-bit exponent compare cost against two fp32 compares?
     That ratio is the entire argument for the exponent bound, and until now
     it has been asserted analytically.

  2. What does Algorithm 1's candidate search cost? The software result showed
     the search is redundant -- the first candidate always wins -- so REPAIR=0
     (substitute zero) and REPAIR=1 (the published search) bracket the saving.

Usage:  ./synth.py [--json out.json]
"""
import json
import os
import re
import subprocess
import sys

HW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RTL = os.path.join(HW, "rtl", "sentry_check.v")
FORMATS = [("fp32", 8, 23), ("fp16", 5, 10)]
MODES = [(0, "exp 1-sided", "1 B/unit"), (1, "exp 2-sided", "2 B/unit"),
         (2, "signed", "8 B/unit"), (3, "signed + floor", "9 B/unit")]
REPAIRS = [(0, "zero"), (1, "Alg-1 search")]


def run(exp_w, mant_w, mode, repair, backend):
    script = (f"read_verilog {RTL}; "
              f"hierarchy -top sentry_check -chparam MODE {mode} "
              f"-chparam EXP_W {exp_w} -chparam MANT_W {mant_w} "
              f"-chparam REPAIR {repair}; {backend}; stat")
    out = subprocess.run(["yosys", "-p", script], capture_output=True,
                         text=True).stdout
    cells = {}
    for line in out.splitlines():
        m = re.match(r"\s+([\$\w]+)\s+(\d+)\s*$", line)
        if m:
            cells[m.group(1)] = int(m.group(2))
    return cells


def synth(exp_w, mant_w, mode, repair):
    """Generic gate count is the headline; Xilinx LUTs are secondary.

    ABC's LUT packing is not monotonic at this scale -- mapped to xc7 it makes
    `signed` come out LARGER than `signed + floor`, which does strictly more
    work, while the generic gate count orders them correctly. So the relative
    comparison rests on gates, and absolute LUT or um^2 figures need Vivado or
    a standard-cell flow rather than this script.
    """
    g = run(exp_w, mant_w, mode, repair, "synth -flatten")
    gates = sum(v for k, v in g.items() if k.startswith("$_"))
    x = run(exp_w, mant_w, mode, repair,
            "synth_xilinx -family xc7 -flatten -noiopad")
    luts = sum(v for k, v in x.items() if k.startswith("LUT"))
    return gates, luts, x.get("CARRY4", 0), g


def main():
    rows = []
    print(f"{'format':<8}{'mode':<16}{'meta':<10}{'repair':<14}"
          f"{'gates':>7}{'xc7 LUT':>9}")
    for fname, exp_w, mant_w in FORMATS:
        for mode, mname, meta in MODES:
            for repair, rname in REPAIRS:
                gates, luts, carry, det = synth(exp_w, mant_w, mode, repair)
                rows.append(dict(format=fname, mode=mname, meta=meta,
                                 repair=rname, gates=gates, lut=luts,
                                 carry4=carry, detail=det))
                print(f"{fname:<8}{mname:<16}{meta:<10}{rname:<14}"
                      f"{gates:>7}{luts:>9}")
        print()

    base = {r["format"]: r for r in rows
            if r["mode"] == "signed" and r["repair"] == "zero"}
    print("exponent vs signed, detection only (REPAIR=zero):")
    for fname, _, _ in FORMATS:
        e = next(r for r in rows if r["format"] == fname
                 and r["mode"] == "exp 1-sided" and r["repair"] == "zero")
        s = base[fname]
        print(f"  {fname}: {s['gates']} gates signed vs {e['gates']} exp "
              f"-> {s['gates'] / e['gates']:.1f}x")

    print("\ncost of Algorithm 1's search (REPAIR=search minus REPAIR=zero):")
    for fname, _, _ in FORMATS:
        for mode, mname, _ in MODES:
            z = next(r for r in rows if r["format"] == fname
                     and r["mode"] == mname and r["repair"] == "zero")
            s = next(r for r in rows if r["format"] == fname
                     and r["mode"] == mname and r["repair"] == "Alg-1 search")
            print(f"  {fname} {mname:<16} {z['gates']:>4} -> {s['gates']:>4}"
                  f" gates   (+{s['gates'] - z['gates']})")

    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
        with open(path, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\n-> {path}")


if __name__ == "__main__":
    main()
