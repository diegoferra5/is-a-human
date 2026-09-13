"""Step 0: is the agent's script the same for everyone, and fair between groups?

0a: does every call hit the same script beats, in the same order?
0b: does each beat fire at the same rate for human and synthetic callers?

Reads transcripts/ + the manifest. Train split only. Prints two tables.
"""
import json, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS = ROOT / "transcripts"
MANIFEST = ROOT / "dataset" / "manifest.csv"
AGENT_CH = 1

# The script beats, in the order we believe the agent runs them.
BEATS = [
    ("greeting",      r"gracias por llamar"),
    ("ask_name",      r"su nombre|con qui[eé]n tengo"),
    ("ask_ref",       r"n[uú]mero de referencia"),
    ("interrupt",     r"que (?:la|lo|le) interrump"),
    ("dilemma",       r"n[oó]mina plus|cr[eé]dito verde"),
    ("confirm_ref",   r"confirm\w*\s+(?:nuevamente\s+)?(?:su|el)\s+n[uú]mero|me confirma"),
    ("is_correct",    r"es correcto"),
    ("folio",         r"su folio es|folio \d"),
]

def agent_turns(call_id):
    p = TRANSCRIPTS / f"{call_id}.json"
    if not p.exists():
        return None
    return [t for t in json.loads(p.read_text())["turns"] if t["channel"] == AGENT_CH]

def beat_hits(turns):
    """First timestamp at which each beat fires, or None."""
    out = {}
    for name, pat in BEATS:
        t0 = next((t["start"] for t in turns if re.search(pat, t["text"], re.I)), None)
        out[name] = t0
    return out

def main():
    man = pd.read_csv(MANIFEST)
    man = man[man.split == "train"]
    rows = []
    for _, r in man.iterrows():
        turns = agent_turns(r.anon_id)
        if turns is None:
            continue
        h = beat_hits(turns)
        h.update(call=r.anon_id, label=r.label, n_agent_turns=len(turns))
        rows.append(h)
    df = pd.DataFrame(rows)
    names = [n for n, _ in BEATS]

    print(f"train calls with transcripts: {len(df)}\n")

    # ---- 0b: fire rate by label -----------------------------------------
    print("0b  DOES EACH BEAT FIRE EQUALLY FOR BOTH GROUPS?")
    print(f"{'beat':<14}{'human':>8}{'synth':>8}{'gap':>8}")
    for n in names:
        fired = df[n].notna()
        h = 100 * fired[df.label == "human"].mean()
        s = 100 * fired[df.label == "synthetic"].mean()
        flag = "  <-- SKEWED" if abs(h - s) > 10 else ""
        print(f"{n:<14}{h:7.1f}%{s:7.1f}%{h-s:+7.1f}{flag}")

    # ---- 0a: same order? -------------------------------------------------
    print("\n0a  DO THE BEATS ALWAYS COME IN THE SAME ORDER?")
    seqs = []
    for _, r in df.iterrows():
        fired = [(r[n], n) for n in names if pd.notna(r[n])]
        seqs.append(" > ".join(n for _, n in sorted(fired)))
    vc = pd.Series(seqs).value_counts()
    print(f"{len(vc)} distinct orderings across {len(df)} calls")
    for seq, c in vc.head(6).items():
        print(f"{c:4d} ({100*c/len(df):4.1f}%)  {seq}")

    # pairwise order consistency: when both beats fire, how often is a before b?
    print("\n    pairwise order consistency (both fired):")
    bad = 0
    for i, a in enumerate(names):
        for b in names[i+1:]:
            both = df[df[a].notna() & df[b].notna()]
            if len(both) < 20:
                continue
            pct = 100 * (both[a] < both[b]).mean()
            if pct < 95:
                print(f"      {a:>12} before {b:<12} {pct:5.1f}%  of {len(both)}")
                bad += 1
    if not bad:
        print("      all pairs >=95% consistent")

    df.to_csv(ROOT / "analysis" / "step0_beats.csv", index=False)
    print(f"\nwrote analysis/step0_beats.csv")

if __name__ == "__main__":
    main()
