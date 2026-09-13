"""Step 1 — cut the script moments out of every call into a blind review folder.

Produces analysis/blind/, a self-contained folder for an independent analysis session:
  BRIEF.md              what to do, with no hypotheses of ours in it
  moments/<name>.md     group A and group B answers to the SAME agent question
  free/<name>.md        random excerpts tied to no probe, so unknown patterns can surface

Group A / group B are the two classes with the labels withheld. The mapping lives in
analysis/blind_key.json, outside the folder, so the reviewer cannot see it.

Sponsor name is scrubbed from every excerpt.
"""
import json, re, random, shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

BRIEF = """# Blind comparison task

## What this is

Excerpts from recorded Spanish-language customer service phone calls. Every call has two
speakers:

- **AGENT** - an automated assistant. It runs the same script on every call, so its side is
  near-identical everywhere.
- **CALLER** - the person or system that dialled in. **This is the one that differs.**

The calls fall into two groups, **A** and **B**. Which group a call belongs to is a property
of the caller, not of the agent. You are not told what distinguishes them, and you should
not try to guess what the categories "are". Guessing the category and then arguing for it is
the failure mode here; several earlier attempts went wrong exactly that way.

## Your task

Find, as exhaustively as you can, **what differs between group A callers and group B
callers**, and back every claim with quoted examples from both sides.

**`calls/A/` and `calls/B/` are the real material** - complete transcripts, equal counts per
group, nothing removed. Around 200k tokens in total, so you can read a great deal of it
directly. Start here. Whole-call patterns - who drives the conversation, how it ends,
whether the caller stays on topic, what happens after a misunderstanding - only exist at
this level.

`moments/` and `free/` are a convenience index built by someone else's regexes, slicing
those same calls at points in the agent's script so that replies to the same question sit
side by side. Useful for comparing one specific point quickly. **Those slices encode
assumptions about what matters - do not let them bound your search.** If the index and the
raw calls disagree, trust the raw calls.

File counts are equal for both groups by construction. They tell you nothing.

## How to work

1. Read whole calls before writing anything. Then read more.
2. Write what you notice about group A and about group B **separately**, then compare.
3. Prefer specific, checkable observations over impressions.
   - weak: "group B sounds more robotic"
   - strong: "group B replies repeat the agent's own nouns back; group A replies use
     pronouns instead - B: 'si, sobre mi cuenta de nomina', A: 'si, esa'"
4. Count roughly when you can. "9 of 25 A, 2 of 25 B" beats "more often".
5. Include differences you find unimpressive or that appear only a few times. Small
   consistent effects are wanted. Do not filter for what sounds clever.
6. Actively look for **absences** - something one group does that the other never does.
7. If a moment shows no difference, say so plainly. Negative results are useful and trusted.

## Important: these are automatic transcripts

Every excerpt came out of an automatic speech-to-text system, and one side of the call is
noticeably noisier audio than the other. Some apparent differences are transcription
artifacts, not speaking differences. Word counts, exact spellings, digit sequences and
punctuation are all unreliable.

Mark every finding as one of:

- **ASR-SAFE** - survives imperfect transcription (structure of the reply, whether it
  answers the question that was asked, whether it introduces new information, how it
  handles being cut off)
- **ASR-FRAGILE** - depends on exact words, spellings, counts or digits being right

Report fragile findings anyway, just labelled.

## Output

Write `findings.md`:

```
## <short name of the pattern>
**Where:** which calls or which moment
**Group A:** what A does, with 2-3 quotes
**Group B:** what B does, with 2-3 quotes
**Rough frequency:** e.g. 11/25 A vs 3/25 B
**ASR:** SAFE | FRAGILE
**Note:** anything that could explain this other than a real difference between callers
```

One section per pattern, ordered by your confidence. Aim for breadth - twenty mediocre
observations are more useful here than three polished ones.

Do not write code. Read and report.
"""

ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS = ROOT / "transcripts"
OUT = ROOT / "analysis" / "blind"
KEY = ROOT / "analysis" / "blind_key.json"
SEED = 20260912
PER_GROUP = 25          # excerpts per group per moment
CALLER, AGENT = 0, 1

SCRUB = re.compile(r"\w*alt[uú]\w*", re.I)   # catches ASR manglings too

# Moments are found by what the AGENT says. Deliberately descriptive names only —
# nothing that hints at what we expect to find.
MOMENTS = {
    "opening":        r"gracias por llamar",
    "asks_name":      r"su nombre|con qui[eé]n tengo",
    "asks_reference": r"n[uú]mero de referencia",
    "interrupts":     r"que (?:la|lo|le) interrump",
    "two_options":    r"n[oó]mina\s*plus.{0,40}cr[eé]dito\s*verde|cr[eé]dito\s*verde.{0,40}n[oó]mina\s*plus",
    "repeats_number": r"me confirma|confirm\w*\s+(?:nuevamente\s+)?(?:su|el)\s+n[uú]mero|es correcto",
    "gives_folio":    r"su folio es|folio \d",
    "closing":        r"algo m[aá]s|que tenga buen|excelente d[ií]a|hasta luego",
}


def clean(s: str) -> str:
    return SCRUB.sub("[BANCO]", s).strip()


def turns_of(call_id: str):
    p = TRANSCRIPTS / f"{call_id}.json"
    if not p.exists():
        return None
    return sorted(json.loads(p.read_text())["turns"], key=lambda x: x["start"])


def excerpt(turns, i, n_before=1, n_after=2):
    """Agent turn at i, a little context before, and the caller's next n_after turns."""
    lines = []
    for t in turns[max(0, i - n_before):i]:
        who = "CALLER" if t["channel"] == CALLER else "AGENT "
        lines.append(f"  [{t['start']:6.1f}] {who} {clean(t['text'])}")
    t = turns[i]
    lines.append(f"* [{t['start']:6.1f}] AGENT  {clean(t['text'])}")
    taken = 0
    for t in turns[i + 1:]:
        who = "CALLER" if t["channel"] == CALLER else "AGENT "
        lines.append(f"  [{t['start']:6.1f}] {who} {clean(t['text'])}")
        if t["channel"] == CALLER:
            taken += 1
            if taken >= n_after:
                break
    return "\n".join(lines)


def main():
    rng = random.Random(SEED)
    man = pd.read_csv(ROOT / "dataset" / "manifest.csv")
    man = man[man.split == "train"]

    # blind the labels: coin flip decides which class becomes A
    labels = sorted(man.label.unique())
    if rng.random() < 0.5:
        labels = labels[::-1]
    to_group = {labels[0]: "A", labels[1]: "B"}

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "moments").mkdir(parents=True)
    (OUT / "free").mkdir(parents=True)

    # ---- full raw transcripts, one file per call, foldered by group ------
    # This is the primary material. The moment files below are only an index into it.
    # Balanced to equal counts: an uneven file count would give the classes away.
    by_group = defaultdict(list)
    for _, r in man.iterrows():
        by_group[to_group[r.label]].append(r.anon_id)
    n_raw = min(len(by_group["A"]), len(by_group["B"]))
    chosen = [(g, cid) for g in ("A", "B") for cid in rng.sample(by_group[g], n_raw)]

    for g, cid in chosen:
        turns = turns_of(cid)
        if not turns:
            continue
        d = OUT / "calls" / g
        d.mkdir(parents=True, exist_ok=True)
        body = "\n".join(
            f"[{t['start']:6.1f}] {'CALLER' if t['channel'] == CALLER else 'AGENT '} "
            f"{clean(t['text'])}" for t in turns)
        (d / f"{cid}.txt").write_text(body)

    # ---- collect every moment in every call ------------------------------
    found = defaultdict(lambda: defaultdict(list))   # moment -> group -> [(call, text)]
    free = defaultdict(list)                          # group -> [(call, text)]
    patterns = {k: re.compile(v, re.I) for k, v in MOMENTS.items()}

    for _, r in man.iterrows():
        turns = turns_of(r.anon_id)
        if not turns:
            continue
        g = to_group[r.label]
        used = set()
        for name, pat in patterns.items():
            for i, t in enumerate(turns):
                if t["channel"] == AGENT and pat.search(t["text"]):
                    found[name][g].append((r.anon_id, excerpt(turns, i)))
                    used.add(i)
                    break
        # one random agent turn that matched nothing, for unforeseen patterns
        spare = [i for i, t in enumerate(turns)
                 if t["channel"] == AGENT and i not in used and len(t["text"].split()) > 4]
        if spare:
            i = rng.choice(spare)
            free[g].append((r.anon_id, excerpt(turns, i)))

    # ---- write one file per moment ---------------------------------------
    counts = {}
    for name in MOMENTS:
        a, b = found[name]["A"], found[name]["B"]
        counts[name] = {"A_total": len(a), "B_total": len(b)}
        sa = rng.sample(a, min(PER_GROUP, len(a)))
        sb = rng.sample(b, min(PER_GROUP, len(b)))
        n = min(len(sa), len(sb))          # equal sizes: totals would leak the classes
        sa, sb = sa[:n], sb[:n]
        lines = [f"# Moment: {name}", "",
                 f"A random sample of {n} calls from each group, at the same point in the "
                 "conversation. Sample sizes are equal by construction and say nothing "
                 "about how common the moment is.", "",
                 "`*` marks the agent turn that defines the moment. Times are seconds.", ""]
        for g, s in (("A", sa), ("B", sb)):
            lines += [f"## GROUP {g}", ""]
            for cid, ex in s:
                lines += [f"### {cid}", "```", ex, "```", ""]
        (OUT / "moments" / f"{name}.md").write_text("\n".join(lines))

    nfree = min(40, len(free["A"]), len(free["B"]))
    fa = rng.sample(free["A"], nfree)
    fb = rng.sample(free["B"], nfree)
    lines = ["# Unstructured excerpts", "",
             "Random agent turns that matched none of the named moments, with the caller's "
             "reply. Here to catch patterns the named moments do not cover.", ""]
    for g, s in (("A", fa), ("B", fb)):
        lines += [f"## GROUP {g}", ""]
        for cid, ex in s:
            lines += [f"### {cid}", "```", ex, "```", ""]
    (OUT / "free" / "unstructured.md").write_text("\n".join(lines))

    KEY.write_text(json.dumps(
        {"group_to_label": {v: k for k, v in to_group.items()}, "seed": SEED,
         "per_group": PER_GROUP, "counts": counts}, indent=2))

    (OUT / "BRIEF.md").write_text(BRIEF)

    print(f"wrote {OUT}")
    for name, c in counts.items():
        print(f"  {name:<16} A={c['A_total']:3d}  B={c['B_total']:3d}")
    print(f"  {'unstructured':<16} A={len(free['A']):3d}  B={len(free['B']):3d}")
    print(f"\nkey (NOT in the blind folder): {KEY}")


if __name__ == "__main__":
    main()
