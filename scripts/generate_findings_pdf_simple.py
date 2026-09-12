#!/usr/bin/env python3
"""Generate a plain-language PDF summary for non-technical readers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CHART_DIR = REPORTS / "_charts"
OUTPUT = REPORTS / "is-a-human-findings-simple.pdf"


def _setup_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BigTitle",
            parent=styles["Title"],
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Tagline",
            parent=styles["Normal"],
            fontSize=13,
            leading=17,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontSize=15,
            leading=19,
            spaceBefore=16,
            spaceAfter=10,
            textColor=colors.HexColor("#1a365d"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["Normal"],
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SimpleBullet",
            parent=styles["Normal"],
            fontSize=11,
            leading=15,
            leftIndent=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Highlight",
            parent=styles["Normal"],
            fontSize=12,
            leading=17,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1a365d"),
            spaceBefore=8,
            spaceAfter=8,
        )
    )
    return styles


def _simple_table(data) -> Table:
    table = Table(data, colWidths=[3.2 * inch, 2.8 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a365d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def _accuracy_chart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(["New calls we hadn't seen before"], [84.5], color="#38a169", height=0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percent correct")
    ax.set_title("How often our test got it right")
    ax.text(84.5, 0, "  84.5%", va="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _volume_chart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = ["Real person", "AI caller"]
    values = [0.066, 0.153]
    bars = ax.bar(labels, values, color=["#3182ce", "#e53e3e"])
    ax.set_ylabel("Average voice loudness")
    ax.set_title("AI callers tend to sound louder on the phone")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _variety_chart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = ["Real person", "AI caller"]
    values = [0.605, 0.244]
    ax.bar(labels, values, color=["#3182ce", "#e53e3e"])
    ax.set_ylabel("How much the voice varies (higher = less robotic)")
    ax.set_title("Real voices change more from moment to moment")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_pdf() -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    accuracy_path = CHART_DIR / "simple_accuracy.png"
    volume_path = CHART_DIR / "simple_volume.png"
    variety_path = CHART_DIR / "simple_variety.png"
    _accuracy_chart(accuracy_path)
    _volume_chart(volume_path)
    _variety_chart(variety_path)

    styles = _setup_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.85 * inch,
        leftMargin=0.85 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title="Can You Tell If a Bank Caller Is Real?",
    )
    story = []

    story.append(Spacer(1, 1.4 * inch))
    story.append(Paragraph("Can You Tell If a Bank Caller Is Real?", styles["BigTitle"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(
            "A plain-English summary of our phone-call voice research",
            styles["Tagline"],
        )
    )
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph(f"{date.today().strftime('%B %d, %Y')}", styles["Tagline"]))
    story.append(PageBreak())

    story.append(Paragraph("What's the problem?", styles["Section"]))
    story.append(
        Paragraph(
            "When you call a bank, you expect the person on the other end to be real. "
            "But today, AI can call in too — using a synthetic voice that sounds human. "
            "That's a security risk: fraudsters could use fake voices to trick customer service.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Our project asks a simple question: <b>listening to a phone recording, can we tell "
            "if the caller is a real person or a machine?</b>",
            styles["Body"],
        )
    )

    story.append(Paragraph("What we did", styles["Section"]))
    story.append(
        Paragraph(
            "We studied <b>353 real phone calls</b> in Mexican Spanish between bank customers "
            "and an automated bank agent. About <b>150 calls</b> were from real people. "
            "About <b>203</b> were from AI systems pretending to be callers (speech recognition + "
            "AI brain + synthetic voice).",
            styles["Body"],
        )
    )
    for item in [
        "Each call was recorded on two separate tracks — one for the caller, one for the bank agent.",
        "We built software to measure <b>how people talk</b> (timing, pauses, who speaks when).",
        "We also measured <b>how the voice sounds</b> (loudness, pitch changes, natural variation).",
        "We checked our work on calls the system had never seen before.",
    ]:
        story.append(Paragraph(f"• {item}", styles["SimpleBullet"]))

    story.append(PageBreak())
    story.append(Paragraph("What we found", styles["Section"]))
    story.append(
        Paragraph(
            "<b>The biggest clue isn't what people say — it's how steady the fake voice sounds.</b>",
            styles["Highlight"],
        )
    )

    story.append(Paragraph("1. AI callers are louder", styles["Section"]))
    story.append(
        Paragraph(
            "On average, synthetic callers come through the phone at roughly <b>twice the loudness</b> "
            "of real people. Think of it like a recording that's been volume-boosted.",
            styles["Body"],
        )
    )
    story.append(Image(str(volume_path), width=4.8 * inch, height=2.7 * inch))

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("2. Real voices are messier — and that's good", styles["Section"]))
    story.append(
        Paragraph(
            "Real people don't sound the same every second. Their volume shifts, their tone wobbles, "
            "they pause unpredictably. AI voices are smoother and more <b>same-y</b> — like a "
            "well-produced audio clip on repeat.",
            styles["Body"],
        )
    )
    story.append(Image(str(variety_path), width=4.8 * inch, height=2.7 * inch))

    story.append(PageBreak())
    story.append(Paragraph("3. AI callers talk more", styles["Section"]))
    story.append(
        Paragraph(
            "Synthetic callers held the floor longer — about <b>26%</b> of the call vs <b>18%</b> for "
            "real people. Real calls had more time with the bank agent speaking.",
            styles["Body"],
        )
    )

    story.append(Paragraph("4. Real people react less predictably", styles["Section"]))
    story.append(
        Paragraph(
            "When the bank agent stops talking, real callers don't always respond on the same beat. "
            "AI callers tend to jump in on a more regular schedule — but the <b>sound</b> of the voice "
            "was still the stronger giveaway overall.",
            styles["Body"],
        )
    )

    story.append(Spacer(1, 0.1 * inch))
    story.append(
        _simple_table(
            [
                ["In plain terms", "What we saw"],
                ["Fake voices", "Louder, smoother, more uniform"],
                ["Real voices", "Quieter, more varied, less predictable"],
                ["Who talks more", "AI callers speak more; real callers get interrupted more"],
            ]
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("How well does it work?", styles["Section"]))
    story.append(
        Paragraph(
            "Using just three sound-pattern measurements, our first automatic test correctly "
            "labeled callers about <b>85 out of 100 times</b> on phone calls it had never heard before.",
            styles["Body"],
        )
    )
    story.append(Image(str(accuracy_path), width=4.8 * inch, height=2.7 * inch))
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "That's promising for a first pass — but not perfect. A bank would need high confidence "
            "before blocking or flagging a caller, so more work remains.",
            styles["Body"],
        )
    )

    story.append(Paragraph("Why this matters", styles["Section"]))
    story.append(
        Paragraph(
            "Millions of people still bank by phone — especially where apps aren't an option. "
            "If synthetic voices become cheap and convincing, phone banking only stays trustworthy "
            "if the system can verify who is calling. This research is an early step toward that.",
            styles["Body"],
        )
    )

    story.append(Paragraph("What comes next", styles["Section"]))
    for item in [
        "Build a live web tool banks could ping with a recording and get an answer back.",
        "Improve accuracy and confidence scores so the system knows when it's unsure.",
        "Test on more voices, accents, and phone conditions so it works in the real world.",
        "Combine sound analysis with conversation patterns for even better results.",
    ]:
        story.append(Paragraph(f"• {item}", styles["SimpleBullet"]))

    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph(
            "<i>This summary is based on exploratory research for a voice-security hackathon. "
            "It is not a finished product ready for production use.</i>",
            styles["Body"],
        )
    )

    doc.build(story)
    return OUTPUT


if __name__ == "__main__":
    print(build_pdf())
