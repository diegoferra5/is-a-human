#!/usr/bin/env python3
"""Generate full PDF findings report for is-a-human exploratory analysis."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CHART_DIR = REPORTS / "_charts"
OUTPUT = REPORTS / "is-a-human-findings.pdf"

TRAIN_TOP = [
    ("caller_rms_mean", 1.50, "synthetic > human"),
    ("caller_spectral_flatness_std", -1.43, "human > synthetic"),
    ("caller_zcr_std", -1.37, "human > synthetic"),
    ("caller_rms_cv", -1.37, "human > synthetic"),
    ("caller_spectral_centroid_std", -1.36, "human > synthetic"),
    ("agent_aligned_recovery_cv", -1.32, "human > synthetic"),
    ("caller_response_latency_cv", -1.30, "human > synthetic"),
    ("caller_crest_factor_cv", -1.28, "human > synthetic"),
]

VAL_TOP = [
    ("caller_rms_mean", 2.26, "synthetic > human"),
    ("caller_crest_factor_cv", -1.82, "human > synthetic"),
    ("caller_zcr_std", -1.81, "human > synthetic"),
    ("caller_rms_cv", -1.56, "human > synthetic"),
    ("caller_crest_factor_std", -1.48, "human > synthetic"),
    ("caller_spectral_centroid_std", -1.47, "human > synthetic"),
    ("caller_spectral_flatness_std", -1.44, "human > synthetic"),
    ("agent_talk_ratio", -1.44, "human > synthetic"),
]

VAL_CONVERSATIONAL = [
    ("agent_talk_ratio", 0.556, 0.049, 0.440, 0.103, -1.44),
    ("caller_talk_ratio", 0.184, 0.044, 0.256, 0.073, 1.19),
    ("caller_response_latency_cv", 0.825, 0.136, 0.703, 0.115, -0.96),
    ("agent_response_latency_std_s", 1.89, 0.98, 3.77, 2.78, 0.90),
    ("agent_response_latency_mean_s", 2.78, 1.00, 4.80, 3.25, 0.84),
]


def _setup_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitlePage",
            parent=styles["Title"],
            fontSize=26,
            leading=32,
            alignment=TA_CENTER,
            spaceAfter=20,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Normal"],
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#444444"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontSize=16,
            leading=20,
            spaceBefore=14,
            spaceAfter=10,
            textColor=colors.HexColor("#1a365d"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubSection",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#2c5282"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    if "ReportBullet" not in styles:
        styles.add(
            ParagraphStyle(
                name="ReportBullet",
                parent=styles["Normal"],
                fontSize=10,
                leading=13,
                leftIndent=14,
                bulletIndent=0,
                spaceAfter=4,
            )
        )
    return styles


def _table(data, col_widths=None) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a365d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _effect_chart(path: Path) -> None:
    features = [row[0].replace("caller_", "").replace("agent_", "")[:22] for row in TRAIN_TOP]
    train_d = [row[1] for row in TRAIN_TOP]
    val_map = {row[0]: row[1] for row in VAL_TOP}
    val_d = [val_map.get(row[0], 0) for row in TRAIN_TOP]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    y = range(len(features))
    ax.barh([i - 0.15 for i in y], train_d, height=0.3, label="Train", color="#3182ce")
    ax.barh([i + 0.15 for i in y], val_d, height=0.3, label="Val", color="#e53e3e")
    ax.set_yticks(list(y))
    ax.set_yticklabels(features, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cohen's d (synthetic − human direction encoded in sign)")
    ax.set_title("Top Feature Separators: Train vs Val Effect Sizes")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _classifier_chart(path: Path) -> None:
    metrics = ["Accuracy", "F1", "AUC"]
    train = [0.869, 0.891, 0.922]
    val = [0.845, 0.845, 0.963]

    x = range(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar([i - width / 2 for i in x], train, width, label="Train", color="#3182ce")
    ax.bar([i + width / 2 for i in x], val, width, label="Val", color="#38a169")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Logistic Baseline (caller_rms_mean, caller_zcr_std, caller_rms_cv)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _rms_chart(path: Path) -> None:
    labels = ["Human", "Synthetic"]
    train_human, train_synth = 0.069, 0.150
    val_human, val_synth = 0.066, 0.153

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, h, s, title in [
        (axes[0], train_human, train_synth, "Train"),
        (axes[1], val_human, val_synth, "Val"),
    ]:
        ax.bar(labels, [h, s], color=["#3182ce", "#e53e3e"])
        ax.set_title(title)
        ax.set_ylabel("Caller RMS Mean")
    fig.suptitle("Strongest Separator: caller_rms_mean")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_pdf() -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    effect_path = CHART_DIR / "effect_sizes.png"
    classifier_path = CHART_DIR / "classifier.png"
    rms_path = CHART_DIR / "rms_mean.png"
    _effect_chart(effect_path)
    _classifier_chart(classifier_path)
    _rms_chart(rms_path)

    styles = _setup_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="is-a-human Exploratory Analysis Findings",
    )
    story = []

    story.append(Spacer(1, 1.6 * inch))
    story.append(Paragraph("is-a-human", styles["TitlePage"]))
    story.append(Paragraph("Exploratory Analysis Findings Report", styles["TitlePage"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(
        Paragraph(
            "Human vs. Synthetic Caller Detection — HackMTY 2026 Challenge Track",
            styles["Subtitle"],
        )
    )
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(f"Generated: {date.today().isoformat()}", styles["Subtitle"]))
    story.append(PageBreak())

    story.append(Paragraph("Executive Summary", styles["Section"]))
    story.append(
        Paragraph(
            "This report documents Phase 0 infrastructure validation and Phase 1 exploratory "
            "analysis for detecting synthetic callers in dual-channel telephony recordings. "
            "We processed 353 Mexican Spanish bank customer-service calls (150 human, 203 synthetic) "
            "using a dual-channel pipeline: in-memory audio demux, Silero VAD, turn-ledger construction, "
            "conversational metrics, unconventional acoustic descriptors, and agent-turn-aligned recovery features.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Key finding:</b> Acoustic uniformity features on the caller channel separate human and "
            "synthetic voices more reliably than talk-time ratios alone. Synthetic callers exhibit "
            "higher mean RMS energy but lower variability across spectral and temporal descriptors. "
            "A logistic baseline using three stable acoustic features achieves <b>84.5% accuracy</b> and "
            "<b>0.963 AUC</b> on the held-out validation split.",
            styles["Body"],
        )
    )

    story.append(Paragraph("Challenge Objective", styles["Section"]))
    story.append(
        Paragraph(
            "Given a stereo 8 kHz phone recording (channel 0 = caller, channel 1 = agent), classify "
            "whether the caller is a real human or a synthetic voice stack (ASR + LLM + TTS). "
            "Deliverable: <font face='Courier'>POST /detect</font> returning "
            "<font face='Courier'>{\"is_synthetic\": bool, \"confidence\": float}</font>.",
            styles["Body"],
        )
    )

    story.append(Paragraph("Dataset", styles["Section"]))
    story.append(
        _table(
            [
                ["Split", "Human", "Synthetic", "Total"],
                ["Train", "113", "169", "282"],
                ["Val", "37", "34", "71"],
                ["Total", "150", "203", "353"],
            ],
            col_widths=[1.4 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch],
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "Audio: stereo, 8 kHz, 16-bit PCM. Duration 61–273 s (mean ~148 s). "
            "Train/val splits are speaker-disjoint. Organizer-provided turn segments "
            "used for alignment validation and recovery features.",
            styles["Body"],
        )
    )

    story.append(Paragraph("Phase 0 — Pipeline Validation", styles["Section"]))
    story.append(
        _table(
            [
                ["Metric", "Val Split (n=71)"],
                ["Caller VAD IoU vs organizer turns", "0.811"],
                ["Agent VAD IoU vs organizer turns", "0.941"],
                ["Mean pipeline latency", "769 ms"],
                ["p95 pipeline latency", "1,041 ms"],
            ],
            col_widths=[3.5 * inch, 2.0 * inch],
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "The foundation pipeline reliably segments speech on both channels and runs within "
            "acceptable latency for post-call batch analysis (~0.8 s per call on CPU).",
            styles["Body"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Methodology", styles["Section"]))
    story.append(Paragraph("Feature Categories (47 numeric features per call)", styles["SubSection"]))
    for item in [
        "<b>Conversational:</b> talk-time ratios, overlap/silence proportions, segment counts, response latencies",
        "<b>Acoustic (unconventional):</b> RMS energy, zero-crossing rate, spectral centroid/flatness, HF/LF ratio, crest factor, intra-call silence gaps",
        "<b>Recovery (agent-turn-aligned):</b> caller response latency after agent stops, post-overlap recovery, barge-in/backchannel counts, agent turn fragmentation",
    ]:
        story.append(Paragraph(f"• {item}", styles["ReportBullet"]))

    story.append(Paragraph("Analysis Protocol", styles["SubSection"]))
    for item in [
        "Extract features on all 353 calls using dual Silero VAD + organizer turn alignment",
        "Compare human vs synthetic distributions via Cohen's d effect sizes",
        "Bootstrap 200 resamples per feature for 95% confidence intervals",
        "Measure train/val rank correlation for feature stability",
        "Train logistic regression on top 3 stable features; evaluate on val",
    ]:
        story.append(Paragraph(f"• {item}", styles["ReportBullet"]))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Primary Hypothesis", styles["SubSection"]))
    story.append(
        Paragraph(
            "Synthetic callers recover from conversational turbulence more consistently than humans. "
            "<b>Result:</b> Partially inverted — humans show <i>more</i> variability in caller response "
            "latency (CV) and agent-aligned recovery timing. Synthetic voices are acoustically more uniform, "
            "not more consistent in turn-taking alone.",
            styles["Body"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Findings — Multi-Run Analysis", styles["Section"]))
    story.append(
        Paragraph(
            "Split rank correlation: <b>0.818</b> — top separators are stable between train and val. "
            "Stable features selected for classifier: <font face='Courier'>caller_rms_mean</font>, "
            "<font face='Courier'>caller_zcr_std</font>, <font face='Courier'>caller_rms_cv</font>.",
            styles["Body"],
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(Image(str(rms_path), width=5.5 * inch, height=2.4 * inch))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Image(str(effect_path), width=6.5 * inch, height=3.4 * inch))

    story.append(PageBreak())
    story.append(Paragraph("Top Separators — Train Split (n=282)", styles["SubSection"]))
    train_rows = [["Feature", "Cohen's d", "Direction"]]
    train_rows.extend([[row[0], f"{row[1]:+.2f}", row[2]] for row in TRAIN_TOP])
    story.append(_table(train_rows, col_widths=[2.8 * inch, 1.0 * inch, 2.0 * inch]))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Top Separators — Val Split (n=71)", styles["SubSection"]))
    val_rows = [["Feature", "Cohen's d", "95% CI", "Direction"]]
    val_ci = {
        "caller_rms_mean": "[+1.74, +2.97]",
        "caller_crest_factor_cv": "[-2.48, -1.52]",
        "caller_zcr_std": "[-2.35, -1.51]",
        "caller_rms_cv": "[-2.07, -1.17]",
        "caller_crest_factor_std": "[-2.23, -1.20]",
        "caller_spectral_centroid_std": "[-2.01, -1.21]",
        "caller_spectral_flatness_std": "[-1.92, -1.17]",
        "agent_talk_ratio": "[-2.12, -1.09]",
    }
    val_rows.extend(
        [[row[0], f"{row[1]:+.2f}", val_ci.get(row[0], ""), row[2]] for row in VAL_TOP]
    )
    story.append(_table(val_rows, col_widths=[2.4 * inch, 0.8 * inch, 1.2 * inch, 1.4 * inch]))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Initial Conversational Features (Val Only)", styles["SubSection"]))
    conv_rows = [["Feature", "Human", "Synthetic", "d"]]
    for name, hm, hs, sm, ss, d in VAL_CONVERSATIONAL:
        conv_rows.append([name, f"{hm:.3f}±{hs:.3f}", f"{sm:.3f}±{ss:.3f}", f"{d:+.2f}"])
    story.append(_table(conv_rows, col_widths=[2.2 * inch, 1.3 * inch, 1.3 * inch, 0.8 * inch]))

    story.append(PageBreak())
    story.append(Paragraph("Classifier Baseline", styles["Section"]))
    story.append(
        Paragraph(
            "Logistic regression trained on train split using "
            "<font face='Courier'>caller_rms_mean</font>, "
            "<font face='Courier'>caller_zcr_std</font>, "
            "<font face='Courier'>caller_rms_cv</font>. "
            "Features standardized using train statistics.",
            styles["Body"],
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        _table(
            [
                ["Split", "Accuracy", "F1", "AUC"],
                ["Train (n=282)", "86.9%", "0.891", "0.922"],
                ["Val (n=71)", "84.5%", "0.845", "0.963"],
            ],
            col_widths=[1.6 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch],
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(Image(str(classifier_path), width=4.5 * inch, height=2.6 * inch))

    story.append(Paragraph("Interpretation", styles["Section"]))
    for item in [
        "<b>Louder but uniform:</b> Synthetic callers have ~2× higher mean RMS on the caller channel, yet lower variance in energy, pitch proxy (ZCR), and spectral descriptors — consistent with TTS output leveling.",
        "<b>Humans are messier:</b> Higher CV in response latencies, crest factor, and spectral centroid std suggests natural conversational variability.",
        "<b>Agent talk ratio:</b> Human calls retain more agent speech (55.6% vs 44.0% on val) — synthetic callers may dominate the caller channel without being interrupted as often.",
        "<b>Recovery features:</b> Agent-aligned recovery CV separates on train (d = −1.32) but is secondary to acoustic features on val.",
        "<b>Generalization:</b> Val AUC (0.963) exceeds train AUC (0.922), suggesting acoustic features transfer without overfitting on this split.",
    ]:
        story.append(Paragraph(f"• {item}", styles["ReportBullet"]))

    story.append(Paragraph("Recommended Next Steps", styles["Section"]))
    for item in [
        "Wire logistic (or gradient-boosted) classifier into <font face='Courier'>POST /detect</font> using stable acoustic features",
        "Calibrate confidence scores on val split (Platt scaling or isotonic regression)",
        "Add caller-only VAD segment extraction before acoustic features to reduce silence contamination",
        "Evaluate on longer/shorter calls separately for robustness",
        "Optional: layer conversational recovery features as secondary signal",
        "Deploy endpoint and measure end-to-end latency under judging conditions",
    ]:
        story.append(Paragraph(f"• {item}", styles["ReportBullet"]))

    story.append(Spacer(1, 0.3 * inch))
    story.append(
        Paragraph(
            "<i>Report generated by is-a-human analysis pipeline. "
            "Dataset licensed for HackMTY 2026 only; not for redistribution.</i>",
            styles["Body"],
        )
    )

    doc.build(story)
    return OUTPUT


if __name__ == "__main__":
    path = build_pdf()
    print(path)
