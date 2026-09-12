from html_reporter import render_report


def test_render_report_embeds_results():
    html = render_report(
        {
            "generated_at": "2026-09-12 22:00 UTC",
            "duration_s": 0.12,
            "exitstatus": 0,
            "counts": {"passed": 1, "failed": 0, "skipped": 0},
            "tests": [
                {
                    "nodeid": "tests/test_vad_benchmark.py::test_iou",
                    "name": "test_iou",
                    "file": "tests/test_vad_benchmark.py",
                    "suite": "vad",
                    "outcome": "passed",
                    "duration_s": 0.01,
                    "error": "",
                    "metrics": {"caller_iou": 0.81},
                }
            ],
        }
    )

    assert "is-a-human" in html
    assert "caller_iou" in html
    assert "test_iou" in html
