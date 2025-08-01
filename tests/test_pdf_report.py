import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODULE_DIR = os.path.join(BASE_DIR, "coding", "survey_analysis_mvp")
sys.path.insert(0, MODULE_DIR)

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import reporting


def test_generate_pdf_report(tmp_path):
    summary = {
        "analysis_target": "Test Column",
        "summary_text": "dummy summary",
        "action_items": [],
        "sentiment_counts": pd.Series([1, 0, 0], index=["positive", "neutral", "negative"]),
        "topic_counts": pd.Series([2], index=["Topic"]),
    }
    output_pdf = tmp_path / "report.pdf"
    reporting.generate_pdf_report(summary, str(output_pdf))
    assert output_pdf.exists()
