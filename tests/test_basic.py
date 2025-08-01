import os
import sys

# Add the survey_analysis_mvp directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODULE_DIR = os.path.join(BASE_DIR, "coding", "survey_analysis_mvp")
sys.path.insert(0, MODULE_DIR)

# Provide a dummy API key so importing analysis does not fail
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import analysis
import reporting

def test_get_tokenizer_runs():
    tok = analysis.get_tokenizer()
    assert tok is not None

def test_set_japanese_font_runs():
    result = reporting.set_japanese_font()
    assert isinstance(result, bool)
