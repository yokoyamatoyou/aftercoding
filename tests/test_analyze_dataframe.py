import os
import sys
import pandas as pd
import asyncio

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODULE_DIR = os.path.join(BASE_DIR, "coding", "survey_analysis_mvp")
sys.path.insert(0, MODULE_DIR)

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import analysis
from analysis import (
    SurveyResponseAnalysis,
    ModerationCategories,
    ModerationScores,
    ModerationResult,
    EmotionScores,
    ComprehensiveAnalysisResult,
)


async def fake_analyze_single_text(text: str, mode: str = "B") -> ComprehensiveAnalysisResult:
    return ComprehensiveAnalysisResult(
        survey_analysis=SurveyResponseAnalysis(
            sentiment="positive",
            key_topics=["topic"],
            verbatim_quote="quote",
            actionable_insight=False,
        ),
        moderation_result=ModerationResult(
            flagged=False,
            categories=ModerationCategories(
                hate=False,
                hate_threatening=False,
                self_harm=False,
                sexual=False,
                sexual_minors=False,
                violence=False,
                violence_graphic=False,
            ),
            category_scores=ModerationScores(
                hate=0.0,
                hate_threatening=0.0,
                self_harm=0.0,
                sexual=0.0,
                sexual_minors=0.0,
                violence=0.0,
                violence_graphic=0.0,
            ),
        ),
        emotion_scores=EmotionScores(
            joy=1.0,
            sadness=0.0,
            fear=0.0,
            surprise=0.0,
            anger=0.0,
            disgust=0.0,
            reason="",
        ),
    )


def test_analyze_dataframe(monkeypatch):
    df = pd.DataFrame({"text": ["a", "b"]})
    monkeypatch.setattr(analysis, "analyze_single_text", fake_analyze_single_text)
    result = asyncio.run(analysis.analyze_dataframe(df, "text", max_concurrent_tasks=2))
    assert "analysis_sentiment" in result.columns
    assert len(result) == 2
    assert all(result["analysis_sentiment"] == "positive")
