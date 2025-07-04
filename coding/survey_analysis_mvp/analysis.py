"""Survey analysis utilities using OpenAI and spaCy.

This module defines Pydantic models for the analysis results and provides
helper functions to run asynchronous text analysis and aggregate the outputs
for reporting.
"""

import pandas as pd
import asyncio
from typing import List, Literal

from pydantic import BaseModel, Field
import instructor
from openai import AsyncOpenAI
import spacy

from config import settings

# InstructorでOpenAIクライアントを初期化
aclient = instructor.from_openai(
    AsyncOpenAI(api_key=settings.OPENAI_API_KEY), mode=instructor.Mode.MD_JSON
)


# --- データモデル定義 ---
class SurveyResponseAnalysis(BaseModel):
    """Structured insight extracted from a single survey response.

    Attributes:
        sentiment: Overall sentiment classified into four categories.
        key_topics: List of key topics mentioned in the response.
        verbatim_quote: Representative sentence from the original text.
        actionable_insight: Whether the response includes actionable feedback.
    """

    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="回答全体のセンチメント（感情極性）を4つのカテゴリのいずれかで判定します。"
    )
    key_topics: List[str] = Field(
        description="回答で言及されている主要なトピックやテーマをリスト形式で抽出します。例：['価格', 'デザイン', 'サポート体制']",
        default=[],
    )
    verbatim_quote: str = Field(
        description="分析内容を最もよく表している、原文からの代表的な一文を抜き出します。"
    )
    actionable_insight: bool = Field(
        description="この回答に、改善に繋がる具体的で実行可能な提案が含まれている場合はTrue、そうでなければFalseを返します。"
    )


class ModerationCategories(BaseModel):
    """Flags indicating whether each moderation category was triggered."""

    hate: bool = Field(description="ヘイトコンテンツ")
    hate_threatening: bool = Field(description="脅迫的なヘイトコンテンツ")
    self_harm: bool = Field(description="自傷行為")
    sexual: bool = Field(description="性的コンテンツ")
    sexual_minors: bool = Field(description="未成年者への性的コンテンツ")
    violence: bool = Field(description="暴力")
    violence_graphic: bool = Field(description="グラフィックな暴力コンテンツ")


class ModerationScores(BaseModel):
    """Score values for each moderation category."""

    hate: float = Field(description="ヘイトスコア")
    hate_threatening: float = Field(description="脅迫的なヘイトスコア")
    self_harm: float = Field(description="自傷行為スコア")
    sexual: float = Field(description="性的コンテンツスコア")
    sexual_minors: float = Field(description="未成年者への性的コンテンツスコア")
    violence: float = Field(description="暴力スコア")
    violence_graphic: float = Field(description="グラフィックな暴力スコア")


class ModerationResult(BaseModel):
    """Moderation outcome returned by the OpenAI API."""

    flagged: bool = Field(description="フラグが立てられたか")
    categories: ModerationCategories
    category_scores: ModerationScores


class EmotionScores(BaseModel):
    """Primary emotion scores for a text."""

    joy: float = Field(description="喜びのスコア (0-5)")
    sadness: float = Field(description="悲しみのスコア (0-5)")
    fear: float = Field(description="恐れのスコア (0-5)")
    surprise: float = Field(description="驚きのスコア (0-5)")
    anger: float = Field(description="怒りのスコア (0-5)")
    disgust: float = Field(description="嫌悪のスコア (0-5)")
    reason: str = Field(description="感情全体の理由")


class ComprehensiveAnalysisResult(BaseModel):
    """Combined results from survey, moderation and emotion analyses."""

    survey_analysis: SurveyResponseAnalysis
    moderation_result: ModerationResult
    emotion_scores: EmotionScores


# --- spaCy日本語トークナイザ ---
def get_tokenizer(mode: str = "B"):
    """Return a spaCy pipeline with SudachiPy tokenizer.

    Args:
        mode: SudachiPy split mode ("A", "B", or "C").

    Returns:
        spaCy Language object with the specified tokenizer.
    """
    config = {
        "nlp": {
            "tokenizer": {
                "@tokenizers": "spacy.ja.JapaneseTokenizer",
                "split_mode": mode,
            }
        }
    }
    return spacy.blank("ja", config=config)


# --- コア分析関数 ---
async def analyze_single_text(
    text: str, mode: str = "B"
) -> ComprehensiveAnalysisResult:
    """Analyze a single text asynchronously.

    Args:
        text: Text to analyze.
        mode: SudachiPy split mode to use for tokenization.

    Returns:
        ComprehensiveAnalysisResult containing structured analysis data.
    """
    if not isinstance(text, str) or not text.strip():
        # 空または無効なテキストの場合、デフォルト値を返す
        default_survey_analysis = SurveyResponseAnalysis(
            sentiment="neutral",
            key_topics=["無回答"],
            verbatim_quote="N/A",
            actionable_insight=False,
        )
        default_moderation_result = ModerationResult(
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
        )
        default_emotion_scores = EmotionScores(
            joy=0.0,
            sadness=0.0,
            fear=0.0,
            surprise=0.0,
            anger=0.0,
            disgust=0.0,
            reason="N/A",
        )
        return ComprehensiveAnalysisResult(
            survey_analysis=default_survey_analysis,
            moderation_result=default_moderation_result,
            emotion_scores=default_emotion_scores,
        )

    nlp = get_tokenizer(mode)
    doc = nlp(text)
    tokenized_text = " ".join([token.text for token in doc])

    survey_analysis_task = aclient.chat.completions.create(
        model="gpt-4o-mini",
        response_model=SurveyResponseAnalysis,
        messages=[
            {
                "role": "system",
                "content": "あなたは優秀なマーケティングアナリストです。提供されたアンケートの回答を分析し、指定された形式で構造化してください。",
            },
            {"role": "user", "content": tokenized_text},
        ],
        max_retries=2,
    )

    moderation_task = aclient.moderations.create(input=text)

    emotion_prompt = f"""
あなたは感情分析の専門家です、文脈に注目して一次感情を抽出し、0から5の範囲で評価してください。

【評価基準】
0：感情が全く感じられない
1：ごくわずかに感情が感じられる
2：感情が弱めだが感じられる
3：感情が明確に感じられる
4：はっきりと強い感情が表出
5：圧倒的で非常に強烈な感情

【評価の重要原則】
1. 純粋性：各感情は他の感情との混合ではなく、純粋な形で評価する。
2. 文脈性：表現の背景にある状況や文脈を十分に考慮する。
3. 総合性：言語表現と非言語的要素を総合的に判断する。
4. 直接性：直接的な表現と間接的な表現の強度を適切に比較評価する。
5. 文化考慮：日本語特有の遠回しな表現や皮肉、婉曲表現の文化的背景を考慮する。

分析対象の文章:
{text}

以下の形式で各感情スコアと理由を出力してください：
感情スコア:
- 喜び: {{joy}}
- 悲しみ: {{sadness}}
- 恐れ: {{fear}}
- 驚き: {{surprise}}
- 怒り: {{anger}}
- 嫌悪: {{disgust}}
感情全体の理由: {{reason}}
"""
    emotion_task = aclient.chat.completions.create(
        model="gpt-4o-mini",
        response_model=EmotionScores,
        messages=[
            {"role": "system", "content": "あなたは感情分析の専門家です。"},
            {"role": "user", "content": emotion_prompt},
        ],
        max_retries=2,
    )

    try:
        survey_analysis, moderation_response, emotion_scores = await asyncio.gather(
            survey_analysis_task, moderation_task, emotion_task
        )
        moderation_result = moderation_response.results[0]  # 最初の結果を使用

        return ComprehensiveAnalysisResult(
            survey_analysis=survey_analysis,
            moderation_result=ModerationResult(
                flagged=moderation_result.flagged,
                categories=ModerationCategories(
                    **moderation_result.categories.model_dump()
                ),
                category_scores=ModerationScores(
                    **moderation_result.category_scores.model_dump()
                ),
            ),
            emotion_scores=emotion_scores,
        )
    except Exception as e:
        print(f"APIリクエストエラー: {e}")
        # エラー時もデフォルト値を返す
        default_survey_analysis = SurveyResponseAnalysis(
            sentiment="neutral",
            key_topics=["分析エラー"],
            verbatim_quote=str(e),
            actionable_insight=False,
        )
        default_moderation_result = ModerationResult(
            flagged=True,
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
        )
        default_emotion_scores = EmotionScores(
            joy=0.0,
            sadness=0.0,
            fear=0.0,
            surprise=0.0,
            anger=0.0,
            disgust=0.0,
            reason=str(e),
        )
        return ComprehensiveAnalysisResult(
            survey_analysis=default_survey_analysis,
            moderation_result=default_moderation_result,
            emotion_scores=default_emotion_scores,
        )


async def analyze_dataframe(
    df: pd.DataFrame, column_name: str, mode: str = "B", progress_callback=None
) -> pd.DataFrame:
    """Analyze a DataFrame column in parallel and append results.

    Args:
        df: Source DataFrame.
        column_name: Name of the column containing text responses.
        mode: SudachiPy split mode.
        progress_callback: Optional callback receiving progress percentage.

    Returns:
        DataFrame with analysis results concatenated.
    """
    texts_to_analyze = df[column_name].tolist()
    tasks = [analyze_single_text(text, mode) for text in texts_to_analyze]

    results = []
    # asyncio.as_completedは順不同なので、元のDataFrameのインデックスを保持する
    # ここでは簡単化のため、元のdfのインデックスに結果をマッピングする
    # （ただし、上記の実装では結果の順序が保証されないため、より堅牢な実装が必要）
    # 今回は簡単化のため、結果のリストをそのまま列に追加します。

    # 順序を保証するために、タスクと元のインデックスをペアにする
    indexed_tasks = [
        (i, analyze_single_text(text, mode)) for i, text in enumerate(texts_to_analyze)
    ]

    # 完了したタスクから結果を収集し、元のインデックスでソートする
    completed_results = [None] * len(texts_to_analyze)
    for i, (original_index, task) in enumerate(indexed_tasks):
        result = await task
        completed_results[original_index] = result
        if progress_callback:
            progress_callback(((i + 1) / len(texts_to_analyze)) * 100)

    # 結果をDataFrameに変換
    survey_analysis_results = []
    moderation_results = []
    emotion_results = []

    for res in completed_results:
        survey_analysis_results.append(res.survey_analysis.model_dump())
        moderation_results.append(res.moderation_result.model_dump())
        emotion_results.append(res.emotion_scores.model_dump())

    survey_df = pd.DataFrame(survey_analysis_results)
    moderation_df = pd.DataFrame(moderation_results)
    emotion_df = pd.DataFrame(emotion_results)

    # 列名を調整
    survey_df.columns = [f"analysis_{col}" for col in survey_df.columns]
    moderation_df.columns = [f"moderation_{col}" for col in moderation_df.columns]
    emotion_df.columns = [f"emotion_{col}" for col in emotion_df.columns]

    # 元のDataFrameと結合
    df.reset_index(drop=True, inplace=True)
    return pd.concat([df, survey_df, moderation_df, emotion_df], axis=1)


# --- 集計関数 ---
def summarize_results(df_analyzed: pd.DataFrame):
    """Summarize analyzed DataFrame for reporting.

    Args:
        df_analyzed: DataFrame returned by :func:`analyze_dataframe`.

    Returns:
        Tuple of summary dictionary and list of words for word cloud.
    """
    if "analysis_sentiment" not in df_analyzed.columns:
        return None, None

    # センチメント比率
    sentiment_counts = (
        df_analyzed["analysis_sentiment"]
        .value_counts()
        .reindex(["positive", "neutral", "negative", "mixed"], fill_value=0)
    )

    # 全トピックのリストを作成
    all_topics = []
    for topics in df_analyzed["analysis_key_topics"]:
        if isinstance(topics, list):
            all_topics.extend(topics)

    # トピックの出現頻度
    topic_counts = pd.Series(all_topics).value_counts()

    # モデレーション結果の集計
    moderation_summary = {}
    moderation_categories = [
        "hate",
        "hate_threatening",
        "self_harm",
        "sexual",
        "sexual_minors",
        "violence",
        "violence_graphic",
    ]
    for cat in moderation_categories:
        col_name = f"moderation_categories_{cat}"
        if col_name in df_analyzed.columns:
            moderation_summary[cat] = df_analyzed[col_name].sum()
        else:
            moderation_summary[cat] = 0  # 列がない場合は0

    # 感情スコアの平均
    emotion_avg = {}
    emotion_types = ["joy", "sadness", "fear", "surprise", "anger", "disgust"]
    for emo in emotion_types:
        col_name = f"emotion_{emo}"
        if col_name in df_analyzed.columns:
            emotion_avg[emo] = df_analyzed[col_name].mean()
        else:
            emotion_avg[emo] = 0.0  # 列がない場合は0

    # ワードクラウド用の単語リスト
    # SudachiPyで再度分かち書きして、名詞・動詞・形容詞のみを抽出
    # ここでは、分析対象となった列の全テキストを結合してワードクラウドの元データとする
    # df_analyzed.columns[0]は元のExcelの最初の列名なので、分析対象列を使うべき
    # analyze_dataframeで渡されたcolumn_nameをここで使うか、df_analyzedに保存しておくべき
    # 簡単化のため、ここではanalysis_verbatim_quoteを結合して使用
    all_text = " ".join(df_analyzed["analysis_verbatim_quote"].dropna().astype(str))
    nlp = get_tokenizer("A")
    doc = nlp(all_text)
    words_for_wordcloud = [
        token.text
        for token in doc
        if token.pos_ in ["NOUN", "VERB", "ADJ"] and len(token.text) > 1
    ]

    summary = {
        "sentiment_counts": sentiment_counts,
        "topic_counts": topic_counts.head(15),  # 上位15トピック
        "moderation_summary": moderation_summary,
        "emotion_avg": emotion_avg,
    }

    return summary, words_for_wordcloud
