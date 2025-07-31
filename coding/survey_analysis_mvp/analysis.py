"""Survey analysis utilities using OpenAI and spaCy.

This module defines Pydantic models for the analysis results and provides
helper functions to run asynchronous text analysis and aggregate the outputs
for reporting.
"""

import pandas as pd
import asyncio
from typing import List, Literal
from functools import lru_cache

import openai

from pydantic import BaseModel, Field
import instructor
from openai import AsyncOpenAI
import spacy

from config import settings

# Read API key from .env or environment variables
openai.api_key = settings.OPENAI_API_KEY


def analyze_survey(file_path: str, column_name: str):
    """Analyze survey responses and summarize positive/negative feedback.

    Args:
        file_path: Path to the Excel file.
        column_name: Column containing free text responses.

    Returns:
        Tuple of (DataFrame with sentiment column, positive summary text,
        negative summary text).
    """
    df = pd.read_excel(file_path)
    texts = df[column_name].fillna("").astype(str)
    sentiments = []

    system_prompt = (
        "あなたはテキスト分析の専門家です。与えられたテキストを"
        "ポジティブ、ネガティブ、ニュートラルのいずれか一つに分類してください。"
    )

    for text in texts:
        if not text.strip():
            sentiments.append("neutral")
            continue
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            sentiment = response.choices[0].message.content.strip().lower()
        except Exception:
            sentiment = "neutral"
        sentiments.append(sentiment)

    df["sentiment"] = sentiments

    positive_texts = df.loc[df["sentiment"] == "positive", column_name].tolist()
    negative_texts = df.loc[df["sentiment"] == "negative", column_name].tolist()

    def summarize(text_list):
        if not text_list:
            return ""
        prompt = (
            "以下の意見リストは、製品アンケートの回答です。マーケティング担当者が"
            "傾向を把握できるよう、重要なポイントを3つにまとめてください。\n\n" +
            "\n".join(text_list)
        )
        try:
            res = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            return res.choices[0].message.content.strip()
        except Exception:
            return ""

    positive_summary = summarize(positive_texts)
    negative_summary = summarize(negative_texts)

    return df, positive_summary, negative_summary

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


class ReportCommentary(BaseModel):
    """LLMによって生成されたレポートの解説文。"""

    summary_text: str = Field(
        description="分析結果全体を3つのポイントで要約した総括文。"
    )
    action_items: List[str] = Field(
        description="分析結果から考えられる具体的なネクストアクションの提案リスト。3つ提案すること。"
    )
    sentiment_commentary: str = Field(
        description="感情分析の円グラフから読み取れるインサイトや注目点を解説する文章。"
    )
    topics_commentary: str = Field(
        description="主要トピックの棒グラフから読み取れるインサイトや注目点を解説する文章。"
    )


# --- spaCy日本語トークナイザ ---
@lru_cache(maxsize=3)
def get_tokenizer(mode: str = "B") -> spacy.Language:
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


async def generate_report_commentary(summary_data: dict) -> ReportCommentary:
    """集計済みデータに基づき、LLMにレポートの解説文を生成させる。"""

    context = f"""
    以下のアンケート分析結果データに基づき、プロのマーケティングアナリストとして、示唆に富んだレポート解説文を生成してください。

    # 感情分析結果 (件数)
    {summary_data.get('sentiment_counts', 'データなし').to_string()}

    # 主要トピック Top 15 (件数)
    {summary_data.get('topic_counts', 'データなし').to_string()}
    """

    try:
        commentary = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            response_model=ReportCommentary,
            messages=[
                {
                    "role": "system",
                    "content": "あなたは、データからインサイトを抽出し、分かりやすく解説する優秀なマーケティングアナリストです。",
                },
                {"role": "user", "content": context},
            ],
            max_retries=2,
        )
        return commentary
    except Exception as e:
        print(f"LLM解説生成エラー: {e}")
        return ReportCommentary(
            summary_text="解説の生成中にエラーが発生しました。",
            action_items=["N/A"],
            sentiment_commentary="解説の生成中にエラーが発生しました。",
            topics_commentary="解説の生成中にエラーが発生しました。",
        )


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
    df: pd.DataFrame,
    column_name: str,
    mode: str = "B",
    progress_callback=None,
    max_concurrent_tasks: int | None = None,
) -> pd.DataFrame:
    """Analyze a DataFrame column in parallel and append results.

    Args:
        df: Source DataFrame.
        column_name: Name of the column containing text responses.
        mode: SudachiPy split mode.
        progress_callback: Optional callback receiving progress percentage.
        max_concurrent_tasks: Maximum number of analysis tasks to run
            concurrently. Defaults to ``settings.MAX_CONCURRENT_TASKS``.

    Returns:
        DataFrame with analysis results concatenated.
    """
    texts_to_analyze = df[column_name].tolist()

    if max_concurrent_tasks is None:
        max_concurrent_tasks = settings.MAX_CONCURRENT_TASKS

    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    async def sem_task(idx: int, text: str):
        async with semaphore:
            result = await analyze_single_text(text, mode)
        return idx, result

    tasks = [
        asyncio.create_task(sem_task(idx, text))
        for idx, text in enumerate(texts_to_analyze)
    ]

    completed_results = [None] * len(texts_to_analyze)
    total = len(tasks)
    finished = 0

    # タスクが完了するたびに結果を格納して進捗を更新
    for coro in asyncio.as_completed(tasks):
        idx, result = await coro
        completed_results[idx] = result
        finished += 1
        if progress_callback:
            progress_callback(finished / total * 100)

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
async def summarize_results(
    df_analyzed: pd.DataFrame,
    column_name: str,
    wordcloud_type: str = "normal",
):
    """Summarize analyzed DataFrame for reporting.

    Args:
        df_analyzed: DataFrame including analysis results.
        column_name: Original text column used for analysis.
        wordcloud_type: Which subset of texts to use for the word cloud.
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
    # 指定された列のオリジナルテキストを使用してワードクラウドを生成
    if column_name in df_analyzed.columns:
        if wordcloud_type == "normal":
            texts = df_analyzed[column_name].dropna().astype(str).tolist()
        elif wordcloud_type == "positive":
            texts = (
                df_analyzed[df_analyzed["analysis_sentiment"].isin(["positive", "neutral"])]
                [column_name]
                .dropna()
                .astype(str)
                .tolist()
            )
        else:
            texts = (
                df_analyzed[df_analyzed["analysis_sentiment"].isin(["negative", "neutral"])]
                [column_name]
                .dropna()
                .astype(str)
                .tolist()
            )
    else:
        texts = df_analyzed["analysis_verbatim_quote"].dropna().astype(str).tolist()

    all_text = " ".join(texts)
    nlp = get_tokenizer("A")
    doc = nlp(all_text)
    words_for_wordcloud = [
        token.text
        for token in doc
        if token.pos_ in ["NOUN", "VERB", "ADJ"] and len(token.text) > 1
    ]

    summary = {
        "sentiment_counts": sentiment_counts,
        "topic_counts": topic_counts.head(15),
        "moderation_summary": moderation_summary,
        "emotion_avg": emotion_avg,
        "analysis_target": f"「{column_name}」列の回答",
    }

    commentary = await generate_report_commentary(summary)
    summary.update(commentary.model_dump())

    return summary, words_for_wordcloud
