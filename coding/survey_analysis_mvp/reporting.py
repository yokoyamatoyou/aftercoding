"""Utility functions for generating PDF reports and visualizations."""

import matplotlib.pyplot as plt
import matplotlib as mpl
import io
import base64
import os
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from fpdf import FPDF, HTMLMixin, __version__ as fpdf_version
from packaging import version
from pathlib import Path
from wordcloud import WordCloud
import warnings

if version.parse(fpdf_version).major < 2:
    raise ImportError(
        f"fpdf2>=2 is required but version {fpdf_version} was found.\n"
        "Please uninstall the old 'fpdf' package and install 'fpdf2'."
    )


class PDF(FPDF, HTMLMixin):
    """Custom PDF class with basic HTML support."""

    pass


import math
from typing import Optional

# フォントファイルのパス (TTF を優先)
FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FONT_PATH_TTF = os.path.join(FONT_DIR, "NotoSansJP-Regular.ttf")
FONT_PATH_OTF = os.path.join(FONT_DIR, "NotoSansJP-Regular.otf")
FONT_PATH = FONT_PATH_TTF if os.path.exists(FONT_PATH_TTF) else FONT_PATH_OTF

# Boldフォントのパス
BOLD_FONT_PATH_TTF = os.path.join(FONT_DIR, "NotoSansJP-Bold.ttf")
BOLD_FONT_PATH_OTF = os.path.join(FONT_DIR, "NotoSansJP-Bold.otf")
BOLD_FONT_PATH = (
    BOLD_FONT_PATH_TTF if os.path.exists(BOLD_FONT_PATH_TTF) else BOLD_FONT_PATH_OTF
)


def _find_font(candidates) -> Optional[str]:
    """候補リストから利用可能なフォントパスを返す"""
    found_ttc = False
    for path in candidates:
        if not os.path.exists(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in (".ttf", ".otf"):
            return path
        if ext == ".ttc":
            found_ttc = True

    for name in ["MS Gothic", "Meiryo", "Yu Gothic", "Noto Sans CJK JP"]:
        try:
            path = mpl.font_manager.findfont(name)
            if os.path.exists(path):
                ext = os.path.splitext(path)[1].lower()
                if ext in (".ttf", ".otf"):
                    return path
                if ext == ".ttc":
                    found_ttc = True
        except Exception:
            continue

    if found_ttc:
        warnings.warn(
            "Only .ttc fonts were found. These may not work with some PDF libraries."
        )

    return None


def find_japanese_fonts() -> tuple[Optional[str], Optional[str]]:
    """Return available regular and bold Japanese font paths.

    Returns:
        Tuple of paths for regular and bold fonts. ``None`` is returned if a
        suitable font could not be located.
    """

    regular = _find_font(
        [
            FONT_PATH,
            os.path.join(os.getenv("WINDIR", "C:\\Windows"), "Fonts", "meiryo.ttc"),
            os.path.join(os.getenv("WINDIR", "C:\\Windows"), "Fonts", "msgothic.ttc"),
            os.path.join(os.getenv("WINDIR", "C:\\Windows"), "Fonts", "YuGothR.ttc"),
            os.path.join(os.getenv("WINDIR", "C:\\Windows"), "Fonts", "YuGothM.ttc"),
        ]
    )

    bold = _find_font([BOLD_FONT_PATH]) if BOLD_FONT_PATH else None
    return regular, bold


AVAILABLE_FONT_PATH, AVAILABLE_BOLD_FONT_PATH = find_japanese_fonts()


def set_japanese_font() -> bool:
    """Configure matplotlib to use a Japanese font.

    Returns:
        ``True`` if a font was set successfully, otherwise ``False``.
    """
    if not AVAILABLE_FONT_PATH:
        mpl.rcParams["axes.unicode_minus"] = False
        return False

    try:
        mpl.font_manager.fontManager.addfont(AVAILABLE_FONT_PATH)
        font_name = mpl.font_manager.FontProperties(
            fname=AVAILABLE_FONT_PATH
        ).get_name()
        mpl.rcParams["font.family"] = font_name
        mpl.rcParams["font.sans-serif"] = [font_name]
    except Exception:
        return False

    mpl.rcParams["axes.unicode_minus"] = False
    return True


def create_sentiment_pie_chart_base64(sentiment_counts: pd.Series) -> str:
    """Create a sentiment pie chart as a Base64 string.

    Args:
        sentiment_counts: Series mapping sentiment labels to counts.

    Returns:
        PNG image data encoded in Base64. Empty string if no data.
    """
    if not set_japanese_font() or sentiment_counts.empty:
        return ""

    fig, ax = plt.subplots()
    ax.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4CAF50", "#FFC107", "#F44336", "#9E9E9E"],
    )
    ax.axis("equal")
    ax.set_title("感情分析サマリー")

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_topics_bar_chart_base64(topic_counts: pd.Series) -> str:
    """Create a horizontal bar chart of topic frequencies.

    Args:
        topic_counts: Series of topic counts.

    Returns:
        PNG image data encoded in Base64. Empty string if no data.
    """
    if not set_japanese_font() or topic_counts.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 8))
    topic_counts.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("主要トピック Top 15")
    ax.set_xlabel("出現回数")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_moderation_bar_chart_base64(moderation_summary: dict) -> str:
    """Create a bar chart summarizing moderation results.

    Args:
        moderation_summary: Mapping of moderation category to count.

    Returns:
        PNG image data encoded in Base64. Empty string if no data.
    """
    if not set_japanese_font() or not moderation_summary:
        return ""

    labels = list(moderation_summary.keys())
    values = list(moderation_summary.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color="skyblue")
    ax.set_title("モデレーション結果サマリー")
    ax.set_ylabel("フラグ数")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_emotion_radar_chart_base64(emotion_avg: dict) -> str:
    """Create a radar chart of average emotion scores.

    Args:
        emotion_avg: Mapping of emotion label to average score.

    Returns:
        PNG image data encoded in Base64. Empty string if no data.
    """
    if not set_japanese_font() or not emotion_avg:
        return ""

    labels = list(emotion_avg.keys())
    stats = list(emotion_avg.values())

    # レーダーチャートの準備
    angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
    angles += angles[:1]  # 閉じたグラフにするため最初の要素を最後に追加
    stats += stats[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color="red", alpha=0.25)
    ax.plot(angles, stats, color="red", linewidth=2)
    ax.set_yticklabels([])  # Y軸のラベルを非表示
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 5)  # スコアの範囲
    ax.set_title("感情スコア平均", va="bottom")

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_pdf_report(summary_data: dict, output_path: str):
    """Generate a PDF report from aggregated data.

    Args:
        summary_data: Dictionary produced by :func:`summarize_results`.
        output_path: Destination file path for the generated PDF.
    """
    if not os.path.exists(FONT_PATH):
        raise FileNotFoundError(
            f"Required font file not found: {FONT_PATH}. Please place NotoSansJP-Regular.ttf in the fonts directory before generating PDFs."
        )
    sentiment_chart = create_sentiment_pie_chart_base64(
        summary_data.get("sentiment_counts", pd.Series())
    )
    topics_chart = create_topics_bar_chart_base64(
        summary_data.get("topic_counts", pd.Series())
    )
    moderation_chart = create_moderation_bar_chart_base64(
        summary_data.get("moderation_summary", {})
    )
    emotion_chart = create_emotion_radar_chart_base64(
        summary_data.get("emotion_avg", {})
    )

    template_dir = Path(__file__).parent
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.html")
    font_uri = (
        Path(AVAILABLE_FONT_PATH).resolve().as_uri() if AVAILABLE_FONT_PATH else ""
    )
    html_out = template.render(
        sentiment_chart_base64=sentiment_chart,
        topics_chart_base64=topics_chart,
        moderation_chart_base64=moderation_chart,
        emotion_chart_base64=emotion_chart,
        topic_table=summary_data.get("topic_counts", pd.Series())
        .to_frame()
        .to_html(header=False),
        font_path=font_uri,
    )

    pdf = PDF()
    pdf.add_page()
    if AVAILABLE_FONT_PATH:
        pdf.add_font("NotoSansJP", "", AVAILABLE_FONT_PATH, uni=True)
        pdf.set_font("NotoSansJP", "", 12)
    if os.path.exists(BOLD_FONT_PATH):
        pdf.add_font("NotoSansJP", "B", BOLD_FONT_PATH, uni=True)
    pdf.write_html(html_out)
    pdf.output(output_path)
    print(f"PDFレポートが '{output_path}' として生成されました。")


def generate_wordcloud(words: list, output_path: str):
    """Generate and save a word cloud image.

    Args:
        words: List of words to visualize.
        output_path: Destination file path for the PNG image.
    """
    if not words:
        print("ワードクラウドを生成するための単語がありません。")
        return

    font_path = AVAILABLE_FONT_PATH
    if not font_path:
        print("日本語フォントが見つからないため、ワードクラウドを生成できません。")
        return

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        font_path=font_path,
        collocations=False,  # 単語のペアを考慮しない
    ).generate(" ".join(words))

    wordcloud.to_file(output_path)
    print(f"ワードクラウドが '{output_path}' として保存されました。")
