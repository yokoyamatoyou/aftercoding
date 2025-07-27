"""Utility functions for generating PDF reports and visualizations."""

import base64
import io
import os


import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML, CSS


FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FONT_REGULAR_PATH = os.path.join(FONT_DIR, "NotoSansJP-Regular.otf")
FONT_BOLD_PATH = os.path.join(FONT_DIR, "NotoSansJP-Bold.otf")




def set_japanese_font() -> bool:
    """Configure matplotlib to use a Japanese font."""

    if not os.path.exists(FONT_REGULAR_PATH):
        mpl.rcParams["axes.unicode_minus"] = False
        return False

    try:
        mpl.font_manager.fontManager.addfont(FONT_REGULAR_PATH)
        font_name = mpl.font_manager.FontProperties(fname=FONT_REGULAR_PATH).get_name()
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


def generate_pdf_report(summary_data: dict, output_path: str) -> None:
    """Generate a PDF report using WeasyPrint.

    This function mirrors the old ``fpdf2`` implementation but renders the
    report from HTML/CSS using WeasyPrint.
    """

    if not os.path.exists(FONT_REGULAR_PATH):
        raise FileNotFoundError(
            "Required font file not found. Please place NotoSansJP fonts in the fonts directory before generating PDFs."
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

    topic_table = (
        summary_data.get("topic_counts", pd.Series())
        .to_frame(name="Count")
        .to_html()
    )

    env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
    template = env.from_string(
        """
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"UTF-8\">
  <link rel=\"stylesheet\" href=\"style.css\">
</head>
<body>
  <h1>アンケート分析レポート</h1>
  <h2>サマリー</h2>
  <p>{{ summary_text }}</p>
  {% if action_items %}
  <ul>
  {% for item in action_items %}<li>{{ item }}</li>{% endfor %}
  </ul>
  {% endif %}
  <h2>感情分析</h2>
  <img src=\"data:image/png;base64,{{ sentiment_chart }}\" alt=\"感情分析グラフ\">
  <p>{{ sentiment_commentary }}</p>
  <h2>主要トピック</h2>
  <img src=\"data:image/png;base64,{{ topics_chart }}\" alt=\"主要トピックグラフ\">
  <p>{{ topics_commentary }}</p>
  <h2>トピック一覧</h2>
  {{ topic_table|safe }}
  {% if moderation_chart %}
  <h2>モデレーション結果</h2>
  <img src=\"data:image/png;base64,{{ moderation_chart }}\" alt=\"モデレーション結果\">
  {% endif %}
</body>
</html>
"""
    )

    html_out = template.render(
        summary_text=summary_data.get("summary_text", ""),
        action_items=summary_data.get("action_items", []),
        sentiment_commentary=summary_data.get("sentiment_commentary", ""),
        topics_commentary=summary_data.get("topics_commentary", ""),
        sentiment_chart=sentiment_chart,
        topics_chart=topics_chart,
        moderation_chart=moderation_chart,
        topic_table=topic_table,
    )

    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    HTML(string=html_out, base_url=os.path.dirname(__file__)).write_pdf(
        output_path, stylesheets=[CSS(css_path)]
    )
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

    font_path = FONT_REGULAR_PATH if os.path.exists(FONT_REGULAR_PATH) else None
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


def create_report(
    df: pd.DataFrame,
    positive_summary: str,
    negative_summary: str,
    wordcloud_type: str,
    text_column: str,
):
    """Generate charts, word clouds and a PDF report using WeasyPrint.

    Args:
        df: Analyzed DataFrame containing a ``sentiment`` column and original text.
        positive_summary: Summary text for positive comments.
        negative_summary: Summary text for negative comments.
        wordcloud_type: "normal", "positive", or "negative" to choose the
            source text for word clouds.
        text_column: Name of the column containing original text responses.
    """

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 1. sentiment chart
    chart_path = os.path.join(output_dir, "sentiment_chart.png")
    sentiment_counts = (
        df["sentiment"]
        .value_counts()
        .reindex(
            [
                "positive",
                "neutral",
                "negative",
            ],
            fill_value=0,
        )
    )
    plt.figure()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="pastel")
    plt.ylabel("Count")
    plt.title("Sentiment Distribution")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    # 2. word cloud
    if wordcloud_type == "normal":
        texts = df[text_column].fillna("").astype(str)
    elif wordcloud_type == "positive":
        texts = (
            df[df["sentiment"].isin(["positive", "neutral"])][text_column]
            .fillna("")
            .astype(str)
        )
    else:
        texts = (
            df[df["sentiment"].isin(["negative", "neutral"])][text_column]
            .fillna("")
            .astype(str)
        )

    wc = WordCloud(
        width=800, height=400, background_color="white", font_path=FONT_REGULAR_PATH
    )
    wc.generate(" ".join(texts))
    wc_path = os.path.join(output_dir, "wordcloud.png")
    wc.to_file(wc_path)

    # 3. render HTML via Jinja2
    env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
    template = env.get_template("report_template.html")
    context = {
        "positive_summary": positive_summary,
        "negative_summary": negative_summary,
        "sentiment_chart_path": chart_path,
        "wordcloud_path": wc_path,
        "total_count": len(df),
    }
    html_out = template.render(**context)

    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    pdf_path = os.path.join(output_dir, "survey_report.pdf")
    HTML(string=html_out, base_url=os.path.dirname(__file__)).write_pdf(
        pdf_path, stylesheets=[CSS(css_path)]
    )
    print(f"PDFレポートが '{pdf_path}' として生成されました。")
