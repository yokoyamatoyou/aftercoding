"""PDF report generation and visualization utilities using fpdf2."""

from __future__ import annotations

import base64
import io
import os
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
from wordcloud import WordCloud

from analysis import get_tokenizer

# --- Constants ---------------------------------------------------------------
A4_WIDTH = 210
A4_HEIGHT = 297
MARGIN = 15

COLOR_PRIMARY = (44, 62, 80)  # #2c3e50
COLOR_SECONDARY = (52, 152, 219)  # #3498db
COLOR_TEXT = (51, 51, 51)  # #333333
COLOR_LIGHT_GRAY = (242, 242, 242)  # #f2f2f2

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FONT_REGULAR_PATH = os.path.join(FONT_DIR, "NotoSansJP-Regular.otf")
FONT_BOLD_PATH = os.path.join(FONT_DIR, "NotoSansJP-Bold.otf")
_FONT_CONFIGURED = False


# --- Matplotlib helper ------------------------------------------------------


def set_japanese_font() -> bool:
    """Configure matplotlib to use bundled Japanese fonts only once."""
    global _FONT_CONFIGURED
    if _FONT_CONFIGURED:
        return True
    if not os.path.exists(FONT_REGULAR_PATH):
        mpl.rcParams["axes.unicode_minus"] = False
        return False

    try:
        mpl.font_manager.fontManager.addfont(FONT_REGULAR_PATH)
        font_name = mpl.font_manager.FontProperties(fname=FONT_REGULAR_PATH).get_name()
        mpl.rcParams["font.family"] = font_name
        mpl.rcParams["font.sans-serif"] = [font_name]
        _FONT_CONFIGURED = True
    except Exception:
        return False

    mpl.rcParams["axes.unicode_minus"] = False
    return True


# --- Chart generation -------------------------------------------------------


def create_sentiment_pie_chart_base64(sentiment_counts: pd.Series) -> str:
    """Return a base64 PNG string of the sentiment distribution pie chart."""
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

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def create_topics_bar_chart_base64(topic_counts: pd.Series) -> str:
    """Return a base64 PNG string of the top topics bar chart."""
    if not set_japanese_font() or topic_counts.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 8))
    topic_counts.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("主要トピック Top 15")
    ax.set_xlabel("出現回数")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def create_moderation_bar_chart_base64(moderation_summary: dict[str, int]) -> str:
    """Return a base64 PNG string of the moderation summary bar chart."""
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

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --- PDF generation ---------------------------------------------------------


class ReportPDF(FPDF):
    """Custom PDF class for 5-page survey reports."""

    def header(self) -> None:  # pragma: no cover - simple header
        pass

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("NotoSansJP", "", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def setup_fonts(self) -> None:
        """Register Japanese fonts."""
        if not (os.path.exists(FONT_REGULAR_PATH) and os.path.exists(FONT_BOLD_PATH)):
            raise FileNotFoundError(
                "NotoSansJPフォントファイルが見つかりません。fontsディレクトリを確認してください。"
            )
        self.add_font("NotoSansJP", "", FONT_REGULAR_PATH, uni=True)
        self.add_font("NotoSansJP", "B", FONT_BOLD_PATH, uni=True)

    # Page builders ------------------------------------------------------
    def create_cover_page(self, analysis_target: str = "（分析対象未設定）") -> None:
        self.add_page()
        self.set_fill_color(*COLOR_PRIMARY)
        self.rect(0, 0, A4_WIDTH, A4_HEIGHT, "F")

        self.set_y(A4_HEIGHT / 3)
        self.set_font("NotoSansJP", "B", 24)
        self.set_text_color(255, 255, 255)
        self.multi_cell(0, 12, "顧客インサイト分析レポート", 0, "C")
        self.ln(10)

        self.set_font("NotoSansJP", "", 14)
        self.multi_cell(0, 10, f"分析対象：{analysis_target}", 0, "C")
        self.ln(20)

        today = datetime.now().strftime("%Y年%m月%d日")
        self.set_font("NotoSansJP", "", 12)
        self.cell(0, 10, f"レポート作成日: {today}", 0, 0, "C")

    def create_summary_page(self, summary_text: str, action_items: list[str]) -> None:
        self.add_page()
        self.set_text_color(*COLOR_TEXT)

        self.set_font("NotoSansJP", "B", 18)
        self.cell(0, 15, "エグゼクティブサマリー", 0, 1, "L")
        self.ln(5)

        self.set_font("NotoSansJP", "B", 12)
        self.cell(0, 10, "■ 分析結果の総括", 0, 1, "L")
        self.set_font("NotoSansJP", "", 10)
        self.multi_cell(0, 7, summary_text, 0, "L")
        self.ln(10)

        self.set_font("NotoSansJP", "B", 12)
        self.cell(0, 10, "■ 推奨されるネクストアクション", 0, 1, "L")
        self.set_font("NotoSansJP", "", 10)
        for item in action_items:
            self.multi_cell(0, 7, f"・ {item}", 0, "L")

    def create_chart_commentary_page(
        self,
        title: str,
        chart_base64: str,
        commentary_text: str,
        chart_width: int = 160,
    ) -> None:
        self.add_page()
        self.set_text_color(*COLOR_TEXT)

        self.set_font("NotoSansJP", "B", 18)
        self.cell(0, 15, title, 0, 1, "L")
        self.ln(5)

        if chart_base64:
            chart_image = io.BytesIO(base64.b64decode(chart_base64))
            x_pos = (A4_WIDTH - chart_width) / 2
            self.image(chart_image, x=x_pos, w=chart_width)
            self.ln(5)

        self.set_font("NotoSansJP", "B", 12)
        self.cell(0, 10, "■ 分析からの示唆", 0, 1, "L")
        self.set_font("NotoSansJP", "", 10)
        self.set_x(MARGIN)
        self.multi_cell(A4_WIDTH - MARGIN * 2, 7, commentary_text, 0, "L")

    def create_appendix_page(self, topic_counts_df: pd.DataFrame) -> None:
        self.add_page()
        self.set_text_color(*COLOR_TEXT)

        self.set_font("NotoSansJP", "B", 18)
        self.cell(0, 15, "付録：データ詳細", 0, 1, "L")
        self.ln(5)

        self.set_font("NotoSansJP", "B", 12)
        self.cell(0, 10, "■ 全トピック一覧", 0, 1, "L")

        self.set_font("NotoSansJP", "B", 10)
        self.cell(120, 8, "トピック", 1, 0, "C")
        self.cell(40, 8, "出現回数", 1, 1, "C")

        self.set_font("NotoSansJP", "", 10)
        for index, row in topic_counts_df.iterrows():
            self.cell(120, 8, f"  {index}", 1, 0, "L")
            self.cell(40, 8, str(row.values[0]), 1, 1, "C")

    def create_wordcloud_page(self, pos_wc: str | None, neg_wc: str | None) -> None:
        """Add a page containing positive and negative word cloud images."""
        if not (pos_wc or neg_wc):
            return

        self.add_page()
        self.set_text_color(*COLOR_TEXT)

        self.set_font("NotoSansJP", "B", 18)
        self.cell(0, 15, "ワードクラウド", 0, 1, "L")
        self.ln(5)

        if pos_wc:
            self.set_font("NotoSansJP", "B", 12)
            self.cell(0, 10, "ポジティブ", 0, 1, "L")
            self.image(pos_wc, x=(A4_WIDTH - 160) / 2, w=160)
            self.ln(10)

        if neg_wc:
            self.set_font("NotoSansJP", "B", 12)
            self.cell(0, 10, "ネガティブ", 0, 1, "L")
            self.image(neg_wc, x=(A4_WIDTH - 160) / 2, w=160)


# --- Entry point ------------------------------------------------------------


def generate_pdf_report(
    summary_data: dict,
    output_path: str,
    pos_wc: str | None = None,
    neg_wc: str | None = None,
) -> None:
    """Generate a multi-page PDF report from analysis results.

    Args:
        summary_data: Dictionary containing chart data and commentary.
        output_path: Destination PDF file path.
        pos_wc: Path to the positive word cloud image.
        neg_wc: Path to the negative word cloud image.
    """
    pdf = ReportPDF()
    pdf.setup_fonts()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.create_cover_page(
        analysis_target=summary_data.get("analysis_target", "アンケート回答")
    )

    pdf.create_summary_page(
        summary_text=summary_data.get("summary_text", "総括テキストがありません。"),
        action_items=summary_data.get(
            "action_items", ["アクションアイテムがありません。"]
        ),
    )

    sentiment_chart = create_sentiment_pie_chart_base64(
        summary_data.get("sentiment_counts", pd.Series())
    )
    pdf.create_chart_commentary_page(
        title="分析詳細①：全体感情分析",
        chart_base64=sentiment_chart,
        commentary_text=summary_data.get("sentiment_commentary", "解説がありません。"),
        chart_width=120,
    )

    topics_chart = create_topics_bar_chart_base64(
        summary_data.get("topic_counts", pd.Series())
    )
    pdf.create_chart_commentary_page(
        title="分析詳細②：主要トピック",
        chart_base64=topics_chart,
        commentary_text=summary_data.get("topics_commentary", "解説がありません。"),
        chart_width=180,
    )

    pdf.create_wordcloud_page(pos_wc, neg_wc)

    topic_df = summary_data.get("topic_counts", pd.Series()).to_frame(name="Count")
    pdf.create_appendix_page(topic_df)

    pdf.output(output_path)
    print(f"新しいデザインのPDFレポートが '{output_path}' として生成されました。")


def generate_wordcloud(words: list[str], output_path: str) -> None:
    """Generate and save a word cloud image."""
    if not words:
        print("ワードクラウドを生成するための単語がありません。")
        return

    font_path = FONT_REGULAR_PATH if os.path.exists(FONT_REGULAR_PATH) else None
    if not font_path:
        print("日本語フォントが見つからないため、ワードクラウドを生成できません。")
        return

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        font_path=font_path,
        collocations=False,
    ).generate(" ".join(words))

    wc.to_file(output_path)
    print(f"ワードクラウドが '{output_path}' として保存されました。")


def create_report(
    df: pd.DataFrame,
    positive_summary: str,
    negative_summary: str,
    wordcloud_type: str,
    column_name: str,
) -> None:
    """Generate charts, word clouds and a PDF report from survey data."""

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # --- Sentiment chart -------------------------------------------------
    counts = (
        df["sentiment"]
        .value_counts()
        .reindex(["positive", "neutral", "negative"], fill_value=0)
    )
    chart_base64 = create_sentiment_pie_chart_base64(counts)
    if chart_base64:
        chart_path = os.path.join(output_dir, "sentiment_chart.png")
        with open(chart_path, "wb") as f:
            f.write(base64.b64decode(chart_base64))

    # --- Word cloud ------------------------------------------------------
    def tokenize(texts: list[str]) -> list[str]:
        nlp = get_tokenizer("A")
        doc = nlp(" ".join(texts))
        return [
            t.text for t in doc if t.pos_ in ["NOUN", "VERB", "ADJ"] and len(t.text) > 1
        ]

    if wordcloud_type == "normal":
        texts = df[column_name].dropna().astype(str).tolist()
        words = tokenize(texts)
        generate_wordcloud(words, os.path.join(output_dir, "wordcloud.png"))
        pos_wc = neg_wc = None
    else:
        pos_texts = (
            df[df["sentiment"].isin(["positive", "neutral"])][column_name]
            .dropna()
            .astype(str)
            .tolist()
        )
        neg_texts = (
            df[df["sentiment"].isin(["negative", "neutral"])][column_name]
            .dropna()
            .astype(str)
            .tolist()
        )
        generate_wordcloud(
            tokenize(pos_texts), os.path.join(output_dir, "positive_wordcloud.png")
        )
        generate_wordcloud(
            tokenize(neg_texts), os.path.join(output_dir, "negative_wordcloud.png")
        )
        pos_wc = os.path.join(output_dir, "positive_wordcloud.png")
        neg_wc = os.path.join(output_dir, "negative_wordcloud.png")

    # --- PDF report ------------------------------------------------------
    # Combine positive and negative summaries for the overall summary text
    summary_text = "\n\n".join(
        s.strip() for s in [positive_summary, negative_summary] if s.strip()
    )

    # Derive action items from the negative summary lines
    action_items = [
        line.strip("・- ") for line in negative_summary.splitlines() if line.strip()
    ]

    summary = {
        "analysis_target": f"「{column_name}」列の回答",
        "summary_text": summary_text,
        "action_items": action_items or ["アクションアイテムがありません。"],
        "sentiment_counts": counts,
        "pos_wc": pos_wc,
        "neg_wc": neg_wc,
    }
    if chart_base64:
        summary["sentiment_commentary"] = ""
    generate_pdf_report(
        summary,
        os.path.join(output_dir, "survey_report.pdf"),
        pos_wc,
        neg_wc,
    )
