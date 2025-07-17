"""Utility functions for generating PDF reports and visualizations."""

import base64
import io
import os
from datetime import datetime


import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF, __version__ as fpdf_version
from packaging import version
from wordcloud import WordCloud

if version.parse(fpdf_version).major < 2:
    raise ImportError(
        f"fpdf2>=2 is required but version {fpdf_version} was found.\n"
        "Please uninstall the old 'fpdf' package and install 'fpdf2'."
    )



# --- 定数定義 ---
A4_WIDTH = 210
A4_HEIGHT = 297
MARGIN = 15

# カラーコード
COLOR_PRIMARY = (44, 62, 80)  # ネイビー (#2c3e50)
COLOR_SECONDARY = (52, 152, 219)  # ブルー (#3498db)
COLOR_TEXT = (51, 51, 51)  # ダークグレー (#333333)
COLOR_LIGHT_GRAY = (242, 242, 242)  # ライトグレー (#f2f2f2)

# フォントパス
FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FONT_REGULAR_PATH = os.path.join(FONT_DIR, "NotoSansJP-Regular.otf")
FONT_BOLD_PATH = os.path.join(FONT_DIR, "NotoSansJP-Bold.otf")


class ReportPDF(FPDF):
    """PDF generator with manual layout for survey reports."""

    def header(self) -> None:
        """Override to draw headers on each page."""
        pass

    def footer(self) -> None:
        """Draw a centered page number in the footer."""
        self.set_y(-15)
        self.set_font("NotoSansJP", "", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def setup_fonts(self) -> None:
        """Register Japanese fonts for the PDF."""
        if not os.path.exists(FONT_REGULAR_PATH) or not os.path.exists(FONT_BOLD_PATH):
            raise FileNotFoundError(
                "NotoSansJPフォントファイルが見つかりません。fontsディレクトリを確認してください。"
            )
        self.add_font("NotoSansJP", "", FONT_REGULAR_PATH, uni=True)
        self.add_font("NotoSansJP", "B", FONT_BOLD_PATH, uni=True)

    # --- ここから各ページを作成するメソッドを追加していく ---

    def create_cover_page(self, analysis_target: str = "（分析対象未設定）") -> None:
        """Add the cover page to the report."""
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

    def create_summary_page(self, summary_text: str, action_items) -> None:
        """Add the executive summary page."""
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
            # multi_cell defaults to positioning the cursor at the end of
            # the written cell. This causes subsequent calls to fail due to
            # insufficient width. Explicitly reset the X position to the
            # left margin and move to the next line after each bullet.
            self.multi_cell(
                0,
                7,
                f"・ {item}",
                border=0,
                align="L",
                new_x="LMARGIN",
                new_y="NEXT",
            )

    def create_chart_commentary_page(
        self,
        title: str,
        chart_base64: str,
        commentary_text: str,
        chart_width: int = 160,
    ) -> None:
        """Add a chart page with commentary."""
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
        """Add an appendix page with the full topic table."""
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

def generate_pdf_report(summary_data: dict, output_path: str):
    """Generate a PDF report from aggregated data."""

    if not os.path.exists(FONT_REGULAR_PATH):
        raise FileNotFoundError(
            "Required font file not found. Please place NotoSansJP fonts in the fonts directory before generating PDFs."
        )

    pdf = ReportPDF()
    pdf.setup_fonts()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ページ1: 表紙
    pdf.create_cover_page(
        analysis_target=summary_data.get("analysis_target", "アンケート回答")
    )

    # ページ2: エグゼクティブサマリー
    pdf.create_summary_page(
        summary_text=summary_data.get("summary_text", "総括テキストがありません。"),
        action_items=summary_data.get(
            "action_items", ["アクションアイテムがありません。"]
        ),
    )

    # ページ3: 感情分析
    sentiment_chart = create_sentiment_pie_chart_base64(
        summary_data.get("sentiment_counts", pd.Series())
    )
    pdf.create_chart_commentary_page(
        title="分析詳細①：全体感情分析",
        chart_base64=sentiment_chart,
        commentary_text=summary_data.get("sentiment_commentary", "解説がありません。"),
        chart_width=120,
    )

    # ページ4: 主要トピック
    topics_chart = create_topics_bar_chart_base64(
        summary_data.get("topic_counts", pd.Series())
    )
    pdf.create_chart_commentary_page(
        title="分析詳細②：主要トピック",
        chart_base64=topics_chart,
        commentary_text=summary_data.get("topics_commentary", "解説がありません。"),
        chart_width=180,
    )

    # ページ5: 付録
    topic_df = summary_data.get("topic_counts", pd.Series()).to_frame(name="Count")
    pdf.create_appendix_page(topic_df)

    pdf.output(output_path)
    print(f"新しいデザインのPDFレポートが '{output_path}' として生成されました。")


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
