import matplotlib.pyplot as plt
import matplotlib as mpl
import io
import base64
import os
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from xhtml2pdf import pisa
from wordcloud import WordCloud
import warnings

import math
from typing import Optional

# フォントファイルのパス (プロジェクト同梱フォント)
FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "NotoSansJP-Regular.otf")


def find_japanese_font() -> Optional[str]:
    """可能な日本語フォントファイルのパスを返す"""
    candidates = [
        FONT_PATH,
        os.path.join(os.getenv("WINDIR", "C:\\Windows"), "Fonts", "meiryo.ttc"),
        os.path.join(os.getenv("WINDIR", "C:\\Windows"), "Fonts", "msgothic.ttc"),
        os.path.join(os.getenv("WINDIR", "C:\\Windows"), "Fonts", "YuGothR.ttc"),
        os.path.join(os.getenv("WINDIR", "C:\\Windows"), "Fonts", "YuGothM.ttc"),
    ]

    found_ttc = False
    for path in candidates:
        if not os.path.exists(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in (".ttf", ".otf"):
            return path
        if ext == ".ttc":
            found_ttc = True

    # matplotlib経由で検索
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
        warnings.warn("Only .ttc fonts were found. These may not work with xhtml2pdf.")

    return None


AVAILABLE_FONT_PATH = find_japanese_font()


def set_japanese_font() -> bool:
    """matplotlibに日本語フォントを設定する"""
    if not AVAILABLE_FONT_PATH:
        mpl.rcParams['axes.unicode_minus'] = False
        return False

    try:
        mpl.font_manager.fontManager.addfont(AVAILABLE_FONT_PATH)
        font_name = mpl.font_manager.FontProperties(fname=AVAILABLE_FONT_PATH).get_name()
        mpl.rcParams['font.family'] = font_name
        mpl.rcParams['font.sans-serif'] = [font_name]
    except Exception:
        return False

    mpl.rcParams['axes.unicode_minus'] = False
    return True


def create_sentiment_pie_chart_base64(sentiment_counts: pd.Series) -> str:
    """感情の割合から円グラフを生成し、Base64エンコードされたPNG文字列を返す"""
    if not set_japanese_font() or sentiment_counts.empty:
        return ""

    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FFC107', '#F44336', '#9E9E9E'])
    ax.axis('equal')
    ax.set_title('感情分析サマリー')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    plt.close(fig)

    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def create_topics_bar_chart_base64(topic_counts: pd.Series) -> str:
    """トピックの出現頻度から棒グラフを生成し、Base64エンコードされたPNG文字列を返す"""
    if not set_japanese_font() or topic_counts.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 8))
    topic_counts.sort_values().plot(kind='barh', ax=ax)
    ax.set_title('主要トピック Top 15')
    ax.set_xlabel('出現回数')
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)

    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def create_moderation_bar_chart_base64(moderation_summary: dict) -> str:
    """モデレーション結果から棒グラフを生成し、Base64エンコードされたPNG文字列を返す"""
    if not set_japanese_font() or not moderation_summary:
        return ""

    labels = list(moderation_summary.keys())
    values = list(moderation_summary.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color='skyblue')
    ax.set_title('モデレーション結果サマリー')
    ax.set_ylabel('フラグ数')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    plt.close(fig)

    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def create_emotion_radar_chart_base64(emotion_avg: dict) -> str:
    """感情スコアの平均からレーダーチャートを生成し、Base64エンコードされたPNG文字列を返す"""
    if not set_japanese_font() or not emotion_avg:
        return ""

    labels = list(emotion_avg.keys())
    stats = list(emotion_avg.values())

    # レーダーチャートの準備
    angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
    angles += angles[:1] # 閉じたグラフにするため最初の要素を最後に追加
    stats += stats[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='red', alpha=0.25)
    ax.plot(angles, stats, color='red', linewidth=2)
    ax.set_yticklabels([]) # Y軸のラベルを非表示
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 5) # スコアの範囲
    ax.set_title('感情スコア平均', va='bottom')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    plt.close(fig)

    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def generate_pdf_report(summary_data: dict, output_path: str):
    """集計データからPDFレポートを生成する"""
    sentiment_chart = create_sentiment_pie_chart_base64(
        summary_data.get('sentiment_counts', pd.Series())
    )
    topics_chart = create_topics_bar_chart_base64(
        summary_data.get('topic_counts', pd.Series())
    )
    moderation_chart = create_moderation_bar_chart_base64(
        summary_data.get('moderation_summary', {})
    )
    emotion_chart = create_emotion_radar_chart_base64(
        summary_data.get('emotion_avg', {})
    )

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('report_template.html')
    html_out = template.render(
        sentiment_chart_base64=sentiment_chart,
        topics_chart_base64=topics_chart,
        moderation_chart_base64=moderation_chart,
        emotion_chart_base64=emotion_chart,
        topic_table=summary_data.get('topic_counts', pd.Series()).to_frame().to_html(header=False),
        font_path=AVAILABLE_FONT_PATH or ""  # xhtml2pdf用に絶対パスを渡す
    )

    with open(output_path, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(
            src=html_out,
            dest=result_file,
            encoding='utf-8'
        )

    if not pisa_status.err:
        print(f"PDFレポートが '{output_path}' として生成されました。")
    else:
        print(f"PDFの生成中にエラーが発生しました: {pisa_status.err}")
        raise IOError("PDF生成に失敗しました。")

def generate_wordcloud(words: list, output_path: str):
    """単語リストからワードクラウド画像を生成・保存する"""
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
        background_color='white',
        font_path=font_path,
        collocations=False # 単語のペアを考慮しない
    ).generate(' '.join(words))

    wordcloud.to_file(output_path)
    print(f"ワードクラウドが '{output_path}' として保存されました。")