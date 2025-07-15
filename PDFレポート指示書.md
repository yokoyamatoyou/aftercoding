# **Codex向けPDFレポート改修タスク指示書**

## **1\. 最終目標**

現在のアンケート分析ツールを改修し、fpdf2ライブラリを使用して、洗練されたデザインの全5ページのPDFレポートを生成する機能を実装します。現在のwrite\_html()を使った方法は廃止し、座標ベースの精密なレイアウト制御に切り替えます。また、LLMを活用してレポート内の解説文を自動生成します。

## **2\. タスク1：reporting.py の完全改修**

**目的:** fpdf2を使い、1ページずつ手動でレイアウトを制御する新しいPDF生成クラスを実装する。

### **ステップ1: 必要なライブラリのインポート**

ファイルの先頭で、以下のライブラリがインポートされていることを確認してください。

import base64  
import io  
import os  
import pandas as pd  
import matplotlib.pyplot as plt  
import matplotlib as mpl  
from fpdf import FPDF  
from wordcloud import WordCloud  
from datetime import datetime

### **ステップ2: 定数の定義**

コードの上部に、レポート全体で使用するデザイン用の定数を定義してください。

\# \--- 定数定義 \---  
A4\_WIDTH \= 210  
A4\_HEIGHT \= 297  
MARGIN \= 15

\# カラーコード  
COLOR\_PRIMARY \= (44, 62, 80\)      \# ネイビー (\#2c3e50)  
COLOR\_SECONDARY \= (52, 152, 219\)   \# ブルー (\#3498db)  
COLOR\_TEXT \= (51, 51, 51\)          \# ダークグレー (\#333333)  
COLOR\_LIGHT\_GRAY \= (242, 242, 242\) \# ライトグレー (\#f2f2f2)

\# フォントパス  
FONT\_DIR \= os.path.join(os.path.dirname(\_\_file\_\_), "fonts")  
FONT\_REGULAR\_PATH \= os.path.join(FONT\_DIR, "NotoSansJP-Regular.otf")  
FONT\_BOLD\_PATH \= os.path.join(FONT\_DIR, "NotoSansJP-Bold.otf")

### **ステップ3: ReportPDF クラスの作成**

FPDFを継承したReportPDFクラスを新規に作成してください。このクラスが新しいPDF生成の核となります。

class ReportPDF(FPDF):  
    def header(self):  
        \# このメソッドは自動的に呼び出される  
        \# ロゴや固定のヘッダーをここに記述する（今回はシンプルに何もしない）  
        pass

    def footer(self):  
        \# このメソッドは自動的に呼び出される  
        self.set\_y(-15)  
        self.set\_font('NotoSansJP', '', 8\)  
        self.set\_text\_color(128)  
        self.cell(0, 10, f'Page {self.page\_no()}', 0, 0, 'C')

    def setup\_fonts(self):  
        """日本語フォントを登録する"""  
        if not os.path.exists(FONT\_REGULAR\_PATH) or not os.path.exists(FONT\_BOLD\_PATH):  
            raise FileNotFoundError("NotoSansJPフォントファイルが見つかりません。fontsディレクトリを確認してください。")  
        self.add\_font('NotoSansJP', '', FONT\_REGULAR\_PATH, uni=True)  
        self.add\_font('NotoSansJP', 'B', FONT\_BOLD\_PATH, uni=True)

    \# \--- ここから各ページを作成するメソッドを追加していく \---

### **ステップ4: 表紙ページ作成メソッドの追加**

ReportPDFクラス内に、レポートの1ページ目（表紙）を作成するメソッド create\_cover\_page を追加してください。

    def create\_cover\_page(self, analysis\_target="（分析対象未設定）"):  
        self.add\_page()  
          
        \# 背景色  
        self.set\_fill\_color(\*COLOR\_PRIMARY)  
        self.rect(0, 0, A4\_WIDTH, A4\_HEIGHT, 'F')

        \# メインタイトル  
        self.set\_y(A4\_HEIGHT / 3\)  
        self.set\_font('NotoSansJP', 'B', 24\)  
        self.set\_text\_color(255, 255, 255\)  
        self.multi\_cell(0, 12, "顧客インサイト分析レポート", 0, 'C')  
        self.ln(10)

        \# サブタイトル  
        self.set\_font('NotoSansJP', '', 14\)  
        self.multi\_cell(0, 10, f"分析対象：{analysis\_target}", 0, 'C')  
        self.ln(20)

        \# 日付  
        today \= datetime.now().strftime("%Y年%m月%d日")  
        self.set\_font('NotoSansJP', '', 12\)  
        self.cell(0, 10, f"レポート作成日: {today}", 0, 0, 'C')

### **ステップ5: エグゼクティブサマリーページ作成メソッドの追加**

ReportPDFクラス内に、2ページ目（エグゼクティブサマリー）を作成するメソッド create\_summary\_page を追加してください。

    def create\_summary\_page(self, summary\_text, action\_items):  
        self.add\_page()  
        self.set\_text\_color(\*COLOR\_TEXT)

        \# ページタイトル  
        self.set\_font('NotoSansJP', 'B', 18\)  
        self.cell(0, 15, "エグゼクティブサマリー", 0, 1, 'L')  
        self.ln(5)

        \# 総括  
        self.set\_font('NotoSansJP', 'B', 12\)  
        self.cell(0, 10, "■ 分析結果の総括", 0, 1, 'L')  
        self.set\_font('NotoSansJP', '', 10\)  
        self.multi\_cell(0, 7, summary\_text, 0, 'L')  
        self.ln(10)

        \# ネクストアクション  
        self.set\_font('NotoSansJP', 'B', 12\)  
        self.cell(0, 10, "■ 推奨されるネクストアクション", 0, 1, 'L')  
        self.set\_font('NotoSansJP', '', 10\)  
        for item in action\_items:  
            self.multi\_cell(0, 7, f"・ {item}", 0, 'L')

### **ステップ6: グラフと解説ページの作成メソッド追加**

ReportPDFクラス内に、グラフとLLMによる解説文を掲載する汎用メソッド create\_chart\_commentary\_page を追加してください。

    def create\_chart\_commentary\_page(self, title, chart\_base64, commentary\_text, chart\_width=160):  
        self.add\_page()  
        self.set\_text\_color(\*COLOR\_TEXT)

        \# ページタイトル  
        self.set\_font('NotoSansJP', 'B', 18\)  
        self.cell(0, 15, title, 0, 1, 'L')  
        self.ln(5)

        \# グラフ画像  
        if chart\_base64:  
            chart\_image \= io.BytesIO(base64.b64decode(chart\_base64))  
            x\_pos \= (A4\_WIDTH \- chart\_width) / 2  
            self.image(chart\_image, x=x\_pos, w=chart\_width)  
            self.ln(5)

        \# 解説文  
        self.set\_font('NotoSansJP', 'B', 12\)  
        self.cell(0, 10, "■ 分析からの示唆", 0, 1, 'L')  
        self.set\_font('NotoSansJP', '', 10\)  
        self.set\_x(MARGIN)  
        self.multi\_cell(A4\_WIDTH \- MARGIN \* 2, 7, commentary\_text, 0, 'L')

### **ステップ7: 付録ページ作成メソッドの追加**

ReportPDFクラス内に、最後の付録ページを作成するメソッド create\_appendix\_page を追加してください。

    def create\_appendix\_page(self, topic\_counts\_df):  
        self.add\_page()  
        self.set\_text\_color(\*COLOR\_TEXT)

        \# ページタイトル  
        self.set\_font('NotoSansJP', 'B', 18\)  
        self.cell(0, 15, "付録：データ詳細", 0, 1, 'L')  
        self.ln(5)

        \# トピック一覧テーブル  
        self.set\_font('NotoSansJP', 'B', 12\)  
        self.cell(0, 10, "■ 全トピック一覧", 0, 1, 'L')  
          
        self.set\_font('NotoSansJP', 'B', 10\)  
        self.cell(120, 8, "トピック", 1, 0, 'C')  
        self.cell(40, 8, "出現回数", 1, 1, 'C')

        self.set\_font('NotoSansJP', '', 10\)  
        for index, row in topic\_counts\_df.iterrows():  
            self.cell(120, 8, f"  {index}", 1, 0, 'L')  
            self.cell(40, 8, str(row.values\[0\]), 1, 1, 'C')

### **ステップ8: generate\_pdf\_report 関数の書き換え**

既存の generate\_pdf\_report 関数を、新しく作成した ReportPDF クラスを使うように全面的に書き換えてください。write\_html は使用しません。

def generate\_pdf\_report(summary\_data: dict, output\_path: str):  
    """  
    分析データから新しいデザインのPDFレポートを生成する。  
    """  
    pdf \= ReportPDF()  
    pdf.setup\_fonts()  
    pdf.set\_auto\_page\_break(auto=True, margin=15)

    \# ページ1: 表紙  
    pdf.create\_cover\_page(analysis\_target=summary\_data.get("analysis\_target", "アンケート回答"))

    \# ページ2: エグゼクティブサマリー  
    \# summary\_dataに 'summary\_text' と 'action\_items' が追加されることを想定  
    pdf.create\_summary\_page(  
        summary\_text=summary\_data.get("summary\_text", "総括テキストがありません。"),  
        action\_items=summary\_data.get("action\_items", \["アクションアイテムがありません。"\])  
    )

    \# ページ3: 感情分析  
    sentiment\_chart \= create\_sentiment\_pie\_chart\_base64(summary\_data.get("sentiment\_counts", pd.Series()))  
    pdf.create\_chart\_commentary\_page(  
        title="分析詳細①：全体感情分析",  
        chart\_base64=sentiment\_chart,  
        commentary\_text=summary\_data.get("sentiment\_commentary", "解説がありません。"),  
        chart\_width=120  
    )

    \# ページ4: 主要トピック  
    topics\_chart \= create\_topics\_bar\_chart\_base64(summary\_data.get("topic\_counts", pd.Series()))  
    pdf.create\_chart\_commentary\_page(  
        title="分析詳細②：主要トピック",  
        chart\_base64=topics\_chart,  
        commentary\_text=summary\_data.get("topics\_commentary", "解説がありません。"),  
        chart\_width=180  
    )

    \# ページ5: 付録  
    topic\_df \= summary\_data.get("topic\_counts", pd.Series()).to\_frame(name="Count")  
    pdf.create\_appendix\_page(topic\_df)  
      
    \# PDFファイルを出力  
    pdf.output(output\_path)  
    print(f"新しいデザインのPDFレポートが '{output\_path}' として生成されました。")

## **3\. タスク2：analysis.py の機能拡張**

**目的:** レポートに掲載する「総括」や「解説文」をLLMに生成させる機能を追加する。

### **ステップ1: 新しいPydanticモデルの追加**

LLMからの解説文出力を構造化するため、ReportCommentary モデルを定義してください。

class ReportCommentary(BaseModel):  
    """LLMによって生成されたレポートの解説文。"""  
    summary\_text: str \= Field(description="分析結果全体を3つのポイントで要約した総括文。")  
    action\_items: List\[str\] \= Field(description="分析結果から考えられる具体的なネクストアクションの提案リスト。3つ提案すること。")  
    sentiment\_commentary: str \= Field(description="感情分析の円グラフから読み取れるインサイトや注目点を解説する文章。")  
    topics\_commentary: str \= Field(description="主要トピックの棒グラフから読み取れるインサイトや注目点を解説する文章。")

### **ステップ2: LLM解説生成関数の追加**

summary\_data を受け取り、LLMに問い合わせて解説文を生成する非同期関数 generate\_report\_commentary を追加してください。

async def generate\_report\_commentary(summary\_data: dict) \-\> ReportCommentary:  
    """  
    集計済みデータに基づき、LLMにレポートの解説文を生成させる。  
    """  
    \# LLMに渡すためのコンテキスト情報を作成  
    context \= f"""  
    以下のアンケート分析結果データに基づき、プロのマーケティングアナリストとして、示唆に富んだレポート解説文を生成してください。

    \# 感情分析結果 (件数)  
    {summary\_data.get("sentiment\_counts", "データなし").to\_string()}

    \# 主要トピック Top 15 (件数)  
    {summary\_data.get("topic\_counts", "データなし").to\_string()}  
    """

    try:  
        commentary \= await aclient.chat.completions.create(  
            model="gpt-4o-mini",  
            response\_model=ReportCommentary,  
            messages=\[  
                {"role": "system", "content": "あなたは、データからインサイトを抽出し、分かりやすく解説する優秀なマーケティングアナリストです。"},  
                {"role": "user", "content": context},  
            \],  
            max\_retries=2,  
        )  
        return commentary  
    except Exception as e:  
        print(f"LLM解説生成エラー: {e}")  
        \# エラー時は空のモデルを返す  
        return ReportCommentary(  
            summary\_text="解説の生成中にエラーが発生しました。",  
            action\_items=\["N/A"\],  
            sentiment\_commentary="解説の生成中にエラーが発生しました。",  
            topics\_commentary="解説の生成中にエラーが発生しました。"  
        )

### **ステップ3: summarize\_results 関数の改修**

この関数は非同期(async def)に変更し、最後に generate\_report\_commentary を呼び出して、その結果を summary 辞書に統合するように修正してください。

\# summarize\_results を async def に変更  
async def summarize\_results(df\_analyzed: pd.DataFrame, column\_name: str):  
    \# (既存の集計処理はそのまま)  
    \# ...  
      
    summary \= {  
        "sentiment\_counts": sentiment\_counts,  
        "topic\_counts": topic\_counts.head(15),  
        "moderation\_summary": moderation\_summary,  
        "emotion\_avg": emotion\_avg,  
        "analysis\_target": f"「{column\_name}」列の回答"  
    }

    \# LLMによる解説文を生成して追加  
    commentary \= await generate\_report\_commentary(summary)  
    summary.update(commentary.model\_dump())

    return summary, words\_for\_wordcloud

## **4\. タスク3：main.py の修正**

**目的:** analysis.py の変更に合わせて、非同期処理の呼び出し方を修正する。

### **ステップ1: run\_analysis\_wrapper の修正**

summarize\_results が非同期関数になったため、run 関数内の呼び出し部分を修正してください。

\# main.py の run\_analysis\_wrapper 内の run 関数  
        async def run():  
            try:  
                self.df\_analyzed \= await analyze\_dataframe(self.df, column, progress\_callback=self.update\_progress)  
                \# summarize\_results の呼び出しを await にする  
                self.summary\_data, self.wordcloud\_words \= await summarize\_results(self.df\_analyzed, column)  
                messagebox.showinfo("完了", "分析が完了しました。結果を保存できます。")  
                \# (以降の処理は同じ)  
            \# ...

以上の指示に従ってコードを修正してください。これにより、高品質なPDFレポートが生成されるようになります。