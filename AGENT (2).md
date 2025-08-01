# **PDFレポート改善・GUI化のためのリファクタリング指示書**

## **1\. 目的**

既存のアンケート分析・PDFレポート生成スクリプトをリファクタリングし、以下の機能改善と品質向上を実現する。

* **GUIの導入**: ユーザーが直感的に操作できるGUIアプリケーションを構築する。  
* **レイアウトの安定化**: fpdf2 ベースの **ReportPDF** クラスで座標指定のレイアウトを行い、グラフや表のレイアウト崩れを根本的に解決する。
* **機能拡張**: ワードクラウドの種類（ノーマル、ポジティブ、ネガティブ）をGUIから選択できるようにする。  
* **モジュール化**: 各機能（UI, 分析, レポート生成）をファイルごとに分離し、メンテナンス性と再利用性を向上させる。

## **2\. 推奨ライブラリ**

* **GUI**: customtkinter (モダンな見た目のGUIを簡単に作成できる)  
* **データ操作**: pandas  
* **PDF生成**: `ReportPDF` (fpdf2) を使用して高品質なPDFを生成
* **HTMLテンプレート**: Jinja2  
* **ワードクラウド**: wordcloud  
* **グラフ描画**: matplotlib, seaborn  
* **AI分析**: openai

requirements.txt に以下を追加・更新してください。

pandas  
openpyxl
openai
customtkinter
fpdf2
jinja2
wordcloud
matplotlib
seaborn

## **3\. 新しいファイル構成案**

/survey\_analysis\_mvp/  
|-- app.py                    \# \[新規\] GUIアプリケーションのメインファイル  
|-- analysis.py               \# \[修正\] データ分析とAIによる感情分析・要約ロジック  
|-- reporting.py              \# \[修正\] グラフ・ワードクラウド生成とPDF出力ロジック  
|-- report\_template.html      \# \[修正\] レポートのレイアウトを定義するHTMLテンプレート  
|-- style.css                 \# \[新規\] report\_template.htmlに適用するCSSファイル  
|-- config.py                 \# APIキーなどの設定ファイル  
|-- fonts/                    \# フォントファイル (既存)  
|-- output/                   \# \[新規\] 生成されたPDFや画像が保存されるディレクトリ  
|   |-- survey\_report.pdf  
|   |-- positive\_wordcloud.png  
|   |-- negative\_wordcloud.png  
|   |-- sentiment\_chart.png  
|-- requirements.txt          \# \[修正\]  
\`-- README.md                 \# \[修正\]

## **4\. 各ファイルの具体的な実装指示**

### **4.1. app.py (GUIアプリケーション) \- 新規作成**

customtkinter を使用して、以下の仕様でGUIアプリケーションを実装する。

* **ウィンドウ**:  
  * タイトル: 「アンケート分析レポート生成ツール」  
  * 外観モード: 「System」 (OSに合わせる)  
  * デフォルトカラーテーマ: 「blue」  
* **UIコンポーネント**:  
  1. **ファイル選択フレーム**:  
     * ラベル: 「1. 分析対象のExcelファイルを選択」  
     * テキストボックス: 選択されたファイルパスを表示（編集不可）  
     * ボタン: 「ファイルを選択」 \- クリックするとファイルダイアログが開き、Excelファイル (.xlsx) を選択できる。  
  2. **列名指定フレーム**:  
     * ラベル: 「2. 分析対象の列名を入力」  
     * 入力フィールド: ユーザーがアンケートの自由回答が記載されている列名を入力する。（例: ご意見・ご感想）  
  3. **ワードクラウド選択フレーム**:  
     * ラベル: 「3. ワードクラウドの種類を選択」  
     * ラジオボタン（3つ）:  
       * ノーマル (全ての回答を使用)  
       * ポジティブ (ポジティブ \+ ニュートラルの回答を使用) \- **デフォルト選択**  
       * ネガティブ (ネガティブ \+ ニュートラルの回答を使用)  
  4. **実行ボタン**:  
     * ボタンテキスト: 「レポート生成開始」  
     * クリック時の動作:  
       * 入力チェック（ファイルが選択されているか、列名が入力されているか）を行う。  
       * analysis.py と reporting.py の関数を呼び出し、レポート生成プロセスを開始する。  
       * 処理中はボタンを無効化する。  
  5. **ステータス表示**:  
     * プログレスバー: 処理の進捗を視覚的に表示する。  
     * ラベル: 「準備完了」「分析中...」「レポート生成中...」「完了」など、現在のステータスをテキストで表示する。

### **4.2. analysis.py (データ分析モジュール) \- 修正**

* **analyze\_survey(file\_path, column\_name) 関数を実装する。**  
  * **入力**: Excelファイルパス、分析対象の列名  
  * **処理**:  
    1. pandasでExcelファイルを読み込む。  
    2. 指定された列の各行のテキストに対して、openai APIを呼び出し、感情を「ポジティブ」「ネガティブ」「ニュートラル」の3つに分類する。  
       * **プロンプト例**: あなたはテキスト分析の専門家です。与えられたテキストを「ポジティブ」「ネガティブ」「ニュートラル」のいずれか一つに分類してください。  
    3. 分類結果をDataFrameの新しい列（例: sentiment）に追加する。  
    4. カテゴリごと（ポジティブ、ネガティブ）に意見をグループ化する。  
    5. 各カテゴリの意見をまとめて、openai APIで要約を生成する。  
       * **プロンプト例**: 以下の意見リストは、製品アンケートの回答です。マーケティング担当者が傾向を把握できるよう、重要なポイントを3つにまとめてください。\\n\\n{意見リスト}  
  * **出力**:  
    * 感情分類済みの DataFrame  
    * ポジティブ意見の要約テキスト  
    * ネガティブ意見の要約テキスト

### **4.3. reporting.py (レポート生成モジュール) \- 修正**

* **create\_report(df, positive\_summary, negative\_summary, wordcloud\_type) 関数を実装する。**  
  * **入力**: 分析済みDataFrame、各要約テキスト、GUIで選択されたワードクラウドの種類  
  * **処理**:  
    1. **感情分析グラフ生成**:  
       * matplotlib / seaborn を使用し、感情分類（ポジティブ, ネガティブ, ニュートラル）の件数や割合を示す円グラフまたは棒グラフを生成する。  
       * 生成したグラフを output/sentiment\_chart.png として保存する。  
    2. **ワードクラウド生成**:  
       * wordcloud\_type の値に応じて、ワードクラウドの元になるテキストを決定する。  
         * ノーマル: 全ての回答テキスト  
         * ポジティブ: ポジティブ \+ ニュートラルの回答テキスト  
         * ネガティブ: ネガティブ \+ ニュートラルの回答テキスト  
       * wordcloud ライブラリを使い、ワードクラウド画像を2つ（ポジティブ用、ネガティブ用、またはノーマル設定時は1つ）生成する。  
       * **フォントパス**: fonts/NotoSansJP-Regular.otf を指定する。  
       * 生成した画像を output/positive\_wordcloud.png などとして保存する。  
    3. **PDF生成**:  
       * Jinja2 を使用して report\_template.html を読み込む。  
       * テンプレートに渡すデータ（コンテキスト）を辞書で作成する。  
         * positive\_summary, negative\_summary  
         * sentiment\_chart\_path (例: output/sentiment\_chart.png)  
         * positive\_wordcloud\_path, negative\_wordcloud\_path  
         * その他、レポートに表示したい統計情報（回答総数など）  
      * Jinja2でHTMLをレンダリングする。
      * `ReportPDF` クラス (fpdf2) を使用し、レンダリングした内容を描画して output/survey\_report.pdf としてPDFを生成する。詳細は `PDFレポート指示書.md` を参照。

### **4.4. report\_template.html & style.css \- 修正・新規作成**

* **report\_template.html**:  
  * Jinja2 のテンプレート構文（例: {{ positive\_summary }}）を使用して、動的にデータを埋め込めるようにする。  
  * 画像は \<img\> タグで表示する。パスは Jinja2 変数で渡す。  
  * CSSファイルをリンクする (\<link rel="stylesheet" href="style.css"\>)。  
* **style.css**:  
  * @font-face を使って NotoSansJP フォントを定義する。  
  * レポート全体のレイアウトを定義する。display: grid や display: flex を活用して、要素を整然と配置し、レイアウト崩れを防ぐ。  
  * **デザイン案**:  
    * A4サイズを基本とする (@page { size: A4; margin: 2cm; })。  
    * ヘッダーにレポートタイトルとロゴ用のスペースを確保。  
    * フッターにページ番号を自動で挿入 (content: counter(page);)。  
    * セクションごとに背景色を変えたり、罫線を入れたりして見やすくする。  
    * 要約テキストとワードクラウド画像を横並びに配置する2カラムレイアウトなどを採用する。