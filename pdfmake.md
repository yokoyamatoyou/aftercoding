# **【AI開発者向け】PDF分析レポート技術仕様書**

## **1\. 目的と対象読者**

### **1.1. 目的**

本ドキュメントは、アンケート分析ツールが生成するPDFサマリーレポートの技術仕様を定義する。分析結果の要点を視覚的に分かりやすく伝え、迅速な意思決定を支援するPDFをプログラムで自動生成することを目的とする。

### **1.2. 対象読者**

本ドキュメントの主たる読者は、\*\*AIコーディングエージェント（例: OpenAI Codex, Gemini）\*\*である。また、本システムの開発・保守を担当する人間の開発者も対象とする。AIが曖昧さなくタスクを解釈し、一貫性のあるコードを生成できるよう、具体的かつ技術的な指示を記述する。

## **2\. アーキテクチャ概要とデータフロー**

### **2.1. 技術スタック**

| コンポーネント | 技術・ライブラリ | バージョン / 備考 |
| :---- | :---- | :---- |
| **言語** | Python | 3.10+ |
| **データ分析** | pandas | 2.x |
| **グラフ描画** | Matplotlib | 3.x |
| **PDF生成** | fpdf2 | 2.7+ |
| **HTMLテンプレート** | Jinja2 | 3.x |
| **日本語フォント** | Noto Sans JP | fonts/ ディレクトリに配置済みのOTF/TTF |

### **2.2. データフロー**

1. **入力**: main.py がExcelファイルからpandas.DataFrameを生成する。  
2. **分析・集計**: analysis.pyのanalyze\_dataframe関数がDataFrameを分析し、summarize\_results関数がレポート用のデータ summary\_data (dict) を生成する。  
3. **レポート生成**: reporting.pyのgenerate\_pdf\_report関数がsummary\_dataを受け取る。  
4. **グラフ生成**: generate\_pdf\_reportは、summary\_data内の各データを引数として、複数のグラフ生成関数（例: create\_sentiment\_pie\_chart\_base64）を呼び出す。各関数はグラフ画像を**Base64エンコードされたPNG文字列**として返す。  
5. **HTMLレンダリング**: Jinja2を使い、report\_template.htmlにグラフのBase64文字列を埋め込み、完全なHTML文字列を生成する。  
6. **PDF出力**: fpdf2がレンダリングされたHTML文字列を解釈し、最終的なPDFファイルとして出力する。

## **3\. レポート構成要素と実装指示**

### **3.1. ヘッダー（タイトル）**

| 項目 | 仕様 | AI向け実装指示 |
| :---- | :---- | :---- |
| **表示テキスト** | 「アンケート一括分析 サマリーレポート」 | report\_template.html内の\<h1\>タグにハードコードされている。変更は不要。 |
| **スタイル** | H1見出し | CSSでスタイルを定義する。font-weight: bold; color: \#003366; border-bottom: 3px solid \#005ab3; |

### **3.2. 感情分析セクション**

| 項目 | 仕様 | AI向け実装指示 |
| :---- | :---- | :---- |
| **セクション名** | 「感情分析」 (H2見出し) | report\_template.html内の\<h2\>タグに記述。 |
| **表示内容** | 回答全体の感情（positive, negative, neutral, mixed）の割合を示す円グラフ。 | **関数 create\_sentiment\_pie\_chart\_base64 を実装せよ。**\<br\>- **引数**: sentiment\_counts: pandas.Series (Index: 感情ラベル, 値: 件数)\<br\>- **戻り値**: str (Base64エンコードされたPNG画像)\<br\>- **処理内容**: \<br\> 1\. matplotlibを使用する。\<br\> 2\. 日本語フォントを設定するヘルパー関数を呼び出す。\<br\> 3\. ax.pieで円グラフを描画。autopct='%1.1f%%'でパーセンテージを表示。\<br\> 4\. colors引数で指定された配色を適用すること。\<br\> 5\. io.BytesIOを使い、画像をメモリ上でPNG形式で保存。\<br\> 6\. base64.b64encodeでエンコードし、UTF-8文字列として返す。\<br\> 7\. plt.close(fig)を呼び出し、メモリを解放すること。\<br\> 8\. 入力Seriesが空の場合は、空文字列""を返す。 |
| **データソース** | summary\_data\['sentiment\_counts'\] | 型: pandas.Series |
| **配色** | positive: \#4CAF50, neutral: \#FFC107, negative: \#F44336, mixed: \#9E9E9E | plt.pieのcolors引数にこのリストを渡す。 |

### **3.3. 主要トピック分析セクション**

| 項目 | 仕様 | AI向け実装指示 |
| :---- | :---- | :---- |
| **セクション名** | 「主要トピック」 (H2見出し) | report\_template.html内の\<h2\>タグに記述。 |
| **表示内容** | 1\. 上位15トピックの水平棒グラフ。\<br\>2. 上位15トピックの表。 | **関数 create\_topics\_bar\_chart\_base64 を実装せよ。**\<br\>- **引数**: topic\_counts: pandas.Series (Index: トピック名, 値: 出現回数)\<br\>- **戻り値**: str (Base64エンコードされたPNG画像)\<br\>- **処理内容**: \<br\> 1\. figsize=(10, 8)でグラフサイズを指定。\<br\> 2\. topic\_counts.sort\_values().plot(kind="barh", ax=ax)で水平棒グラフを描画。\<br\> 3\. plt.tight\_layout()でレイアウトを自動調整。\<br\> 4\. その他は感情分析グラフと同様のフローでBase64文字列を生成。\<br\>\<br\>**表の実装**: generate\_pdf\_report内でtopic\_counts.to\_frame().to\_html(header=False)を呼び出し、HTMLテーブル文字列を生成してテンプレートに渡す。 |
| **データソース** | summary\_data\['topic\_counts'\] | 型: pandas.Series |

### **3.4. モデレーション結果セクション**

| 項目 | 仕様 | AI向け実装指示 |
| :---- | :---- | :---- |
| **セクション名** | 「モデレーション結果」 (H2見出し) | report\_template.html内の\<h2\>タグに記述。 |
| **表示内容** | 不適切コンテンツカテゴリごとの件数を示す棒グラフ。 | **関数 create\_moderation\_bar\_chart\_base64 を実装せよ。**\<br\>- **引数**: moderation\_summary: dict\[str, int\]\<br\>- **戻り値**: str (Base64エンコードされたPNG画像)\<br\>- **処理内容**: \<br\> 1\. figsize=(10, 6)でグラフサイズを指定。\<br\> 2\. ax.barで棒グラフを描画。色はskyblue。\<br\> 3\. X軸ラベルが重ならないよう plt.xticks(rotation=45, ha="right") を設定。\<br\> 4\. その他は感情分析グラフと同様のフローでBase64文字列を生成。 |
| **データソース** | summary\_data\['moderation\_summary'\] | 型: dict |


### **4.6. PDF生成統括**

| 項目 | 仕様 | AI向け実装指示 |
| :---- | :---- | :---- |
| **統括関数** | generate\_pdf\_report | **関数 generate\_pdf\_report を実装・修正せよ。**\<br\>- **引数**: summary\_data: dict, output\_path: str\<br\>- **処理内容**:\<br\> 1\. summary\_dataから各データを取得し、上記のグラフ生成関数をそれぞれ呼び出してBase64文字列を取得する。\<br\> 2\. Jinja2 Environmentを初期化し、report\_template.htmlをロードする。\<br\> 3\. template.render()を使い、取得したBase64文字列とその他のデータをテンプレートに渡してHTML文字列を生成する。\<br\> 4\. PDF(FPDF, HTMLMixin)のインスタンスを作成する。\<br\> 5\. **重要**: pdf.add\_page()を呼び出した後、pdf.add\_font()でNoto Sans JPフォント（通常・太字）を登録する。uni=Trueを必ず指定すること。\<br\> 6\. pdf.set\_font()で基本フォントを設定する。\<br\> 7\. pdf.write\_html(html\_out)でHTMLをPDFに書き込む。\<br\> 8\. pdf.output(output\_path)でファイルに保存する。\<br\> 9\. フォントファイルが存在しない場合はFileNotFoundErrorを発生させること。 |

