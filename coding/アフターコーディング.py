import asyncio
import os
import json
import pandas as pd
from openai import OpenAI
import customtkinter as ctk
from tkinter import filedialog, messagebox, StringVar, DoubleVar, Listbox
import shutil

class EnhancedTextAnalysisTool:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.df = None
        self.batch_size = 10
        self.max_tokens_per_request = 2048
        self.max_texts_per_request = 20
        self.temperature = 0.0
        self.top_p = 0.0
        self.current_dictionary = "default"
        self.ensure_dictionary_dir()
        self.ensure_default_dictionary()

    def ensure_dictionary_dir(self):
        """辞書ディレクトリが存在することを確認"""
        if not os.path.exists("dictionaries"):
            os.makedirs("dictionaries")

    def ensure_default_dictionary(self):
        """デフォルト辞書ファイルが存在することを確認"""
        excel_path = os.path.join("dictionaries", "default.xlsx")
        json_path = os.path.join("dictionaries", "default.json")
        
        if not os.path.exists(excel_path) and not os.path.exists(json_path):
            # デフォルト辞書を作成
            default_dict = {
                "主要カテゴリ": [
                    "行政サービス", "施設", "人的対応", "制度", "情報提供", "インフラ", "イベント"
                ],
                "具体的対象": [
                    "窓口対応", "ゴミ収集", "道路", "公園", "駐車場", "申請手続き", "オンラインシステム",
                    "職員対応", "電話対応", "メール対応", "案内表示", "公共交通", "子育て支援"
                ],
                "感情評価": [
                    "満足", "不満", "要望", "感謝", "疑問", "提案", "混乱", "期待", "失望"
                ]
            }
            
            # JSONとして保存
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(default_dict, f, ensure_ascii=False, indent=2)
            
            # Excelとしても保存
            self.save_dictionary_to_excel(default_dict, excel_path)

    def get_available_dictionaries(self):
        """利用可能な辞書ファイルのリストを取得（ExcelとJSON両方）"""
        self.ensure_dictionary_dir()
        excel_dicts = [f.split('.')[0] for f in os.listdir("dictionaries") if f.endswith('.xlsx')]
        json_dicts = [f.split('.')[0] for f in os.listdir("dictionaries") if f.endswith('.json')]
        
        # 重複を除去した辞書名リストを返す
        return list(set(excel_dicts + json_dicts))

    def load_tag_dictionary(self, dictionary_name="default"):
        """指定された名前の辞書を読み込む（Excel優先）"""
        try:
            excel_path = os.path.join("dictionaries", f"{dictionary_name}.xlsx")
            json_path = os.path.join("dictionaries", f"{dictionary_name}.json")
            
            if os.path.exists(excel_path):
                # Excelファイルから辞書を読み込む
                df = pd.read_excel(excel_path)
                dictionary = {}
                for column in df.columns:
                    tags = df[column].dropna().astype(str).tolist()
                    tags = [tag.strip() for tag in tags if tag.strip() and tag != "ここに入力"]
                    if tags:
                        dictionary[column] = tags
                return dictionary
            
            elif os.path.exists(json_path):
                # JSONファイルから辞書を読み込む
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            
            else:
                # デフォルト辞書を返す
                self.ensure_default_dictionary()
                return self.load_tag_dictionary()
        
        except Exception as e:
            print(f"辞書読み込みエラー: {e}")
            # デフォルト辞書を返す
            self.ensure_default_dictionary()
            return self.load_tag_dictionary()

    def save_dictionary_to_excel(self, dictionary, file_path):
        """辞書をExcelファイルに保存"""
        try:
            # 最大のタグ数を取得
            max_tags = max(len(tags) for tags in dictionary.values())
            
            # データフレームを作成
            data = {}
            for category, tags in dictionary.items():
                # タグリストを必要な長さに拡張
                extended_tags = tags + [''] * (max_tags - len(tags))
                data[category] = extended_tags
                
            # データフレームに変換
            df = pd.DataFrame(data)
            
            # Excelに出力
            df.to_excel(file_path, index=False)
            return True
        except Exception as e:
            print(f"Excel保存エラー: {str(e)}")
            return False

    def export_dictionary(self, dictionary_name, target_path):
        """辞書をエクスポート（Excel形式）"""
        # 辞書を読み込む
        dictionary = self.load_tag_dictionary(dictionary_name)
        
        # Excelとして保存
        return self.save_dictionary_to_excel(dictionary, target_path)

    def import_dictionary(self, source_path, dictionary_name):
        """辞書をインポート（Excel形式）"""
        try:
            # Excelファイルを読み込む
            df = pd.read_excel(source_path)
            
            # 辞書に変換
            dictionary = {}
            for column in df.columns:
                tags = df[column].dropna().astype(str).tolist()
                tags = [tag.strip() for tag in tags if tag.strip() and tag != "ここに入力"]
                if tags:
                    dictionary[column] = tags
            
            # 最低限必要なキーが含まれているか確認
            if not all(key in dictionary for key in ["主要カテゴリ", "具体的対象", "感情評価"]):
                return False
            
            # 辞書を保存
            json_path = os.path.join("dictionaries", f"{dictionary_name}.json")
            excel_path = os.path.join("dictionaries", f"{dictionary_name}.xlsx")
            
            # JSONとして保存
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(dictionary, f, ensure_ascii=False, indent=2)
            
            # コピー先にExcelを保存
            return self.save_dictionary_to_excel(dictionary, excel_path)
            
        except Exception as e:
            print(f"辞書インポートエラー: {e}")
            return False

    def create_dictionary_template(self):
        """辞書作成用のテンプレートを生成"""
        template = pd.DataFrame({
            "主要カテゴリ": ["行政サービス", "施設", "人的対応", "制度", "情報提供", "インフラ", "イベント", "ここに入力"],
            "具体的対象": ["窓口対応", "ゴミ収集", "道路", "公園", "駐車場", "申請手続き", "オンラインシステム", "ここに入力"],
            "感情評価": ["満足", "不満", "要望", "感謝", "疑問", "提案", "混乱", "期待", "ここに入力"]
        })
        
        template_path = os.path.join("dictionaries", "テンプレート.xlsx")
        template.to_excel(template_path, index=False)
        return template_path

    async def moderate_text_batch(self, texts):
        if len(texts) > self.max_texts_per_request:
            texts = texts[:self.max_texts_per_request]
            print(f"警告: テキスト数が制限を超えたため、最初の{self.max_texts_per_request}個のみを処理します")
        
        total_chars = sum(len(text) for text in texts)
        if total_chars > self.max_tokens_per_request * 2:
            print("警告: トークン数が制限を超える可能性があります")
        
        try:
            response = await asyncio.to_thread(
                self.client.moderations.create,
                input=texts,
                model="omni-moderation-latest"
            )
            return response.results
        except Exception as e:
            print(f"モデレーションエラー: {e}")
            return [None] * len(texts)

    async def get_aggressiveness_scores_batch(self, texts):
        async def process_chunk(chunk):
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that analyzes text for emotions."},
                    {"role": "user", "content": self._create_scoring_prompt(chunk)}
                ]
                
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model="gpt-4o-mini-2024-07-18",
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                
                scores_and_reasons = self._parse_batch_response(response.choices[0].message.content)
                return scores_and_reasons
            except Exception as e:
                print(f"スコアリングエラー: {e}")
                return [(None, None)] * len(chunk)

        chunks = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        tasks = [process_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        
        return [item for sublist in results for item in sublist]

    async def extract_tags_batch(self, texts):
        async def process_chunk(chunk):
            try:
                messages = [
                    {"role": "system", "content": "あなたはテキスト内容を分析し、適切なタグを抽出する専門家です。"},
                    {"role": "user", "content": self._create_tagging_prompt(chunk, self.current_dictionary)}
                ]
                
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model="gpt-4o-mini-2024-07-18",
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                
                tags = self._parse_tags_response(response.choices[0].message.content)
                return tags
            except Exception as e:
                print(f"タグ抽出エラー: {e}")
                return [None] * len(chunk)

        chunks = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        tasks = [process_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        
        return [item for sublist in results for item in sublist]

    def _create_scoring_prompt(self, texts):
        base_prompt = """
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
"""
        for i, text in enumerate(texts, 1):
            base_prompt += f"\n投稿ID: {i}\n{text}\n"
        
        base_prompt += """
以下の形式で各投稿について出力してください：
投稿ID: {ID}
感情スコア:
- 喜び: {スコア}
- 悲しみ: {スコア}
- 恐れ: {スコア}
- 驚き: {スコア}
- 怒り: {スコア}
- 嫌悪: {スコア}
感情全体の理由: {理由}"""
        return base_prompt

    def _create_tagging_prompt(self, texts, dictionary_name="default"):
        # 辞書読み込み
        tag_dict = self.load_tag_dictionary(dictionary_name)
        
        # プロンプトに辞書を組み込む
        dict_section = "【使用可能なタグリスト】\n"
        for category, tags in tag_dict.items():
            dict_section += f"- {category}: {', '.join(tags)}\n"
        
        base_prompt = f"""
あなたはテキスト内容を分析し、適切なタグを抽出する専門家です。

{dict_section}

【タグ抽出の階層構造】
- 第一タグ：主要カテゴリ（上記の「主要カテゴリ」リストから選択）
- 第二タグ：具体的な対象（上記の「具体的対象」リストから選択）
- 第三タグ：感情・評価・状態（上記の「感情評価」リストから選択）

【タグ抽出の重要な指針】
1. 各階層のタグは必ず上記のリストから選択すること
2. リストにない表現は最も近い概念のタグを選択すること
3. 全てのタグは名詞または名詞句で表現すること
4. 文脈から判断して適切なタグを各階層から1つずつ選択すること
5. 該当するものがない場合はタグを空欄にすること（無理に選択しない）

【良いタグ付けの例】
テキスト：「市役所での手続きが複雑で時間がかかりすぎる。もっと簡略化してほしい。」
タグ： 行政サービス, 申請手続き, 不満

テキスト：「新しいオンラインシステムは使いやすくて助かります。ありがとうございます。」
タグ： 情報提供, オンラインシステム, 感謝

分析対象の文章:
"""
        for i, text in enumerate(texts, 1):
            base_prompt += f"\n投稿ID: {i}\n{text}\n"
        
        base_prompt += """
以下の形式で各投稿について出力してください：
投稿ID: {ID}
タグ: 主要カテゴリタグ, 具体的対象タグ, 感情評価タグ
※内容に該当するものがない場合は、空欄のままでも構いません"""
        return base_prompt

    def _parse_batch_response(self, response_text):
        results = []
        current_scores = {}
        current_reason = None
        
        lines = response_text.split('\n')
        i = 0
        current_id = None
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('投稿ID:'):
                # 新しい投稿の開始
                if current_id is not None and current_scores and current_reason:
                    results.append((current_scores, current_reason))
                
                current_id = line.replace('投稿ID:', '').strip()
                current_scores = {}
                current_reason = None
            
            elif line.startswith('感情スコア:'):
                # 感情スコアセクションの開始
                j = i + 1
                while j < min(i + 7, len(lines)):
                    score_line = lines[j].strip()
                    if score_line.startswith('- '):
                        parts = score_line[2:].split(':')
                        if len(parts) == 2:
                            emotion, score = parts
                            try:
                                current_scores[emotion.strip()] = float(score.strip())
                            except ValueError:
                                current_scores[emotion.strip()] = 0.0
                    j += 1
                i = j - 1  # 次の行に進む前にiを調整
            
            elif line.startswith('感情全体の理由:'):
                current_reason = line.replace('感情全体の理由:', '').strip()
            
            i += 1
        
        # 最後の項目を追加
        if current_id is not None and current_scores and current_reason:
            results.append((current_scores, current_reason))
        
        return results

    def _parse_tags_response(self, response_text):
        results = []
        
        lines = response_text.split('\n')
        i = 0
        current_id = None
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('投稿ID:'):
                current_id = line.replace('投稿ID:', '').strip()
            
            elif line.startswith('タグ:'):
                tags_str = line.replace('タグ:', '').strip()
                current_tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
                
                # 3つのタグを確保（足りない場合は空文字）
                while len(current_tags) < 3:
                    current_tags.append('')
                
                # 3つまでに制限
                current_tags = current_tags[:3]
                results.append(current_tags)
            
            i += 1
        
        return results

    async def analyze_file(self, progress_callback=None):
        if self.df is None or '投稿内容' not in self.df.columns:
            return False

        texts = self.df['投稿内容'].tolist()
        total_texts = len(texts)
        
        batches = [texts[i:i + self.batch_size] for i in range(0, total_texts, self.batch_size)]
        
        all_categories = []
        all_emotions = []
        all_tags = []
        
        for i, batch in enumerate(batches):
            moderation_task = self.moderate_text_batch(batch)
            scoring_task = self.get_aggressiveness_scores_batch(batch)
            tagging_task = self.extract_tags_batch(batch)
            
            batch_categories, batch_emotions, batch_tags = await asyncio.gather(
                moderation_task, scoring_task, tagging_task
            )
            
            batch_categories = batch_categories[:len(batch)]
            batch_emotions = batch_emotions[:len(batch)]
            batch_tags = batch_tags[:len(batch)]
            
            all_categories.extend(batch_categories)
            all_emotions.extend(batch_emotions)
            all_tags.extend(batch_tags)
            
            if progress_callback:
                progress = min(100, int((i + 1) * self.batch_size * 100 / total_texts))
                progress_callback(progress)

        all_categories = all_categories[:len(self.df)]
        all_emotions = all_emotions[:len(self.df)]
        all_tags = all_tags[:len(self.df)]

        # モデレーション結果の処理
        category_names = ["hate", "hate/threatening", "self-harm", "sexual", "sexual/minors", "violence", "violence/graphic"]
        
        for name in category_names:
            self.df[f'{name}_flag'] = False
            self.df[f'{name}_score'] = 0.0
        
        for i, result in enumerate(all_categories):
            if result:
                for name in category_names:
                    field_name = name.replace("/", "_")
                    self.df.at[i, f'{name}_flag'] = getattr(result.categories, field_name, False)
                    self.df.at[i, f'{name}_score'] = getattr(result.category_scores, field_name, 0.0)
        
        # 感情分析結果の処理
        emotion_columns = {
            '喜び': 'joy_score',
            '悲しみ': 'sadness_score',
            '恐れ': 'fear_score',
            '驚き': 'surprise_score',
            '怒り': 'anger_score',
            '嫌悪': 'disgust_score'
        }
        
        for col in emotion_columns.values():
            self.df[col] = 0.0
        self.df['emotion_reason'] = ''

        for i, (scores, reason) in enumerate(all_emotions):
            if scores and reason:
                for emotion_ja, col_name in emotion_columns.items():
                    self.df.at[i, col_name] = scores.get(emotion_ja, 0.0)
                self.df.at[i, 'emotion_reason'] = reason
        
        # タグ抽出結果の処理
        self.df['tag1'] = ''
        self.df['tag2'] = ''
        self.df['tag3'] = ''

        for i, tags in enumerate(all_tags):
            if tags:
                for j, tag in enumerate(tags[:3]):
                    self.df.at[i, f'tag{j+1}'] = tag
        
        return True


class EnhancedTextAnalysisGUI:
    def __init__(self):
        self.tool = EnhancedTextAnalysisTool()
        
        # ウィンドウの設定
        self.root = ctk.CTk()
        self.root.title("テキスト分析ツール - 感情分析・タグ抽出")
        self.root.geometry("800x700")
        
        # テーマの設定
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # 変数の初期化
        self.temperature_var = DoubleVar(value=0.0)
        self.top_p_var = DoubleVar(value=0.0)
        self.dict_var = StringVar()
        
        self.setup_gui()

    def setup_gui(self):
        # メインフレーム
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # タイトル
        title_label = ctk.CTkLabel(
            main_frame, 
            text="テキスト分析ツール", 
            font=("Helvetica", 24, "bold")
        )
        title_label.pack(pady=10)
        
        # サブタイトル
        subtitle_label = ctk.CTkLabel(
            main_frame, 
            text="感情分析・モデレーション・タグ抽出", 
            font=("Helvetica", 14)
        )
        subtitle_label.pack(pady=5)
        
        # ステータスラベル
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="ステータス: 準備完了",
            font=("Helvetica", 12)
        )
        self.status_label.pack(pady=5)
        
        # 設定フレーム
        settings_frame = ctk.CTkFrame(main_frame)
        settings_frame.pack(pady=10, padx=20, fill="x")
        
        # 辞書選択フレーム
        dict_frame = ctk.CTkFrame(settings_frame)
        dict_frame.pack(pady=5, padx=10, fill="x")
        
        dict_label = ctk.CTkLabel(dict_frame, text="タグ辞書:", font=("Helvetica", 12))
        dict_label.pack(side="left", padx=5)
        
        # 辞書リストを取得
        dictionaries = self.tool.get_available_dictionaries()
        if dictionaries:
            self.dict_var.set(dictionaries[0])
        else:
            self.dict_var.set("default")
            
        self.dict_dropdown = ctk.CTkComboBox(
            dict_frame,
            values=dictionaries,
            variable=self.dict_var,
            width=200
        )
        self.dict_dropdown.pack(side="left", padx=5)
        
        # 辞書更新ボタン
        refresh_dict_button = ctk.CTkButton(
            dict_frame,
            text="更新",
            command=self.refresh_dictionaries,
            width=60
        )
        refresh_dict_button.pack(side="left", padx=5)
        
        # テンプレートボタン
        template_button = ctk.CTkButton(
            dict_frame,
            text="テンプレート取得",
            command=self.get_template,
            width=120
        )
        template_button.pack(side="left", padx=5)
        
        # パラメータフレーム
        param_frame = ctk.CTkFrame(settings_frame)
        param_frame.pack(pady=5, padx=10, fill="x")
        
        # 温度設定
        temp_label = ctk.CTkLabel(param_frame, text="Temperature:", font=("Helvetica", 12))
        temp_label.pack(side="left", padx=5)
        
        temp_slider = ctk.CTkSlider(
            param_frame,
            from_=0.0,
            to=0.05,
            number_of_steps=5,
            variable=self.temperature_var,
            width=150
        )
        temp_slider.pack(side="left", padx=5)
        
        temp_value_label = ctk.CTkLabel(param_frame, textvariable=self.temperature_var, width=40)
        temp_value_label.pack(side="left", padx=5)
        
        # Top-P設定
        top_p_label = ctk.CTkLabel(param_frame, text="Top-P:", font=("Helvetica", 12))
        top_p_label.pack(side="left", padx=5)
        
        top_p_slider = ctk.CTkSlider(
            param_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=10,
            variable=self.top_p_var,
            width=150
        )
        top_p_slider.pack(side="left", padx=5)
        
        top_p_value_label = ctk.CTkLabel(param_frame, textvariable=self.top_p_var, width=40)
        top_p_value_label.pack(side="left", padx=5)
        
        # 辞書管理フレーム
        dict_manage_frame = ctk.CTkFrame(settings_frame)
        dict_manage_frame.pack(pady=5, padx=10, fill="x")
        
        # 辞書インポートボタン
        import_dict_button = ctk.CTkButton(
            dict_manage_frame,
            text="辞書インポート",
            command=self.import_dictionary,
            width=150
        )
        import_dict_button.pack(side="left", padx=5)
        
        # 辞書エクスポートボタン
        export_dict_button = ctk.CTkButton(
            dict_manage_frame,
            text="辞書エクスポート",
            command=self.export_dictionary,
            width=150
        )
        export_dict_button.pack(side="left", padx=5)
        
        # 辞書編集ボタン
        edit_dict_button = ctk.CTkButton(
            dict_manage_frame,
            text="Excelで辞書編集",
            command=self.edit_dictionary_in_excel,
            width=150
        )
        edit_dict_button.pack(side="left", padx=5)
        
        # プログレスバーフレーム
        progress_frame = ctk.CTkFrame(main_frame)
        progress_frame.pack(pady=10, padx=20, fill="x")
        
        # プログレスバー
        self.progress = ctk.CTkProgressBar(progress_frame)
        self.progress.pack(pady=10, padx=20, fill="x")
        self.progress.set(0)
        
        # 説明テキスト
        info_text = ctk.CTkLabel(
            main_frame,
            text="このツールは投稿内容の感情分析、モデレーション、タグ抽出を行います。\n「投稿内容」列を含むExcelファイルを選択してください。",
            font=("Helvetica", 12),
            wraplength=600
        )
        info_text.pack(pady=10)
        
        # ボタンフレーム
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(pady=20, fill="x")
        
        # ファイル選択ボタン
        load_button = ctk.CTkButton(
            button_frame,
            text="ファイルを選択",
            command=self.load_excel_file,
            width=200
        )
        load_button.pack(pady=10)
        
        # 分析開始ボタン
        analyze_button = ctk.CTkButton(
            button_frame,
            text="分析開始",
            command=self.start_analysis,
            width=200
        )
        analyze_button.pack(pady=10)
        
        # 結果保存ボタン
        save_button = ctk.CTkButton(
            button_frame,
            text="結果を保存",
            command=self.save_results,
            width=200
        )
        save_button.pack(pady=10)
        
        # フッター
        footer_label = ctk.CTkLabel(
            main_frame,
            text="© 2025 テキスト分析ツール",
            font=("Helvetica", 10)
        )
        footer_label.pack(pady=10)

    def refresh_dictionaries(self):
        """辞書リストを更新"""
        dictionaries = self.tool.get_available_dictionaries()
        self.dict_dropdown.configure(values=dictionaries)
        if dictionaries and self.dict_var.get() not in dictionaries:
            self.dict_var.set(dictionaries[0])

    def get_template(self):
        """テンプレートを取得して保存"""
        template_path = self.tool.create_dictionary_template()
        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile="辞書テンプレート.xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        
        if save_path:
            try:
                shutil.copy2(template_path, save_path)
                messagebox.showinfo("成功", f"テンプレートを保存しました: {save_path}")
            except Exception as e:
                messagebox.showerror("エラー", f"テンプレート保存エラー: {str(e)}")

    def import_dictionary(self):
        """辞書ファイルをインポート（Excel形式）"""
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return
            
        # 辞書名を入力
        dialog = ctk.CTkInputDialog(text="辞書名を入力してください:", title="辞書インポート")
        dictionary_name = dialog.get_input()
        
        if not dictionary_name:
            return
            
        if dictionary_name.endswith('.xlsx'):
            dictionary_name = dictionary_name[:-5]
            
        success = self.tool.import_dictionary(file_path, dictionary_name)
        
        if success:
            messagebox.showinfo("成功", f"辞書「{dictionary_name}」をインポートしました")
            self.refresh_dictionaries()
        else:
            messagebox.showerror("エラー", "辞書のインポートに失敗しました。正しい形式のExcelファイルか確認してください。")

    def export_dictionary(self):
        """選択中の辞書をエクスポート（Excel形式）"""
        dictionary_name = self.dict_var.get()
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile=f"{dictionary_name}.xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        
        if not file_path:
            return
            
        success = self.tool.export_dictionary(dictionary_name, file_path)
        
        if success:
            messagebox.showinfo("成功", f"辞書「{dictionary_name}」をエクスポートしました")
        else:
            messagebox.showerror("エラー", "辞書のエクスポートに失敗しました")

    def edit_dictionary_in_excel(self):
        """選択中の辞書をExcelで直接編集"""
        dictionary_name = self.dict_var.get()
        
        if not dictionary_name:
            messagebox.showerror("エラー", "辞書が選択されていません")
            return
        
        # 辞書ファイルパスの準備
        excel_path = os.path.join("dictionaries", f"{dictionary_name}.xlsx")
        json_path = os.path.join("dictionaries", f"{dictionary_name}.json")
        
        # 辞書が存在するか確認
        if not os.path.exists(excel_path) and os.path.exists(json_path):
            # Excelファイルがなければ、JSONから作成
            dictionary = self.tool.load_tag_dictionary(dictionary_name)
            self.tool.save_dictionary_to_excel(dictionary, excel_path)
        elif not os.path.exists(excel_path) and not os.path.exists(json_path):
            messagebox.showerror("エラー", f"辞書「{dictionary_name}」が見つかりません")
            return
        
        # OSのデフォルトアプリケーションでExcelファイルを開く
        try:
            import subprocess
            import platform
            
            system = platform.system()
            if system == 'Windows':
                os.startfile(excel_path)
            elif system == 'Darwin':  # macOS
                subprocess.call(['open', excel_path])
            else:  # Linux
                subprocess.call(['xdg-open', excel_path])
                
            messagebox.showinfo("情報", f"辞書「{dictionary_name}」を外部アプリケーションで開きました。編集後、アプリケーションを再起動するか、辞書リストの「更新」ボタンをクリックしてください。")
        except Exception as e:
            messagebox.showerror("エラー", f"ファイルを開けませんでした: {str(e)}")

    def update_progress(self, value):
        self.progress.set(value / 100)
        self.root.update_idletasks()

    def load_excel_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            try:
                self.tool.df = pd.read_excel(file_path, sheet_name=0)
                if '投稿内容' not in self.tool.df.columns:
                    messagebox.showerror("エラー", "ファイルに「投稿内容」列が見つかりません")
                    self.tool.df = None
                    return
                
                self.status_label.configure(text=f"ステータス: ファイルを読み込みました（{len(self.tool.df)}件）")
                messagebox.showinfo("成功", f"{len(self.tool.df)}件のデータを読み込みました")
            except Exception as e:
                self.status_label.configure(text="ステータス: ファイル読み込みに失敗しました")
                messagebox.showerror("エラー", f"読み込みエラー: {e}")

    async def analyze_wrapper(self):
        # 設定を適用
        self.tool.temperature = self.temperature_var.get()
        self.tool.top_p = self.top_p_var.get()
        self.tool.current_dictionary = self.dict_var.get()
        
        success = await self.tool.analyze_file(self.update_progress)
        if success:
            self.status_label.configure(text="ステータス: 分析が完了しました")
            messagebox.showinfo("完了", "感情分析・モデレーション・タグ抽出が完了しました")
        else:
            self.status_label.configure(text="ステータス: 分析に失敗しました")
            messagebox.showerror("エラー", "分析に失敗しました")

    def start_analysis(self):
        if self.tool.df is None:
            messagebox.showerror("エラー", "ファイルを先にアップロードしてください")
            return

        self.status_label.configure(text="ステータス: 分析を実行中...")
        asyncio.run(self.analyze_wrapper())

    def save_results(self):
        if self.tool.df is None:
            messagebox.showerror("エラー", "保存する結果がありません")
            return
            
        if not any(col.startswith('tag') for col in self.tool.df.columns):
            messagebox.showerror("エラー", "先に分析を実行してください")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if file_path:
            try:
                self.tool.df.to_excel(file_path, index=False)
                self.status_label.configure(text="ステータス: 結果を保存しました")
                messagebox.showinfo("成功", "結果を保存しました")
            except Exception as e:
                self.status_label.configure(text="ステータス: 保存に失敗しました")
                messagebox.showerror("エラー", f"保存エラー: {e}")

    def run(self):
        try:
            self.root.mainloop()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"エラーが発生しました: {e}")
            input("Enterキーを押して終了...")


# アプリケーション起動
if __name__ == "__main__":
    try:
        gui = EnhancedTextAnalysisGUI()
        gui.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"エラーが発生しました: {e}")
        input("Enterキーを押して終了...")
