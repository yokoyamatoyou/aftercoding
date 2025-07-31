import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import os
import shutil

from analysis import analyze_survey
from reporting import create_report


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("アンケート分析レポート生成ツール")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.geometry("600x400")

        self.file_path = ""
        self.df_analyzed = None
        self.output_dir = os.path.join(os.path.dirname(__file__), "output")
        self.pdf_path = None
        self.wordcloud_file = None

        # file frame
        file_frame = ctk.CTkFrame(self)
        file_frame.pack(padx=20, pady=10, fill="x")
        ctk.CTkLabel(file_frame, text="1. 分析対象のExcelファイルを選択").pack(anchor="w")
        inner = ctk.CTkFrame(file_frame)
        inner.pack(fill="x")
        self.file_entry = ctk.CTkEntry(inner, state="disabled", width=400)
        self.file_entry.pack(side="left", padx=5, pady=5, fill="x", expand=True)
        ctk.CTkButton(inner, text="ファイルを選択", command=self.select_file).pack(side="left", padx=5)

        # column frame
        column_frame = ctk.CTkFrame(self)
        column_frame.pack(padx=20, pady=10, fill="x")
        ctk.CTkLabel(column_frame, text="2. 分析対象の列名を入力").pack(anchor="w")
        self.column_entry = ctk.CTkEntry(column_frame)
        self.column_entry.pack(fill="x", padx=5, pady=5)

        # wordcloud type frame
        wc_frame = ctk.CTkFrame(self)
        wc_frame.pack(padx=20, pady=10, fill="x")
        ctk.CTkLabel(wc_frame, text="3. ワードクラウドの種類を選択").pack(anchor="w")
        self.wc_var = ctk.StringVar(value="positive")
        options = [("ノーマル", "normal"), ("ポジティブ", "positive"), ("ネガティブ", "negative")]
        btn_frame = ctk.CTkFrame(wc_frame)
        btn_frame.pack(pady=5)
        for text, val in options:
            ctk.CTkRadioButton(btn_frame, text=text, variable=self.wc_var, value=val).pack(side="left", padx=10)

        # run frame
        run_frame = ctk.CTkFrame(self)
        run_frame.pack(padx=20, pady=20, fill="x")
        self.run_button = ctk.CTkButton(
            run_frame, text="レポート生成開始", command=self.run
        )
        self.run_button.pack(pady=5)
        self.progress = ctk.CTkProgressBar(run_frame)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=5, pady=5)
        self.status = ctk.CTkLabel(run_frame, text="準備完了")
        self.status.pack()

        # save frame
        save_frame = ctk.CTkFrame(self)
        save_frame.pack(padx=20, pady=10, fill="x")
        self.save_excel_button = ctk.CTkButton(
            save_frame,
            text="分析結果をExcelに保存",
            command=self.save_excel,
            state="disabled",
        )
        self.save_excel_button.pack(side="left", padx=5, pady=5, expand=True)
        self.save_pdf_button = ctk.CTkButton(
            save_frame,
            text="サマリーPDFを保存",
            command=self.save_pdf,
            state="disabled",
        )
        self.save_pdf_button.pack(side="left", padx=5, pady=5, expand=True)
        self.save_wordcloud_button = ctk.CTkButton(
            save_frame,
            text="ワードクラウドを保存",
            command=self.save_wordcloud,
            state="disabled",
        )
        self.save_wordcloud_button.pack(side="left", padx=5, pady=5, expand=True)

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if path:
            self.file_path = path
            self.file_entry.configure(state="normal")
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, path)
            self.file_entry.configure(state="disabled")

    def run(self):
        if not self.file_path:
            messagebox.showerror("エラー", "Excelファイルを選択してください")
            return
        column = self.column_entry.get().strip()
        if not column:
            messagebox.showerror("エラー", "列名を入力してください")
            return

        self.run_button.configure(state="disabled")
        self.progress.set(0)
        self.status.configure(text="分析中...")
        self.update()
        try:
            df, pos_sum, neg_sum = analyze_survey(self.file_path, column)
            self.progress.set(0.5)
            self.status.configure(text="レポート生成中...")
            self.update()
            create_report(df, pos_sum, neg_sum, self.wc_var.get(), column)
            self.df_analyzed = df
            self.pdf_path = os.path.join(self.output_dir, "survey_report.pdf")
            if self.wc_var.get() == "normal":
                self.wordcloud_file = os.path.join(self.output_dir, "wordcloud.png")
            elif self.wc_var.get() == "positive":
                self.wordcloud_file = os.path.join(self.output_dir, "positive_wordcloud.png")
            else:
                self.wordcloud_file = os.path.join(self.output_dir, "negative_wordcloud.png")
            self.progress.set(1)
            self.status.configure(text="完了")
            self.save_excel_button.configure(state="normal")
            if os.path.exists(self.pdf_path):
                self.save_pdf_button.configure(state="normal")
            if os.path.exists(self.wordcloud_file):
                self.save_wordcloud_button.configure(state="normal")
            messagebox.showinfo("完了", "レポートを output フォルダに保存しました")
        except Exception as e:
            messagebox.showerror("エラー", str(e))
        finally:
            self.run_button.configure(state="normal")

    def save_excel(self):
        if self.df_analyzed is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if path:
            try:
                self.df_analyzed.to_excel(path, index=False)
                messagebox.showinfo("成功", f"分析結果を {path} に保存しました。")
            except Exception as e:
                messagebox.showerror("保存エラー", f"Excelファイルの保存に失敗しました:\n{e}")

    def save_pdf(self):
        if not self.pdf_path or not os.path.exists(self.pdf_path):
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
        )
        if path:
            try:
                shutil.copy(self.pdf_path, path)
                messagebox.showinfo("成功", f"PDFレポートを {path} に保存しました。")
            except Exception as e:
                messagebox.showerror("保存エラー", f"PDFの保存に失敗しました:\n{e}")

    def save_wordcloud(self):
        if not self.wordcloud_file or not os.path.exists(self.wordcloud_file):
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG images", "*.png")],
        )
        if path:
            try:
                shutil.copy(self.wordcloud_file, path)
                messagebox.showinfo("成功", f"ワードクラウドを {path} に保存しました。")
            except Exception as e:
                messagebox.showerror("保存エラー", f"ワードクラウドの保存に失敗しました:\n{e}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
