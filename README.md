# Aftercoding

This repository contains a prototype survey analysis tool that processes Japanese free-text responses.
It leverages spaCy with SudachiPy and OpenAI's models to summarize sentiment, topics and actionable insights from Excel files.
Results can be exported to Excel, PDF reports and word‑cloud images.  The PDF
reports are created using **fpdf2** via the `ReportPDF` class and include graphs
rendered with Matplotlib.

The analysis pipeline makes extensive use of **asyncio** and `AsyncOpenAI` so
multiple API calls can run concurrently.  You can control the concurrency level
with the `MAX_CONCURRENT_TASKS` setting.

For detailed setup and usage instructions, see [coding/survey_analysis_mvp/README.md](coding/survey_analysis_mvp/README.md).

### Requirements
- **Fonts:** NotoSansJP Regular and Bold fonts are already provided under `coding/survey_analysis_mvp/fonts/`. If you wish to replace them, add TTF or OTF versions of `NotoSansJP-Regular` and `NotoSansJP-Bold` to that folder.
- **API key:** Copy `.env.example` to `.env` and set `OPENAI_API_KEY` to your key.
  `analysis.py` automatically assigns `openai.api_key` from this variable (or any
  `OPENAI_API_KEY` found in your environment). You can optionally set
  `MAX_CONCURRENT_TASKS` to control how many API requests run concurrently
  (default is 5).

### Running the application

Launch the GUI with:

```bash
python coding/survey_analysis_mvp/main.py
```

Windows users can also double‑click `start_app.bat` which runs the same command.

### Programmatic usage

For automated processing you can call `analysis.analyze_dataframe` directly. It
accepts a `pandas.DataFrame` and the column name to analyze, returning the
original data combined with structured results. The previous `analyze_survey`
helper has been removed.

### Testing

An optional check is available to ensure all Python modules compile correctly,
even if file paths include non‑ASCII characters.  Run:

```bash
python scripts/compile_all.py
```

This script gathers all `*.py` files using `pathlib` and compiles each one with
`py_compile`, ensuring paths with Japanese characters are handled reliably.

To run the unit tests, install the dependencies listed under
`coding/survey_analysis_mvp/requirements.txt` and execute `pytest` from the
repository root:

```bash
pip install -r coding/survey_analysis_mvp/requirements.txt
pytest
```

### Repository notes

The repository root contains a `.gitignore` configured to exclude Python bytecode,
virtual environment folders and generated `output/` artifacts. Contributors
should ensure their local environments respect these settings when committing
changes.

This project is released under the terms of the [MIT License](LICENSE).

