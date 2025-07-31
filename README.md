# Aftercoding

This repository contains a prototype survey analysis tool that processes Japanese free-text responses.
It leverages spaCy with SudachiPy and OpenAI's models to summarize sentiment, topics and actionable insights from Excel files.
Results can be exported to Excel, PDF reports and word-cloud images.

For detailed setup and usage instructions, see [coding/survey_analysis_mvp/README.md](coding/survey_analysis_mvp/README.md).

### Requirements
- **Fonts:** NotoSansJP Regular and Bold fonts are already provided under `coding/survey_analysis_mvp/fonts/`. If you wish to replace them, add TTF or OTF versions of `NotoSansJP-Regular` and `NotoSansJP-Bold` to that folder.
- **API key:** Copy `.env.example` to `.env` and set `OPENAI_API_KEY` to your key.
  `analysis.py` automatically assigns `openai.api_key` from this variable (or any
  `OPENAI_API_KEY` found in your environment). You can optionally set
  `MAX_CONCURRENT_TASKS` to control how many API requests run concurrently
  (default is 5).

### Testing

To verify that all Python modules are syntactically correct even when file paths
contain non-ASCII characters, run:

```bash
python scripts/compile_all.py
```

This script gathers all `*.py` files using `pathlib` and compiles each one with
`py_compile`, ensuring paths with Japanese characters are handled reliably.

### Repository notes

The repository root contains a `.gitignore` configured to exclude Python bytecode,
virtual environment folders and generated `output/` artifacts. Contributors
should ensure their local environments respect these settings when committing
changes.

This project is released under the terms of the [MIT License](LICENSE).

