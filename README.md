# Aftercoding

This repository contains a prototype survey analysis tool that processes Japanese free-text responses.
It leverages spaCy with SudachiPy and OpenAI's models to summarize sentiment, topics and actionable insights from Excel files.
Results can be exported to Excel, PDF reports and word-cloud images.

For detailed setup and usage instructions, see [coding/survey_analysis_mvp/README.md](coding/survey_analysis_mvp/README.md).

### Requirements
- **Fonts:** NotoSansJP Regular and Bold fonts are already provided under `coding/survey_analysis_mvp/fonts/`. If you wish to replace them, add TTF or OTF versions of `NotoSansJP-Regular` and `NotoSansJP-Bold` to that folder.
- **API key:** Set `OPENAI_API_KEY` in your environment or in a `.env` file so the application can access the OpenAI API.

### Testing

To verify that all Python modules are syntactically correct even when file paths
contain non-ASCII characters, run:

```bash
python scripts/compile_all.py
```

This script gathers all `*.py` files using `pathlib` and compiles each one with
`py_compile`, ensuring paths with Japanese characters are handled reliably.

