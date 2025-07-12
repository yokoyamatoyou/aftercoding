# Aftercoding

This repository contains a prototype survey analysis tool that processes Japanese free-text responses.
It leverages spaCy with SudachiPy and OpenAI's models to summarize sentiment, topics and actionable insights from Excel files.
Results can be exported to Excel, PDF reports and word-cloud images.

For detailed setup and usage instructions, see [coding/survey_analysis_mvp/README.md](coding/survey_analysis_mvp/README.md).

### Requirements
- **Fonts:** Place TTF versions of `NotoSansJP-Regular` and `NotoSansJP-Bold` in the `fonts/` folder to render Japanese text correctly.
- **API key:** Set `OPENAI_API_KEY` in your environment or in a `.env` file so the application can access the OpenAI API.

