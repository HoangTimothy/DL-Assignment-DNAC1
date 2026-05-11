---
title: dbpedia-text-demo
sdk: gradio
app_file: app.py
---

# DBPedia Text Classification Demo

## Configure model
- In Space Settings -> Variables, set `HF_SPACE_MODEL_ID` to your fine-tuned model repo.
- If not set, app falls back to a generic DistilBERT classifier head.

## Local test
```bash
pip install -r requirements.txt
python app.py
```
