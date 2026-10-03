# Text Summarizer (T5 + FastAPI)

Fine-tuned **t5-small** on the [SAMSum](https://huggingface.co/datasets/Samsung/samsum) dialogue dataset and served it through a **FastAPI** app with a simple web UI.


## Features
- Fine-tuning pipeline in a Jupyter notebook (4,000 train / 500 validation samples, 6 epochs)
- REST API: `POST /summarize/`
- Web UI to paste text and get a summary

## Tech stack
Python, PyTorch, Hugging Face Transformers, FastAPI, HTML/CSS/JS

## Project structure
```
Summarizer-HF/
├── app.py                  # FastAPI backend
├── index.html              # web UI
├── text_summarizer.ipynb   # data cleaning + fine-tuning
├── requirements.txt
└── README.md
```

## Setup
```bash
git clone https://github.com/<your-username>/Summarizer-HF.git
cd Summarizer-HF
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the model
The model weights are not stored in this repo. Use one of these options:

**Option A: Download from Hugging Face**
```bash
export MODEL_PATH="<your-username>/t5-samsum-summarizer"   # Windows: set MODEL_PATH=...
```

**Option B: Train it yourself**
1. Download the SAMSum CSV files (`samsum-train.csv`, `samsum-validation.csv`) into the project folder.
2. Run all cells in `text_summarizer.ipynb`. This creates `./saved_summary_model`.

## Run
```bash
uvicorn app:app --reload
```
Open http://127.0.0.1:8000

## API example
```bash
curl -X POST http://127.0.0.1:8000/summarize/ \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "Amy: hi, are we meeting today? Bob: yes, at 5 pm at the cafe."}'
```
Response:
```json
{"summary": "..."}
```

## Limitations
- Trained on short chat-style dialogues, so summaries of long articles or news transcripts will be weaker.
- Trained on a subset of SAMSum with t5-small, so it's a learning project rather than a production model.
- Input is truncated to 512 tokens.

## Future improvements
- Train on the full dataset and a larger model (t5-base)
- Add ROUGE evaluation
- Dockerize and deploy
