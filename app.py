import os
import re

import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0",
)

# Local folder or Hugging Face model id, e.g. "your-username/t5-samsum-summarizer"
MODEL_PATH = os.getenv("MODEL_PATH", "./saved_summary_model")

model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

templates = Jinja2Templates(directory=".")


class DialogueInput(BaseModel):
    dialogue: str


def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)  # line breaks
    text = re.sub(r"\s+", " ", text)  # extra spaces
    text = re.sub(r"<.*?>", " ", text)  # html tags
    return text.strip().lower()


def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(targets[0], skip_special_tokens=True)


# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
