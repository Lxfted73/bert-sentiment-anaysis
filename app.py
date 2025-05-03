import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

try:
    model = AutoModelForSequenceClassification.from_pretrained("bert-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("bert-sentiment")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    raise

class Review(BaseModel):
    text: str

@app.post("/predict")
async def predict(review: Review):
    try:
        inputs = tokenizer(
            review.text,
            padding="max_length",
            truncation=True,
            max_length=256,  # Match main.py
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            sentiment = "Positive" if probs[0][1] > probs[0][0] else "Negative"
            confidence = probs[0][1].item() if sentiment == "Positive" else probs[0][0].item()
        return {"sentiment": sentiment, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)