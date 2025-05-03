# Sentiment Analysis with BERT on IMDb Dataset

This project uses Hugging Face's BERT (`bert-base-uncased`) to perform sentiment analysis on the IMDb dataset, classifying movie reviews as **Positive** or **Negative**. The training script (`main.py`) fine-tunes BERT on a subsampled dataset, and the FastAPI application (`app.py`) serves predictions via a local REST API. Optimized for efficiency, the model trains in ~5 minutes on an M3 Mac with MPS acceleration.

## Features
- **Fast Training**: Fine-tunes BERT on 1,000 IMDb samples in ~5 minutes.
- **Local API**: Provides a `/predict` endpoint for real-time sentiment analysis.
- **Visualization**: Generates a confusion matrix to evaluate performance.
- **Lightweight**: Designed for local execution with minimal resource demands.

## Results
- **Dataset**: 900 train, 100 validation, 1,000 test samples (subsampled from IMDb).
- **Training Time**: ~61.55 seconds (1 epoch, batch size 16).
- **Test Metrics** (2025-05-03):
  - Accuracy: 80.40%
  - Precision: 87.06%
  - Recall: 70.29%
  - F1 Score: 77.78%
- **Outputs**:
  - Model and tokenizer: `bert-sentiment/`
  - Confusion Matrix: `figures/confusion_matrix.png`
  - Cached datasets: `tokenized_imdb_train/`, `tokenized_imdb_test/`

## Prerequisites
- **Hardware**: Mac with M3 chip (MPS recommended) or CPU.
- **Software**:
  - Python 3.11
  - Anaconda
  - PyCharm (optional)
- **Dependencies**: Listed in `requirements.txt`.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>cd /Users/thebowofapollo/PycharmProjects/bert-sentiment-analysis
