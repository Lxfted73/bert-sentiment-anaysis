import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"  # Suppress Xet Storage warning

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Create figures directory
print(f"[{datetime.now()}] Creating figures directory if it doesn't exist...")
if not os.path.exists("figures"):
    os.makedirs("figures")
    print(f"[{datetime.now()}] Figures directory created.")
else:
    print(f"[{datetime.now()}] Figures directory already exists.")

# Set device to MPS (M3) or CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[{datetime.now()}] Using device: {device}")

# Load IMDb dataset
print(f"[{datetime.now()}] Loading IMDb dataset...")
dataset = load_dataset("imdb")
print(f"[{datetime.now()}] Dataset loaded successfully:")
print(dataset)

# Subsample dataset for faster training
print(f"[{datetime.now()}] Subsampling dataset...")
train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))  # 1,000 training samples
test_dataset = dataset["test"].shuffle(seed=42).select(range(1000))    # 1,000 test samples
print(f"[{datetime.now()}] Subsampled dataset: Train={len(train_dataset)}, Test={len(test_dataset)}")

# Load tokenizer
print(f"[{datetime.now()}] Loading BERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(f"[{datetime.now()}] Tokenizer loaded.")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,  # Reduced for M3 memory efficiency
        return_tensors="pt"
    )

# Apply tokenization
print(f"[{datetime.now()}] Tokenizing dataset...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)
print(f"[{datetime.now()}] Dataset tokenized.")

# Format dataset for PyTorch
print(f"[{datetime.now()}] Formatting dataset for PyTorch...")
tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_train.set_format("torch")
tokenized_test = tokenized_test.remove_columns(["text"])
tokenized_test = tokenized_test.rename_column("label", "labels")
tokenized_test.set_format("torch")
print(f"[{datetime.now()}] Dataset formatted.")

# Split train into train and validation
print(f"[{datetime.now()}] Splitting train dataset into train and validation...")
train_val_split = tokenized_train.train_test_split(test_size=0.1, seed=42)
train_dataset = train_val_split["train"]  # 900 samples
val_dataset = train_val_split["test"]     # 100 samples
print(f"[{datetime.now()}] Dataset split completed:")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(tokenized_test)}")

# Cache tokenized dataset
print(f"[{datetime.now()}] Caching tokenized dataset to disk...")
tokenized_train.save_to_disk("tokenized_imdb_train")
tokenized_test.save_to_disk("tokenized_imdb_test")
print(f"[{datetime.now()}] Dataset cached to 'tokenized_imdb_train' and 'tokenized_imdb_test'.")

# Load model
print(f"[{datetime.now()}] Loading BERT model...")
try:
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    print(f"[{datetime.now()}] Model weights downloaded and initialized.")
    model.to(device)
    print(f"[{datetime.now()}] Model moved to {device}.")
except Exception as e:
    print(f"[{datetime.now()}] Error loading model: {e}")
    raise

# Define metrics
def compute_metrics(pred):
    print(f"[{datetime.now()}] Computing evaluation metrics...")
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    print(f"[{datetime.now()}] Metrics computed: Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Training arguments
print(f"[{datetime.now()}] Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./bert-sentiment",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Increased for speed
    per_device_eval_batch_size=16,
    num_train_epochs=1,  # Reduced to 1 epoch
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=False,  # MPS doesn't support fp16
    logging_steps=10,  # Log every 10 steps for small dataset
)
print(f"[{datetime.now()}] Training arguments configured.")

# Initialize Trainer
print(f"[{datetime.now()}] Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
print(f"[{datetime.now()}] Trainer initialized.")

# Train
print(f"[{datetime.now()}] Starting training...")
try:
    trainer.train()
    print(f"[{datetime.now()}] Training completed successfully.")
except Exception as e:
    print(f"[{datetime.now()}] Error during training: {e}")
    raise

# Save model and tokenizer
print(f"[{datetime.now()}] Saving model and tokenizer...")
model.save_pretrained("bert-sentiment")
tokenizer.save_pretrained("bert-sentiment")
print(f"[{datetime.now()}] Model and tokenizer saved to 'bert-sentiment'.")

# Evaluate on test set
print(f"[{datetime.now()}] Evaluating on test set...")
test_results = trainer.evaluate(tokenized_test)
print(f"[{datetime.now()}] Test results:", test_results)

# Get predictions for confusion matrix
print(f"[{datetime.now()}] Generating predictions for confusion matrix...")
predictions = trainer.predict(tokenized_test)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids
print(f"[{datetime.now()}] Predictions generated.")

# Plot confusion matrix
print(f"[{datetime.now()}] Plotting confusion matrix...")
cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("figures/confusion_matrix.png")
plt.show()
print(f"[{datetime.now()}] Confusion matrix saved to 'figures/confusion_matrix.png'.")

# Example predictions
print(f"[{datetime.now()}] Printing example predictions...")
sample_reviews = test_dataset["text"][:5]
sample_labels = test_dataset["label"][:5]
sample_preds = preds[:5]
for review, label, pred in zip(sample_reviews, sample_labels, sample_preds):
    print(f"Review: {review[:100]}...")
    print(f"True Label: {'Positive' if label == 1 else 'Negative'}")
    print(f"Predicted: {'Positive' if pred == 1 else 'Negative'}\n")
print(f"[{datetime.now()}] Example predictions printed.")