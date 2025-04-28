# INVEST Classifier with BERT for Multi-Label Classification
#!D:/Program Files/Apache Software Foundation/Apache24/cgi-bin/SimpleFlaskApp/venv/Scripts/python.exe

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import json
import datetime
import os

import sys

try:
    from transformers import T5Tokenizer
except ImportError as e:
    print("T5Tokenizer requires SentencePiece. Please install it using `pip install sentencepiece`.")
    raise e


# Optional: For backtranslation or paraphrasing
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer


# Config
LABELS = ['I', 'N', 'V', 'E', 'S', 'T']
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-5

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# Dataset class
class INVESTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }
        


# Model
class MultiLabelBERT(nn.Module):
    def __init__(self, hidden_dim=768, output_dim=6):
        super(MultiLabelBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.pooler_output)
        return self.sigmoid(self.fc(x))

# Training loop with early stopping
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device, epochs=10, patience=2):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate(model, val_loader, criterion, device, epoch)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

        #creating logs for plotting
        logs = {'train_loss': [], 'val_loss': []}
        logs['train_loss'].append(total_loss / len(train_loader))
        logs['val_loss'].append(val_loss)


        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
# Evaluation

def evaluate(model, val_loader, criterion, device, epoch=None):
    y_true, y_pred = [], []
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    f1 = f1_score(y_true, y_pred > 0.5, average='macro')
    precision = precision_score(y_true, y_pred > 0.5, average='macro')
    recall = recall_score(y_true, y_pred > 0.5, average='macro')
    roc_auc = roc_auc_score(y_true, y_pred, average='macro')

    if epoch is not None:
        print(f"Epoch {epoch+1}: F1={f1:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, ROC-AUC={roc_auc:.2f}")

    return total_loss / len(val_loader)

#Parsing and backtranslation

# Load paraphrasing model (Option B). This is to improve backtranslation or paraphrasing
para_model_name = "Vamsi/T5_Paraphrase_Paws"
para_tokenizer = T5Tokenizer.from_pretrained(para_model_name)
para_model = AutoModelForSeq2SeqLM.from_pretrained(para_model_name).to(device)

def paraphrase(text, num_return_sequences=1):
    input_text = f"paraphrase: {text}"
    encoding = para_tokenizer.encode_plus(
        input_text, 
        return_tensors="pt", 
        max_length=128, 
        padding=True,
        truncation=True
    ).to(device)
    outputs = para_model.generate(
        **encoding,
        max_length=256,
        num_return_sequences=num_return_sequences,
        num_beams=4,
        do_sample=True, 
        top_k=50,
        top_p=0.95,
        temperature=1.5,
        early_stopping=True
    )
    return [para_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]




# Load dataset
df = pd.read_csv("invest_user_stories.csv")
texts = df['story'].astype(str).fillna("").tolist()
labels = df[LABELS].fillna(0).astype(np.float32).values
labels = (labels > 0).astype(int)

# Stratified split
from skmultilearn.model_selection import IterativeStratification
stratifier = IterativeStratification(n_splits=2, order=1)
train_indices, test_indices = next(stratifier.split(np.array(texts).reshape(-1, 1), labels))
train_texts = [texts[i] for i in train_indices]
test_texts = [texts[i] for i in test_indices]
train_labels = labels[train_indices]
test_labels = labels[test_indices]

# After splitting, check the distribution of labels in train and test sets
def label_stats(name, y):
    # print(f"\n{name} label distribution:") Uncomment this line to see the distribution of labels
    for i, label in enumerate(LABELS):
        counts = np.bincount(y[:, i].astype(int))
      #  print(f"{label}: {dict(enumerate(counts))}")Uncomment this line to see the distribution of labels

label_stats("Train", train_labels)
label_stats("Test", test_labels)

#Identify samples to augments like under performer S
# Filter training examples where 'S' label is 1
s_index = LABELS.index('S')
s_positive_mask = train_labels[:, s_index] == 1
s_positive_texts = [train_texts[i] for i in range(len(train_texts)) if s_positive_mask[i]]
s_positive_labels = train_labels[s_positive_mask]

# 
aug_texts = []
aug_labels = []

for i, text in enumerate(s_positive_texts):
    try:
        new_texts = paraphrase(text, num_return_sequences=2)  # <-- Change to backtranslate(text) if needed
        for new_text in new_texts:
            aug_texts.append(new_text)
            aug_labels.append(s_positive_labels[i])
    except Exception as e:
        print(f"Augmentation error: {e}")

#
aug_labels = np.array(aug_labels)
train_texts += aug_texts
train_labels = np.vstack([train_labels, aug_labels])


# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# DataLoaders
train_dataset = INVESTDataset(train_texts, train_labels, tokenizer)
test_dataset = INVESTDataset(test_texts, test_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# This function is called to train and evaluate the model
def train_and_evaluate():
    model = MultiLabelBERT().to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)

    train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, device, EPOCHS)

    # Load best model after early stopping
    try:
        model.load_state_dict(torch.load("best_model.pt"))
    except Exception as e:
        print(f"Load best model file error: {e}")

    return model

# Threshold tuning

# Evaluate on test set
def optimize_thresholds(model, loader):
    model.eval()
    y_true, y_scores = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)

            y_true.extend(labels.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    thresholds = get_optimal_thresholds(y_true, y_scores, LABELS)
    return thresholds, y_true, y_scores

# Threshold tuning
from sklearn.metrics import roc_curve, auc
# Function to get the best thresholds for each label
def get_optimal_thresholds(y_true, y_probs, labels):
    thresholds = {}
    for i, label in enumerate(labels):
        y_true_col = y_true[:, i]
        y_scores_col = y_probs[:, i]

        if len(np.unique(y_true_col)) < 2:
            print(f"Skipping {label} due to only one class present.")
            thresholds[label] = 0.5
            continue

        fpr, tpr, thresh = roc_curve(y_true_col, y_scores_col)
        youden_index = np.argmax(tpr - fpr)
        best_thresh = float(thresh[youden_index])
        thresholds[label] = best_thresh

    return thresholds

#Evaluate and store model predictions
def evaluate_and_save_results(y_true, y_scores, thresholds, out_file="results_metrics.json"):
        preds = (y_scores > np.array([thresholds[label] for label in LABELS])).astype(int)

        report = {}
        for i, label in enumerate(LABELS):
            report[label] = {
                "f1": round(f1_score(y_true[:, i], preds[:, i]), 3),
                "precision": round(precision_score(y_true[:, i], preds[:, i]), 3),
                "recall": round(recall_score(y_true[:, i], preds[:, i]), 3),
            }

        with open(out_file, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))

# Model and optimization in modular format

#New modualization for metrics visualization
model = train_and_evaluate()
thresholds, y_true, y_scores = optimize_thresholds(model, test_loader)
evaluate_and_save_results(y_true, y_scores, thresholds)

	
# Generate optimized thresholds dynamically based on validation data
#invest_labels = ['I', 'N', 'V', 'E', 'S', 'T']
thresholds = get_optimal_thresholds(y_true, y_scores, LABELS)

with open("thresholds_log.json", "a") as f:
    json.dump({str(datetime.datetime.now()): thresholds}, f)
    f.write("\n")
    
threshold_array = np.array([thresholds[label] for label in LABELS])#use thresholds to make predictions
# Final evaluation using optimized thresholds
preds = (y_scores > [thresholds[label] for label in LABELS]).astype(int)

print("\nEvaluation Report:")
for i, label in enumerate(LABELS):
    print(f"{label} - F1: {f1_score(y_true[:, i], preds[:, i]):.2f}, Precision: {precision_score(y_true[:, i], preds[:, i]):.2f}, Recall: {recall_score(y_true[:, i], preds[:, i]):.2f}")

# Unit test for paraphrasing
def test_paraphrase(test_text):
    #test_text = "As a user, I want to log in so that I can access the dashboard."
    paraphrased_texts = paraphrase(test_text, num_return_sequences=2)
    assert len(paraphrased_texts) == 2, "Expected 2 paraphrased texts"
    for text in paraphrased_texts:
        assert isinstance(text, str), f"Expected string, got {type(text)}: {text}"
    #print("Paraphrase test passed!")
if __name__ == "__main__":
    #test_paraphrase()
    test_text = "As a user, I want to log in so that I can access the dashboard."
    test_paraphrase(test_text)
    print("Paraphrase test passed!")


# Save model
save_path = "models/invest_bert"
os.makedirs(save_path, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_path, "invest_bert_model.pt"))
tokenizer.save_pretrained(save_path)
torch.save(model.state_dict(), "invest_bert_model.pt")
tokenizer.save_pretrained("invest_tokenizer")

#Save model and tokenizer for paraphrasing
para_model.save_pretrained("invest_paraphrase_model")
para_tokenizer.save_pretrained("invest_paraphrase_tokenizer")

# Generate metrics visualization
import matplotlib.pyplot as plt

def plot_metrics(logs):
    epochs = range(1, len(logs['train_loss']) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, logs['train_loss'], label='Train Loss')
    plt.plot(epochs, logs['val_loss'], label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_plot.png")
    plt.show()






