import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.data.poker_dataset import PokerDataset
from src.models.poker_transformer import PokerTransformer
from src.vocab.poker_vocab import get_vocab_sizes

# Config
TRAIN_DATA_PATH = "dataset/processed/train.csv"
TEST_DATA_PATH = "dataset/processed/test.csv"
MAX_ACTION_LEN = 15  # only for this dataset
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 5  # early stopping patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
df_train = pd.read_csv(TRAIN_DATA_PATH)
df_train.fillna({"action_sequence": ""}, inplace=True)

df_test = pd.read_csv(TEST_DATA_PATH)
df_test.fillna({"action_sequence": ""}, inplace=True)

# Create Dataset and Dataloader
train_dataset = PokerDataset(df_train, max_action_len=MAX_ACTION_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = PokerDataset(df_test, max_action_len=MAX_ACTION_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Get Vocab Sizes
vocab_sizes = get_vocab_sizes()

# Initialize Model
model = PokerTransformer(
    vocab_sizes=vocab_sizes,
    d_model=128,
    nhead=8,
    num_layers=4,
    dim_feedforward=512,
    dropout=0.1,
    max_seq_len=8 + MAX_ACTION_LEN,  # 8 = round + 7 cards
    num_classes=3,  # C, R, F
).to(DEVICE)
model = torch.compile(model)

# Optimizer & Loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training Loop
best_model_state_dict = None
best_f1 = 0
best_epoch = 0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_total = 0
    total_loss = 0

    for batch_x, batch_type, batch_y in train_loader:
        batch_x = batch_x.to(DEVICE)
        batch_type = batch_type.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(batch_x, batch_type)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        train_total += batch_x.size(0)
        total_loss += loss.item() * batch_x.size(0)

    avg_loss = total_loss / train_total

    model.eval()
    all_preds = []
    all_labels = []
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for batch_x, batch_type, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            batch_type = batch_type.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            logits = model(batch_x, batch_type)
            loss = criterion(logits, batch_y)

            val_total += batch_x.size(0)
            val_loss += loss.item() * batch_x.size(0)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_val_loss = val_loss / val_total

    # metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
        f"Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
    )

    # Early Stopping
    if f1 > best_f1:
        best_f1 = f1
        best_epoch = epoch
        patience_counter = 0
        best_model_state_dict = model.state_dict()
    else:
        patience_counter += 1
    if patience_counter >= PATIENCE:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break

# Save the Model
os.makedirs("checkpoints", exist_ok=True)
if best_model_state_dict is not None:
    torch.save(best_model_state_dict, "checkpoints/poker_transformer_best.pt")
    print("Model saved to checkpoints/poker_transformer_best.pt")
    print(f"Best model was from epoch {best_epoch} with F1 = {best_f1:.4f}")

torch.save(model.state_dict(), f"checkpoints/poker_transformer_last.pt")
print(f"Model saved to checkpoints/poker_transformer_last.pt")
