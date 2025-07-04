import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import re

def get_hn_titles(n=1000):
    ids = requests.get("https://hacker-news.firebaseio.com/v0/newstories.json").json()[:n]
    titles = []
    for i in ids:
        item = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{i}.json").json()
        if item and "title" in item:
            titles.append(item["title"])
    return titles

positive_words = {"great", "love", "success", "awesome", "win", "amazing", "good", "wow", "interesting"}
negative_words = {"fail", "bug", "down", "dead", "angry", "bad", "hate", "terrible", "crash"}

def label_sentiment(title):
    words = re.findall(r"\w+", title.lower())
    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)
    if pos > neg:
        return 0
    elif neg > pos:
        return 2
    else:
        return 1

def simple_tokenizer(text):
    return re.findall(r"\w+", text.lower())

def build_vocab(dataset, min_freq=1):
    word_counts = {}
    for title, _ in dataset:
        for token in simple_tokenizer(title):
            word_counts[token] = word_counts.get(token, 0) + 1
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode(text, vocab, max_len=20):
    tokens = simple_tokenizer(text)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens[:max_len]]
    return ids + [vocab["<PAD>"]] * (max_len - len(ids))

class HNDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title, label = self.data[idx]
        x = torch.tensor(encode(title, self.vocab), dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

class MoodClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        return self.fc(pooled)

titles = get_hn_titles(1200)
labeled = [(t, label_sentiment(t)) for t in titles]
random.shuffle(labeled)

split = int(0.8 * len(labeled))
train_data, test_data = labeled[:split], labeled[split:]
vocab = build_vocab(train_data)

train_ds = HNDataset(train_data, vocab)
test_ds = HNDataset(test_data, vocab)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MoodClassifier(len(vocab), embed_dim=64, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
print(f"Test Accuracy: {100 * correct / total:.2f}%")

idx2label = {0: "positive", 1: "neutral", 2: "negative"}

def predict(title):
    model.eval()
    x = torch.tensor([encode(title, vocab)], dtype=torch.long).to(device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).item()
    return idx2label[pred]

with open("hn_moods.txt", "w", encoding="utf-8") as f:
    for title in titles:
        mood = predict(title)
        f.write(f"{title} -> {mood}\n")
