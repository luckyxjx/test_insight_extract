import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from collections import Counter
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import re
import numpy as np

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Tokenize and pad sequences
def tokenize_and_pad(data, vocab, pad_idx=0):
    tokenized_data = [torch.tensor([vocab.get(word, vocab['<unk>']) for word in sentence.split()]) for sentence in data]
    padded_data = torch.nn.utils.rnn.pad_sequence(tokenized_data, batch_first=True, padding_value=pad_idx)
    return padded_data

# Build vocabulary
def build_vocab(data, vocab_size=10000):
    counter = Counter()
    for sentence in data:
        counter.update(sentence.split())
    vocab = {word: idx for idx, (word, _) in enumerate(counter.most_common(vocab_size))}
    vocab['<pad>'] = 0  # Add padding token to vocab
    vocab['<unk>'] = 1  # Add unknown token
    return vocab

# Encoder definition
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell

# Attention mechanism
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # Repeat hidden across the sequence length
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]
        return torch.softmax(attention, dim=1)

# Decoder definition
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, attention, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]
        a = self.attention(hidden[-1], encoder_outputs)  # [batch_size, src_len]
        a = a.unsqueeze(1)  # [batch_size, 1, src_len]
        weighted = torch.bmm(a, encoder_outputs)  # [batch_size, 1, hid_dim]
        rnn_input = torch.cat((embedded, weighted), dim=2)  # [batch_size, 1, emb_dim + hid_dim]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # [batch_size, 1, hid_dim]
        prediction = self.fc_out(torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1))
        return prediction, hidden, cell

# Seq2Seq Model definition
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1
        return outputs

# Training function
def train(model, iterator, optimizer, criterion, clip=1):
    model.train()
    epoch_loss = 0
    for src, trg in iterator:
        optimizer.zero_grad()
        output = model(src, trg)
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Save the model and vocabulary
def save_model(model, vocab, model_path='finetuned_model.pth', vocab_path='vocab.pth'):
    torch.save(model.state_dict(), model_path)
    torch.save(vocab, vocab_path)

# Main function to train and save the model
def main():
    # Parameters
    INPUT_DIM = 10000
    OUTPUT_DIM = 10000
    EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.5
    PAD_IDX = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    articles = dataset["train"]["article"][:100]  # Using first 100 samples
    summaries = dataset["train"]["highlights"][:100]

    clean_articles = [preprocess_text(article) for article in articles]
    clean_summaries = [preprocess_text(summary) for summary in summaries]
    train_articles, val_articles, train_summaries, val_summaries = train_test_split(
        clean_articles, clean_summaries, test_size=0.2, random_state=42
    )

    vocab = build_vocab(train_articles + train_summaries)
    train_articles_padded = tokenize_and_pad(train_articles, vocab)
    train_summaries_padded = tokenize_and_pad(train_summaries, vocab)

    # Initialize model
    encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    attention = Attention(HID_DIM)
    decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, attention, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Training loop
    N_EPOCHS = 10
    for epoch in range(N_EPOCHS):
        train_loss = train(model, [(train_articles_padded, train_summaries_padded)], optimizer, criterion)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

    # Save the trained model and vocab
    save_model(model, vocab)
    print("Model and vocab saved.")

if __name__ == "__main__":
    main()
