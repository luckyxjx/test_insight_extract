import torch
import torch.optim as optim
import torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import re
import pickle  # To save vocab

# Define helper functions for text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Load the dataset and preprocess
def load_and_preprocess_data(max_samples=100):
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    articles = dataset["train"]["article"][:max_samples]
    summaries = dataset["train"]["highlights"][:max_samples]

    clean_articles = [preprocess_text(article) for article in articles]
    clean_summaries = [preprocess_text(summary) for summary in summaries]

    # Split into train and validation sets
    train_articles, val_articles, train_summaries, val_summaries = train_test_split(
        clean_articles, clean_summaries, test_size=0.2, random_state=42
    )

    return train_articles, val_articles, train_summaries, val_summaries

# Define the vocabulary and padding
def build_vocab(texts, vocab_size=10000):
    word_counts = {}
    for text in texts:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1
    vocab = {word: idx for idx, (word, _) in enumerate(sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:vocab_size])}
    vocab["<unk>"] = vocab_size
    vocab["<pad>"] = vocab_size + 1
    return vocab

def tokenize_and_pad(data, vocab, max_len=512, pad_idx=0):
    tokenized_data = []
    for sentence in data:
        tokens = sentence.split()
        token_ids = [vocab.get(word, vocab.get("<unk>")) for word in tokens]  # Handle unknown words
        token_ids = token_ids[:max_len]  # Truncate to max_len if too long
        token_ids += [pad_idx] * (max_len - len(token_ids))  # Pad to max_len
        tokenized_data.append(torch.tensor(token_ids))
    return torch.stack(tokenized_data)  # Stack tensors to a batch

# Encoder and Decoder with Attention mechanism
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

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # Repeat hidden state for each word in encoder output
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]
        return torch.softmax(attention, dim=1)

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
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden[-1], encoder_outputs)
        a = a.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1))
        return prediction, hidden, cell

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
def train(model, train_data, optimizer, criterion, clip=1):
    model.train()
    epoch_loss = 0
    for src, trg in train_data:
        optimizer.zero_grad()
        output = model(src, trg)
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_data)

# Main function to fine-tune the model
def fine_tune():
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
    train_articles, val_articles, train_summaries, val_summaries = load_and_preprocess_data(max_samples=100)

    # Build vocabulary
    vocab = build_vocab(train_articles + train_summaries)

    # Tokenize and pad data
    train_articles_padded = tokenize_and_pad(train_articles, vocab, max_len=512, pad_idx=PAD_IDX)
    train_summaries_padded = tokenize_and_pad(train_summaries, vocab, max_len=512, pad_idx=PAD_IDX)

    # Save vocab
    with open("vocab.pkl", "wb") as vocab_file:
        pickle.dump(vocab, vocab_file)

    # Model initialization
    encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    attention = Attention(HID_DIM)
    decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, attention, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Prepare training data
    train_data = [(train_articles_padded, train_summaries_padded)]

    # Train the model
    N_EPOCHS = 10
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_data, optimizer, criterion)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "finetuned_model.pth")

if __name__ == "__main__":
    fine_tune()
