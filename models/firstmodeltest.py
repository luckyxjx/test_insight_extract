import re
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


# Load CNN/DailyMail dataset and preprocess the first 100 samples
def load_and_preprocess_data(max_samples=100):
    # Load dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    # Extract articles and summaries
    articles = dataset["train"]["article"][:max_samples]
    summaries = dataset["train"]["highlights"][:max_samples]

    # Preprocess text (clean and lowercase)
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        return text

    clean_articles = [preprocess_text(article) for article in articles]
    clean_summaries = [preprocess_text(summary) for summary in summaries]

    # Split into train and validation sets
    train_articles, val_articles, train_summaries, val_summaries = train_test_split(
        clean_articles, clean_summaries, test_size=0.2, random_state=42
    )

    return train_articles, val_articles, train_summaries, val_summaries


# Tokenize and pad the articles and summaries
def tokenize_and_pad(data, vocab_size=10000, pad_idx=0):
    tokenized_data = [torch.randint(1, vocab_size, (len(sentence.split()),)) for sentence in data]
    padded_data = pad_sequence(tokenized_data, batch_first=True, padding_value=pad_idx)
    return padded_data


# Define the Encoder
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


# Define the Attention mechanism
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hid_dim], encoder_outputs: [batch_size, src_len, hid_dim]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, hid_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]
        return torch.softmax(attention, dim=1)


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, attention, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # Fix input size: LSTM expects (hid_dim + emb_dim) instead of (hid_dim * 2 + emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        # Adjust the final output layer to match the correct concatenated dimensions
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch_size], hidden: [n_layers, batch_size, hid_dim]
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]

        # Attention calculation
        a = self.attention(hidden[-1], encoder_outputs)  # [batch_size, src_len]
        a = a.unsqueeze(1)  # [batch_size, 1, src_len]

        # Weighted sum of attention
        weighted = torch.bmm(a, encoder_outputs)  # [batch_size, 1, hid_dim]

        # Concatenate the embedded input and the attention-weighted context vector
        rnn_input = torch.cat((embedded, weighted), dim=2)  # [batch_size, 1, emb_dim + hid_dim]

        # Pass through the LSTM
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # [batch_size, 1, hid_dim]

        # Generate prediction
        prediction = self.fc_out(
            torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1))  # [batch_size, output_dim]

        return prediction, hidden, cell


# Define the Seq2Seq Model
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

        input = trg[:, 0]  # First input to the decoder is the <sos> token

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
    for i, (src, trg) in enumerate(iterator):
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


# Main function
def main():
    # Parameters
    INPUT_DIM = 10000  # Adjust these as needed based on the vocab size
    OUTPUT_DIM = 10000
    EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.5
    PAD_IDX = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data
    train_articles, val_articles, train_summaries, val_summaries = load_and_preprocess_data(max_samples=100)

    # Tokenize and pad the data
    train_articles_padded = tokenize_and_pad(train_articles, vocab_size=INPUT_DIM, pad_idx=PAD_IDX)
    train_summaries_padded = tokenize_and_pad(train_summaries, vocab_size=OUTPUT_DIM, pad_idx=PAD_IDX)

    # Model initialization
    encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    attention = Attention(HID_DIM)
    decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, attention, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Prepare training data
    train_iterator = [(train_articles_padded, train_summaries_padded)]  # Add more batches if needed

    # Train the model
    N_EPOCHS = 10
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_iterator, optimizer, criterion)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")


if __name__ == "__main__":
    main()
