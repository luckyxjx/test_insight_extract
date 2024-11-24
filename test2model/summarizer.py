import torch
import pickle
from finetune_summarizer import Seq2Seq, Encoder, Decoder, Attention, preprocess_text, tokenize_and_pad


# Load vocab correctly
def load_vocab(vocab_path="vocab.pkl"):
    with open(vocab_path, "rb") as vocab_file:
        vocab = pickle.load(vocab_file)
    return vocab


# Load the fine-tuned model
def load_model(model_path, device):
    model = Seq2Seq(
        Encoder(10000, 256, 512, 2, 0.5),
        Decoder(10000, 256, 512, 2, Attention(512), 0.5),
        device
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    return model


# Summarizing function
def summarize_text(model, input_text, vocab, device, max_len=512):
    input_text = preprocess_text(input_text)

    # Tokenize and pad the input
    tokenized_input = tokenize_and_pad([input_text], vocab, max_len=max_len, pad_idx=vocab["<pad>"]).to(device)

    # Generate summary
    with torch.no_grad():
        output = model(tokenized_input.squeeze(0), tokenized_input.squeeze(0), teacher_forcing_ratio=0)
        predicted_ids = output.argmax(dim=2)

    # Convert predicted tokens back to text
    summary = ' '.join([list(vocab.keys())[list(vocab.values()).index(idx.item())] for idx in predicted_ids[0] if
                        idx.item() != vocab["<pad>"]])
    return summary
