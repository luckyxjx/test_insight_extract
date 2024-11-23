import streamlit as st
import torch
from model import Seq2Seq, Encoder, Decoder, Attention, preprocess_text, tokenize_and_pad

# Load the model and tokenizer
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the model weights from the saved file
    model = Seq2Seq(Encoder(10000, 256, 512, 2, 0.5), Decoder(10000, 256, 512, 2, Attention(512), 0.5), device)
    model.load_state_dict(torch.load('finetuned_model.pth'))  # Make sure the model path is correct
    model.to(device)
    model.eval()

    # Load the vocab from the saved file
    vocab = torch.load('vocab.pth')  # Ensure the vocab path is correct
    return model, vocab, device

# Preprocess input text
def preprocess_input(text):
    return preprocess_text(text)

# Summarize the input text
def summarize_text(model, device, vocab, input_text):
    input_text = preprocess_input(input_text)
    tokens = torch.tensor([tokenize_and_pad([input_text], vocab)]).to(device)
    output = model(tokens, tokens)  # Modify as per your decoder input/output
    summary = ' '.join([list(vocab.keys())[idx.item()] for idx in output.argmax(dim=-1).squeeze()])
    return summary

# Streamlit UI
def main():
    st.title("Text Summarization")
    st.write("Enter text and get the summarized version.")

    input_text = st.text_area("Input text", height=300)

    if st.button("Summarize"):
        if input_text:
            model, vocab, device = load_model()
            summary = summarize_text(model, device, vocab, input_text)
            st.write("Summary:")
            st.write(summary)
        else:
            st.write("Please enter text for summarization.")

if __name__ == "__main__":
    main()
