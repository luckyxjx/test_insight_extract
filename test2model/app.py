import streamlit as st
from summarizer import load_model, summarize_text, load_vocab
import torch

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained fine-tuned model and vocab
model = load_model('finetuned_model.pth', device)
vocab = load_vocab("vocab.pkl")

# Define Streamlit app
def main():
    st.title("Text Summarizer")
    st.write("Enter text for summarization:")

    input_text = st.text_area("Input Text")

    if st.button('Summarize'):
        if input_text:
            summary = summarize_text(model, input_text, vocab, device)
            st.write("Summary:")
            st.write(summary)
        else:
            st.write("Please enter text to summarize.")

if __name__ == "__main__":
    main()
