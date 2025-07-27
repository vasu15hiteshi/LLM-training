import streamlit as st
import torch
from minbpe import RegexTokenizer
from transformer.model import GPTLanguageModel
import os

# --- Model and Tokenizer Paths ---
TOKENIZER_PATH = os.path.join("tokenizer", "my_tokenizer.model")
CHECKPOINT_PATH = os.path.join("fine_tuning", "checkpoint_19.pth")  # Change if you want a different checkpoint

# --- Load Tokenizer ---
tokenizer = RegexTokenizer()
tokenizer.load(TOKENIZER_PATH)

def get_vocab_size(tokenizer):
    vocab = tokenizer.vocab
    special_tokens = tokenizer.special_tokens
    return len(vocab) + len(special_tokens)

# --- Model Hyperparameters (must match training) ---
block_size = 512
n_embd = 512
n_head = 8
n_layer = 4
vocab_size = get_vocab_size(tokenizer)
dropout = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
@st.cache_resource(show_spinner=True)
def load_model():
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        device=device,
        ignore_index=tokenizer.special_tokens.get("<|padding|>", -100),
    ).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    # Remove _orig_mod. prefix if present
    cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    return model

model = load_model()

# --- Helper: Format input for model ---
def get_input_tokens(message: str) -> torch.Tensor:
    # Add special tokens if your training used them
    input_text = f"<|startoftext|>{message}<|separator|>"
    input_tokens = tokenizer.encode(input_text, allowed_special="all")
    input_tokens = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
    return input_tokens

# --- Streamlit UI ---
st.title("💬 Chat with My Custom Model")
st.write("This model is trained to talk like me! Type a message below and see how it responds.")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", "Hello!")
if st.button("Send"):
    with st.spinner("Generating response..."):
        input_tokens = get_input_tokens(user_input)
        model_answer = ""
        # Generate until <|endoftext|> or max tokens
        while True:
            output_tokens = model.generate(input_tokens=input_tokens, max_new_tokens=1)
            last_token = output_tokens[0, -1].item()
            if last_token == tokenizer.special_tokens.get("<|endoftext|>"):
                break
            input_tokens = torch.cat((input_tokens, output_tokens[:, -1:]), dim=1)
            model_answer += tokenizer.decode([last_token])
            if output_tokens.shape[1] > block_size:
                break
        st.session_state.history.append((user_input, model_answer))

# Display chat history
for user, bot in st.session_state.history:
    st.markdown(f"**You:** {user}")
    st.markdown(f"**Bot:** {bot}") 