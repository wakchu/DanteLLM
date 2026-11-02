# MyLLM - Dante GPT

A character-level language model (GPT) built from scratch using PyTorch. This model is trained on Dante Alighieri's *Divina Commedia* and is designed for educational purposes.

## How it Works

This model is a **Decoder-only Transformer**, the same architecture behind GPT-2/3/4, but scaled down (~3.2M parameters) to run efficiently on standard hardware.

### Key Components:
1.  **Tokenization**: Uses character-level encoding. Every unique character in the Italian text (letters, punctuation, spaces) is mapped to an integer.
2.  **Embeddings**: Each token is converted into a 256-dimensional vector. We also add **Position Embeddings** so the model knows the order of characters.
3.  **Self-Attention**: The "brain" of the model. It allows each character to look at previous characters to understand context and predict the next one.
4.  **MLP (Feed-Forward)**: Processes the information gathered by the attention heads.
5.  **Training**: The model learns by predicting the next character in a sequence. We use **Cross-Entropy Loss** to measure its mistakes and **AdamW** to update its weights.

## Requirements

This project uses [uv](https://docs.astral.sh/uv/) for fast and reliable dependency management.

- Python 3.12+
- PyTorch
- tqdm (for progress bars)

## Getting Started

### 1. Setup
Clone the repository and install dependencies:
```bash
cd DanteLLM
uv sync
```

#### GPU & CPU Support
`pyproject.toml` is pre-configured to install the CUDA-enabled PyTorch build (`cu126`) on Windows and Linux via `uv sync` / `uv run`, which natively supports both NVIDIA GPUs and CPU execution (as well as Apple Silicon `mps` on macOS). No manual PyTorch reinstallation is required.

### 2. Training
To start training the model on the provided Dante dataset:
```bash
uv run train.py
```
*Note: The script automatically uses a compatible **CUDA GPU** (or Apple Silicon `mps`) if available, and safely falls back to the **CPU** on older hardware or systems without a dedicated GPU. During training, the best checkpoint based on validation loss is automatically saved as `model_best.pt`.*

### 3. Text Generation
Once training starts, checkpoints are saved every 1000 steps, and the best validation model is saved to `model_best.pt`. You can generate text from a checkpoint at any time:
```bash
uv run generate.py model_best.pt --prompt "Nel mezzo del cammin" --max_new_tokens 300 --temperature 0.75
```
This is an example:
<img width="937" height="299" alt="image" src="https://github.com/user-attachments/assets/585c703f-3167-4bc5-bfb8-8ce39583db2e" />


## Dataset
The model is trained on `data/divina_commedia.txt`, which contains the complete text of Dante's *Divine Comedy* (Inferno, Purgatorio, Paradiso).
- **Total characters**: ~535,500
- **Unique characters (Vocab)**: 77

## Model Configuration
- **Layers**: 4
- **Attention Heads**: 4
- **Embedding Dim**: 256
- **Context Window (Block Size)**: 256 tokens
- **Dropout**: 0.2

## Acknowledgments
Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and the "LLM from Scratch" tutorial.
