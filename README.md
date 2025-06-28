# 🧠 Transformer-Based Math LLM (Prototype)

## Overview
This project is a lightweight language model built from scratch in **NumPy**, designed to solve basic arithmetic problems like `86 - 11 =`. The goal was to understand and implement the core mechanics of a transformer-style LLM, including attention, tokenization, and training logic — all without using deep learning frameworks like PyTorch or TensorFlow.

---

## Features Implemented

### Tokenizer
- Character-level tokenization
- Supports special tokens: `<PAD>`, `<START>`, `<END>`, `<UNK>`
- Includes `encode()` and `decode()` methods

### Dataset Loader
- Loads question-answer pairs
- Encodes text into token sequences
- Pads or truncates sequences to a fixed length

### Model Architecture
- Token embeddings and positional encodings
- Multi-head self-attention (single layer)
- Feed-forward network (FFN)
- Layer normalization
- Output projection layer

### Training Loop
- Cross-entropy loss function
- Manual gradient updates using pure NumPy
- Logs training and validation loss over epochs

### Inference
- Greedy decoding implemented in `inference.py`
- **Known issue:** Currently outputs only "1" for all inputs due to decoding/token alignment bug

---

## Things Learned
- Gained hands-on experience implementing transformer internals
- Learned how tokenization, attention, and loss functions work at a low level
- Discovered challenges in inference decoding and token alignment
- Improved numerical debugging and backprop intuition in NumPy


---

## How to Run

1. Clone the repo  
   `git clone https://github.com/yourusername/llm-math-transformer`
2. Install NumPy
   `pip install numpy`
3. Train the model  
   `python train.py`
4. Run inference (experimental)  
   `python inference.py`

---

## Next Steps

- Fix inference output and decoding logic
- Improve positional encoding and batching
- Support longer sequences and varied math formats
- Add performance metrics (e.g., accuracy)

---

## Why This Project?

Even though most ML work uses high-level frameworks, this project was built to:
- Develop a deep, practical understanding of how LLMs function
- Build intuition for training dynamics, attention, and tokenization
- Become a better applied data scientist through low-level experimentation
---

## License

MIT License – free to use, fork, and build upon.
