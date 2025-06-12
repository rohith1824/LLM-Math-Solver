# LLM Math Solver (NumPy)

## Project Overview

LLM Math Solver is a lightweight, transformer-based language model built entirely from scratch using NumPy. The primary focus is to gain a deep, hands-on understanding of the underlying mechanics of modern LLM architectures by implementing each component step by step. This solver is specifically tailored to handle simple arithmetic and math problems—such as addition, subtraction, multiplication, and division—and to produce accurate numerical answers.

## Goals

- **Core Implementation:** Build transformer building blocks — token embeddings, positional encodings, multi-head self-attention, layer normalization, and feed-forward networks — using only NumPy.
- **Math-Focused Training:** Create or generate a minimal dataset of simple math expressions and train the model to map questions (e.g., “What is 2 + 2?”) to correct answers (e.g., “4”).
- **Modular Codebase:** Organize code into reusable modules under `src/`, with clear interfaces for each transformer component.
- **Reproducible Experiments:** Include Jupyter notebooks demonstrating training loops, loss curves, and evaluation metrics for math tasks.
- **Testing & Validation:** Provide unit tests in `tests/` to verify the correctness of each module and the end-to-end model behavior.
- **Lightweight & Transparent:** Avoid external deep-learning frameworks — everything must be transparent NumPy operations to build intuition and mathematical understanding.

*This project is designed as both a learning tool and a foundation for experimenting with simple LLM capabilities on arithmetic tasks.*
