# Baby-GPT: Pure NumPy Causal Transformer Q&A Model

A complete, lightweight **Autoregressive Causal Transformer (GPT)** built from scratch using **pure NumPy** with zero PyTorch / deep learning framework dependencies.

---

## 🧠 Architecture Overview

$$\text{Token IDs } [t_0, t_1, \dots, t_T] \xrightarrow{\text{Embedding}} X_0 \xrightarrow{\text{Causal Self-Attention}} X_1 \xrightarrow{\text{FeedForward (FFN)}} X_2 \xrightarrow{\text{LM Head}} \text{Logits} \xrightarrow{\text{Softmax}} \text{Next Token Probabilities}$$

### Core Transformer Components (Implemented from Scratch in NumPy)

1. **Token + Positional Embeddings:**
   $$X_0 = W_{\text{tok}}[t] + W_{\text{pos}}[\text{pos}]$$
2. **Causal Masked Self-Attention:**
   $$Q = X_0 W_Q + b_Q, \quad K = X_0 W_K + b_K, \quad V = X_0 W_V + b_V$$
   $$\text{Scores} = \frac{Q K^T}{\sqrt{d_{\text{model}}}} + \text{CausalMask} \quad (\text{future tokens masked to } -\infty)$$
   $$\text{AttnWeights} = \text{Softmax}(\text{Scores}), \quad X_1 = X_0 + \text{AttnWeights} \cdot V \cdot W_O + b_O$$
3. **FeedForward Network (FFN) with Residual Connection:**
   $$X_2 = X_1 + \text{ReLU}(X_1 W_{\text{ff1}} + b_{\text{ff1}}) W_{\text{ff2}} + b_{\text{ff2}}$$
4. **Language Model Output Head:**
   $$\text{Logits} = X_2 W_{\text{head}} + b_{\text{head}} \quad \in \mathbb{R}^{T \times V}$$
5. **Autoregressive Generation Loop (`generate`):**
   Feeds the current prompt, computes logits for the last token, samples the next token, appends it, and repeats until the `<eos>` token is emitted.

---

## 🚀 How to Run

Navigate to the `baby_gpt/` directory:

```bash
cd baby_gpt
```

### 1. (Optional) Generate Training Dataset
```bash
uv run python qa_data.py
```

### 2. Train Baby-GPT
```bash
uv run python train.py
```
```text
=== Baby-GPT Q&A Language Model (Pure NumPy) ===
Vocabulary Size: 548 tokens | Training Examples: 1500

Epoch  1/40 | Cross-Entropy Loss: 4.1249
Epoch 10/40 | Cross-Entropy Loss: 0.4510
Epoch 20/40 | Cross-Entropy Loss: 0.2345
Epoch 40/40 | Cross-Entropy Loss: 0.2190

✓ Model successfully saved to 'baby_gpt.pkl'
```

---

### 3. Interactive Q&A Chat Loop
```bash
uv run python chat.py
```
```text
============================================================
🤖 Baby-GPT Q&A Language Model (Pure NumPy Transformer)
Ask questions about Travel, Food, Health, Tech, or Finance!
Type 'quit', 'exit', or 'q' to stop
============================================================

Ask a question: Where did you stay in Paris?
Baby-GPT: at a boutique hotel with a balcony overlooking the eiffel tower.

Ask a question: What did you cook for Sunday dinner?
Baby-GPT: homemade butter chicken with garlic naan for the whole family.

Ask a question: What is your bench press record?
Baby-GPT: hit a new personal record of 225 lbs for 5 clean reps.

Ask a question: What framework did you use for the API?
Baby-GPT: built a high - performance rest api using python and fastapi.

Ask a question: What index fund do you invest in?
Baby-GPT: invested in the low - cost s & p 500 index fund voo every month.
```

Or pass a single question directly:
```bash
uv run python chat.py "What did you see in Iceland?"
# -> Baby-GPT: the northern lights appeared at night and we drove past frozen waterfalls.
```

---

## 📁 Folder Structure

```text
baby_gpt/
├── qa_data.py       # Dataset generator (1,500 prompt-response pairs)
├── qa_data.jsonl    # Training data
├── tokenizer.py     # QATokenizer (<bos>, <eos>, <pad>, <unk>)
├── model.py         # BabyGPT (Causal Attention, FFN, Backprop in pure NumPy)
├── train.py         # Training loop with sample generation & checkpointing
├── chat.py          # Interactive CLI Q&A inference engine
├── baby_gpt.pkl     # Trained model checkpoint
└── README.md        # Documentation
```
