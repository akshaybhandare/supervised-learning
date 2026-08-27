# Supervised Learning: Personal Note Category Classifier

A fast, lightweight, and robust multi-class text classifier that categorizes personal notes into **5 core categories**:
- ✈️ **`travel`** — Trips, destinations, flights, trains, hotel reservations, itineraries
- 🍕 **`food`** — Cooking, recipes, restaurants, dining, baking, groceries, street food
- 🏋️ **`health`** — Workouts, gym, running, lifting, mobility, sleep, nutrition, recovery
- 💻 **`technology`** — Software, coding, cloud, APIs, Docker, Kubernetes, Linux, databases
- 💰 **`finance`** — Budgeting, investing, index funds, Roth IRA, 401(k), savings, debt, taxes

---

## 🧠 Architecture Overview

The system combines **pre-trained Transformer sentence representations** with a **custom 2-layer Neural Network** implemented in pure NumPy with backpropagation and mini-batch SGD:

$$\text{Raw Text} \xrightarrow{\text{all-MiniLM-L6-v2}} \underbrace{X \in \mathbb{R}^{384}}_{\text{Pre-trained Embedding}} \xrightarrow{W_1, b_1} \underbrace{z_1 \xrightarrow{\text{ReLU}} h_1}_{\text{64 Hidden Neurons}} \xrightarrow{W_2, b_2} \underbrace{\text{logits} \xrightarrow{\text{Softmax}} \text{Probabilities}}_{\text{5 Categories}}$$

### Key Design Highlights
1. **Context-Aware Embeddings:** Uses `all-MiniLM-L6-v2` (Sentence-Transformers) to produce dense 384-dimensional semantic vectors. Automatically handles subwords, synonyms, typos, and sentence context.
2. **Custom NumPy Neural Network:** Implements He initialization, ReLU activation, cross-entropy loss, and mini-batch backpropagation from scratch with zero framework bloat.
3. **99.6% Evaluation Accuracy:** Converges in seconds on CPU and generalizes across natural, unseen human phrasing.

---

## 📊 Dataset Statistics

- **Total Examples:** 5,500
  - `train.jsonl`: **5,000 examples** (1,000 balanced notes per category)
  - `eval.jsonl`: **500 examples** (100 balanced notes per category)
- **Format:** JSON Lines (`{"text": "...", "label": "..."}`)

---

## 🚀 Quickstart with `uv`

This project is configured with [`uv`](https://github.com/astral-sh/uv) for fast, deterministic dependency management.

### 1. Install Dependencies
```bash
uv sync
```

### 2. Train the Model
Trains the 2-layer classifier on top of the sentence embeddings and saves the weights to `model.pkl`:
```bash
uv run ai.py
```

### 3. Run Live Interactive Inference
Launch the interactive CLI prompt to classify notes in real-time:
```bash
uv run infer.py
```

**Example interactive session:**
```text
==================================================
Live Note Classifier (all-MiniLM-L6-v2 + Neural Network)
Type 'quit', 'exit', or 'q' to stop
==================================================

Enter note: Modern chips are awesome I can eat them forever
Prediction:  FOOD (99.7% confidence)
Probabilities:
  food        :  99.7%  █████████████████████████████
  technology  :   0.1%  
  travel      :   0.1%  
  health      :   0.0%  
  finance     :   0.0%  
--------------------------------------------------

Enter note: Going to travel to Chennai on Chennai Express
Prediction:  TRAVEL (98.1% confidence)
Probabilities:
  travel      :  98.1%  █████████████████████████████
  technology  :   1.4%  
  food        :   0.3%  
  health      :   0.1%  
  finance     :   0.1%  
--------------------------------------------------
```

You can also pass single notes directly via command line arguments:
```bash
uv run infer.py "Booked a train ticket and hotel for the trip"
# -> Prediction: TRAVEL (100.0% confidence)
```

---

## 🛠️ Programmatic Python Usage

```python
from infer import NoteClassifier

classifier = NoteClassifier("model.pkl")
result = classifier.predict("Hit a new personal record on barbell squats: 315 lbs today")

print(result["category"])     # 'health'
print(result["confidence"])   # 0.998...
print(result["probabilities"])
```

---

## 📁 Repository Structure

```text
.
├── pyproject.toml        # uv project configuration and dependencies
├── uv.lock               # Deterministic dependency lockfile
├── train.jsonl           # 5,000 training examples
├── eval.jsonl            # 500 validation examples
├── generate_dataset.py   # Dataset generator script
├── ai.py                 # Central training pipeline (SentenceTransformer + 2-layer NN)
├── infer.py              # Live interactive inference utility
├── tokenizer.py          # Word-level tokenizer baseline
├── embedding.py          # Standalone NumPy embedding matrix baseline
├── model.pkl             # Serialized trained model weights & metadata
└── README.md             # Project documentation
```

---

## 📜 License

MIT License
