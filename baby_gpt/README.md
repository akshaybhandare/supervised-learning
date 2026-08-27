# Baby-GPT: 2-Stage Full GPT Lifecycle (Pre-training & Post-training / SFT)

A pure **NumPy Causal Transformer** that faithfully demonstrates the exact **2-stage lifecycle** of modern Large Language Models (like LLaMA and ChatGPT) from scratch.

---

## 🧠 The 2 Stages of GPT

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Pre-training (The Base Model)                                                 │
│ ──────────────────────────────────────                                                 │
│ • Input: Raw continuous knowledge paragraphs (data/pretrain_corpus.txt).               │
│          (No questions, no answers, no chat format).                                   │
│ • Objective: Next-token prediction across raw human knowledge and facts.               │
│ • Output: base_model.pkl                                                               │
│                                                                                        │
│ ⚠️ Base Model Behavior:                                                                │
│    • Prompt: "Paris is the capital city of"                                            │
│      -> Completes: "france, located along the seine river..." ✅                        │
│    • Prompt: "User: Where did you stay in Paris? Assistant:"                           │
│      -> Rambles / fails because it was never trained on chat formatting! ❌             │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Post-Training / Supervised Fine-Tuning (The Chat Model)                       │
│ ────────────────────────────────────────────────────────────────                       │
│ • Input: Curated dialogue prompt-response pairs (data/sft_dialogues.jsonl).            │
│ • Objective: Adapt the knowledgeable Base Model to conversational instruction format.   │
│ • Output: chat_model.pkl                                                               │
│                                                                                        │
│ ✨ Chat Model Behavior:                                                                │
│    • Prompt: "User: Where did you stay in Paris?"                                      │
│      -> Assistant: "I stayed at a boutique hotel overlooking the Eiffel Tower." ✅      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 How to Run the Full Pipeline

Navigate to `baby_gpt/`:
```bash
cd baby_gpt
```

### Step 0: (Optional) Generate the Raw Corpus & SFT Datasets
```bash
uv run python create_datasets.py
```

### Step 1: Pre-train the Base Model (Stage 1)
Trains the model on raw knowledge paragraphs and saves `base_model.pkl`:
```bash
uv run python 1_pretrain.py
```

### Step 2: Inspect Base Model Behavior (Why Base Models Can't Chat)
See how the Base Model completes raw sentences vs why it fails on chat prompts:
```bash
uv run python 2_test_base.py
```

### Step 3: Fine-tune the Chat Model via SFT (Stage 2)
Takes `base_model.pkl` and fine-tunes it on conversational dialogues into `chat_model.pkl`:
```bash
uv run python 3_post_train.py
```

### Step 4: Live Interactive Chat with the Fine-Tuned Model
```bash
uv run python 4_chat.py
```

**Live Chat Output:**
```text
=================================================================
🤖 Baby-GPT Chat Model (Post-Trained / SFT)
Ask questions about Travel, Food, Health, Tech, or Finance!
Type 'quit', 'exit', or 'q' to exit
=================================================================

You: Where did you stay in Paris?
Baby-GPT: i stayed at a boutique hotel with a balcony overlooking the eiffel tower.

You: What is butter chicken served with?
Baby-GPT: butter chicken is traditionally served with warm garlic naan and steamed rice.

You: What is a milestone for bench press?
Baby-GPT: a bench press of 225 pounds is a classic strength milestone.

You: What is FastAPI used for?
Baby-GPT: fastapi is a python web framework used to build high - performance rest apis.

You: What is a Roth IRA?
Baby-GPT: a roth ira is an account where post - tax money grows and is withdrawn tax - free in retirement.

You: What makes Iceland unique?
Baby-GPT: iceland is famous for dramatic waterfalls, volcanic landscapes, and the green northern lights.
```

---

## 📁 Repository Layout

```text
baby_gpt/
├── data/
│   ├── pretrain_corpus.txt    # Stage 1: Raw knowledge paragraphs
│   └── sft_dialogues.jsonl    # Stage 2: Conversational instruction dialogues
├── create_datasets.py         # Dataset generator for both stages
├── tokenizer.py               # Tokenizer (<bos>, <eos>, <pad>, <unk>)
├── model.py                   # BabyGPT Causal Transformer (NumPy)
├── 1_pretrain.py              # Stage 1 Pre-training script -> base_model.pkl
├── 2_test_base.py             # Script demonstrating Base Model vs Chat Model
├── 3_post_train.py            # Stage 2 SFT Post-training script -> chat_model.pkl
├── 4_chat.py                  # Interactive Chat CLI for Chat Model
├── base_model.pkl             # Serialized Base Model weights
├── chat_model.pkl             # Serialized Chat Model weights
└── README.md                  # Detailed 2-stage documentation
```
