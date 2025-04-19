# 🎬 Sentiment Analysis with LSTM and BERT

This project explores sentiment classification of movie reviews by comparing a custom-built LSTM-based Recurrent Neural Network (RNN) and a fine-tuned BERT model. The objective is to determine which architecture better captures the nuances in textual sentiment using a real-world dataset.

---

## 📚 Dataset

- **Source:** IMDB Movie Review dataset (binary classification: positive/negative)
- **Preprocessing:**
  - Lowercasing text
  - Removing punctuation and stopwords
  - Tokenization using TorchText / Hugging Face Tokenizer
  - Word padding/truncation to a fixed sequence length

---

## 🧠 Model Architectures

### 1. **LSTM-based RNN (Built from Scratch)**

- **Embedding Layer:** Converts word indices into dense vectors.
- **LSTM Layer:** Processes the word embeddings to capture temporal dependencies.
- **Fully Connected Layer:** Maps the final hidden state to a binary output.
- **Activation:** Sigmoid for binary classification.

**Key Hyperparameters:**
- Embedding size: 128
- Hidden size: 256
- Number of layers: 2
- Dropout: 0.5
- Optimizer: Adam
- Loss: Binary Cross Entropy Loss

### 2. **Pretrained BERT Classifier**

- **Base Model:** `bert-base-uncased` from Hugging Face Transformers
- **Fine-Tuning:** Trained the entire BERT model with a classification head on top.
- **Tokenizer:** BERT's WordPiece tokenizer

**Key Hyperparameters:**
- Learning rate: 2e-5
- Epochs: 3
- Max sequence length: 256
- Batch size: 16
- Optimizer: AdamW
- Scheduler: Linear warmup & decay

---

## ⚙️ Training Pipeline

1. **Dataloader Construction:**
   - Created custom PyTorch `Dataset` and `DataLoader` classes for both models.
   - Incorporated padding, attention masks (for BERT), and batched iteration.

2. **Training Loop:**
   - Forward pass
   - Backward pass with gradient clipping (for LSTM)
   - Validation after each epoch
   - Model checkpointing based on validation performance

3. **Evaluation Metrics:**
   - Accuracy
   - Precision / Recall / F1-score
   - Confusion Matrix
   - ROC-AUC (optional)

---

## 📊 Results Summary

| Model         | Accuracy | F1-Score | Notes                           |
|---------------|----------|----------|---------------------------------|
| LSTM-RNN      | 81%      | 0.80     | Tended to miss subtle negations |
| BERT-Base     | 89%      | 0.88     | Superior in context awareness   |

---


## 🔍 Key Learnings

- LSTMs are effective for modeling word sequences but struggle with long-range dependencies.
- BERT significantly improves performance by capturing bidirectional context and transfer learning.
- Data preprocessing and proper tokenization are critical for NLP pipelines.
- Autograding compatibility requires modular saving/loading of models with consistent interfaces.

---
