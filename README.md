# Sentiment Analysis with LSTM and BERT

This project compares two NLP approaches—an LSTM-based Recurrent Neural Network and a fine-tuned BERT transformer—for classifying the sentiment of movie reviews.

## 📌 Overview

The goal is to analyze raw movie review text and determine if the sentiment is positive or negative. We implemented:
- A custom LSTM-based RNN from scratch to learn word sequences and temporal patterns.
- A BERT-based classifier using transfer learning to capture contextual embeddings from pre-trained language models.

## 🧠 Key Features

- **Text Preprocessing:** Tokenization, padding, and batching for sequential data.
- **LSTM Model:** Trained on word sequences to capture dependencies in textual sentiment.
- **BERT Model:** Fine-tuned Hugging Face BERT for contextual understanding.
- **Performance Comparison:** Evaluated both models on accuracy and generalization.

## 📊 Results

| Model | Accuracy |
|-------|----------|
| LSTM  | 81%      |
| BERT  | 89%      |

## 📁 Project Structure

