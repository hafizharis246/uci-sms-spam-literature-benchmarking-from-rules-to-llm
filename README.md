# SMS Spam Detection Systems (UCI Corpus Benchmark)

A comprehensive implementation, evaluation, and literature review of text classification paradigms—ranging from legacy rule-based systems to modern Large Language Models (LLMs)—strictly focused on the **UCI SMS Spam Collection dataset** (5,574 English messages).

---

## 📌 Repository Overview

This repository hosts both the academic literature and practical code implementations benchmarking the evolution of SMS spam filtering using the UCI dataset. The main objective is to provide a complete side-by-side comparison across five sequential technological paradigms:

1. **Rule-Based & Heuristics**: Regex patterns, custom fuzzy rules, and handcrafted feature sets.
2. **Traditional Machine Learning**: TF-IDF vectorization paired with statistical classifiers (Naïve Bayes, Support Vector Machines, Random Forest).
3. **Deep Learning Architectures**: Sequential hybrid neural networks (1D-CNN + LSTM).
4. **Transformers**: Contextual bidirectional self-attention models (BERT).
5. **Large Language Models (LLMs)**: Zero-shot, few-shot, and fine-tuned open-source and commercial LLMs (e.g., Mixtral 8x7B, LLaMA-2/3, Mistral, GPT-4).

The repository is organized into two primary root folders:
* **`Research Papers related to uci-sms-spams-data`**: Contains the foundational and modern research papers centered around the UCI SMS dataset and spam filtering methodologies.
* **`Codes`**: Contains scripts, data preprocessing pipelines, notebook benchmarks, and execution setups for each of the five model paradigms.

---

## ⚙️ Evolutionary Model Paradigms

### 1. Rule-Based Systems
Initial approaches evaluated explicit rules, structural heuristics (URL presence, capital letter counts, phone number regex), and Binary Particle Swarm Optimization (BPSO)-driven fuzzy rule generation. Although fast, these models lacked robustness against simple text obfuscation.

### 2. Traditional Machine Learning
Statistical models utilize word frequency matrices ($n$-grams, TF-IDF). Baseline benchmarks established by Almeida et al. showed that Linear Support Vector Machines (SVM) outperformed standard algorithms like Naïve Bayes and Decision Trees.

### 3. Deep Learning (1D-CNN + LSTM)
Combines 1D Convolutional Neural Networks to capture spatial $n$-gram features with Long Short-Term Memory (LSTM) layers to preserve the sequential context and long-range dependencies of text messages.

### 4. Transformers (BERT)
Fine-tunes transformer models (`bert-base-uncased`) to capture bidirectional contextual semantics and complex sentence structures, drastically reducing false positive rates.

### 5. Large Language Models (LLMs)
Evaluates state-of-the-art LLMs using zero-shot/few-shot prompting and parameter-efficient fine-tuning (LoRA). Fine-tuned open-source models like Mixtral-8x7B demonstrate high resilience against adversarial perturbations and concept drift.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* PyTorch / TensorFlow
* CUDA-compatible GPU (required for training CNN-LSTM, BERT, and fine-tuning LLMs)

### Setup & Installation

```bash
# Clone the repository
git clone [https://github.com/hafizharis246/uci-sms-spam-literature-benchmarking-from-rules-to-llm.git](https://github.com/hafizharis246/uci-sms-spam-literature-benchmarking-from-rules-to-llm)
