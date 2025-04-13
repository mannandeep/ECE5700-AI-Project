# ECE5700-AI-Project

# Hybrid PEFT Fine-Tuning for GPT-2 (LoRA + Prefix-Tuning)

This project demonstrates a hybrid parameter-efficient fine-tuning (PEFT) method that combines Low-Rank Adaptation (LoRA) and Prefix-Tuning on GPT-2 Medium. The approach is designed to adapt large language models for assistant-style dialogue while requiring less than 1% of the model parameters to be updated.

The model is trained on OpenAssistant conversations and evaluated using perplexity and BLEU scores. The project includes both quantitative evaluation and an interactive chatbot interface for qualitative analysis.

---

## Getting Started

You can either download and run the notebook locally, or open it directly in Google Colab.

### Option 1: Run Locally

1. Download the notebook file:  
   [`PEFT_GPT2_Hybrid.ipynb`](https://github.com/mannandeep/ECE5700-AI-Project/blob/main/PEFT_GPT2_Hybrid.ipynb)

2. Install the required packages:
   ```bash
   pip install transformers datasets evaluate tqdm
