# ECE 57000 - Artificial Intelligence | Hybrid PEFT Fine-Tuning for GPT-2 (LoRA + Prefix-Tuning)

This project demonstrates a hybrid parameter-efficient fine-tuning (PEFT) approach for adapting GPT-2 Medium to an assistant-style dialogue generation task. It combines two efficient fine-tuning methods—LoRA and Prefix-Tuning—to achieve strong adaptation with under 1% of model parameters updated.

The hybrid model is trained on conversations from the OpenAssistant dataset and evaluated both qualitatively (via chat interface) and quantitatively (via perplexity and BLEU score). All components—LoRA, Prefix-Tuning, and their integration—are implemented manually without high-level wrappers to maximize transparency and flexibility.

---

## Features

- Custom implementation of LoRA and Prefix-Tuning
- Combined GPT-2 hybrid architecture for PEFT
- Fine-tuned on the OpenAssistant dataset
- Interactive chat interface (before and after training)
- Quantitative evaluation using perplexity and BLEU

---

## How to Use

You can either download and run the notebook locally, or upload it to Google Colab.

### Option 1: Run Locally

1. Download the notebook:  
   `PEFT_GPT2_Hybrid.ipynb` from this repository.

2. Install required packages:
   ```
   pip install transformers datasets evaluate tqdm
   ```

3. Run the notebook:
   ```
   jupyter notebook
   ```

Note: A CUDA-compatible GPU is recommended for training.

---

### Option 2: Run in Google Colab

1. Open [https://colab.research.google.com](https://colab.research.google.com)
2. Upload the `.ipynb` notebook file
3. Go to `Runtime > Change runtime type`
   - Select GPU as the hardware accelerator
   - (Optional) Enable High-RAM runtime
4. Run all cells sequentially

This project was developed and tested on Google Colab Pro with GPU acceleration (T4 or A100).

---

## Evaluation Results

| Metric      | Value     |
|-------------|-----------|
| Perplexity  | 3.39      |
| BLEU Score  | 0.0196    |

Evaluation was conducted on 100 user-assistant interactions from the OpenAssistant dataset. Despite limited training time, the hybrid model demonstrates improved coherence and fluency over the frozen GPT-2 baseline.

---

## References and Source Code

This implementation is based on the following foundational research papers:

1. **Prefix-Tuning: Optimizing Continuous Prompts for Generation**  
   - Xiang Lisa Li, Percy Liang (ACL 2021)  
   - [Paper](https://arxiv.org/abs/2101.00190)  
   - Code (archived): [https://github.com/xiaoxuan520/prefix-tuning](https://github.com/xiaoxuan520/prefix-tuning)  
   > *Note: This code repository may no longer be actively maintained or public.*

2. **LoRA: Low-Rank Adaptation of Large Language Models**  
   - Edward J. Hu et al. (ICLR 2022)  
   - [Paper](https://arxiv.org/abs/2106.09685) | [Code](https://github.com/microsoft/LoRA)

While inspired by these works, this project reimplements both methods independently to support full control and experimentation.

---

## Acknowledgements

This project was completed as part of a graduate-level Artificial Intelligence course (ECE 57000, Spring 2025). It leverages the Hugging Face Transformers, Datasets, and Evaluate libraries to support model training and evaluation.

Some text descriptions, explanations, and documentation were refined for clarity and formatting with the assistance of a large language model (LLM). The author conducted all conceptual work, implementation, and analysis independently.
