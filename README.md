# Chatbot Based on Deep Neural Networks

**Author**: Vladyslav Malashchuk  
**Faculty**: Brno University of Technology (VUT), Faculty of Information Technology (FIT)  
**Thesis Title**: Chatbot Based on Deep Neural Networks  

---

## 📝 Project Description  
This bachelor's thesis introduces chatbot development, explaining how different architectures work and how to create your own chatbot using **PyTorch**. The project covers both **retrieval-based models** (simpler, rule-based chatbots) and **transformer-based models** (such as GPT and LLaMA).

By following this work, users will learn how to implement a chatbot using **PyTorch**, train models on different datasets, and experiment with both basic and advanced architectures.  

### Key Features:  
- Implementation of **retrieval-based** and **transformer-based** chatbots using PyTorch.  
- Customizable training pipelines for fine-tuning transformer models.  
- Support for training on different datasets.  
- Pre-provided datasets for both **open-domain** conversations and a **medical chatbot**.  

This project is a great starting point for anyone looking to understand chatbot development and build their own AI-powered assistant using **PyTorch**. 🚀

My thesis: [Chatbot Based on Deep Neural Networks - VUT](https://www.vut.cz/en/students/final-thesis/detail/164616)

---

## 🛠️ Technologies & Libraries  

- **Frameworks**: PyTorch, Hugging Face Transformers  
- **Preprocessing**: Tokenization, dataset augmentation, context formatting, stemming, bag-of-words processing  
- **Models**: GPT-2, LLaMA, Transformer, NeuralNet (implementations included)  
- **Evaluation**: BLEU, ROUGE, and custom conversational metrics  
- **Tools**: Git, JSON configuration, modular code structure, dataset processing  

### 🔧 Included Python Libraries  
- **Data Handling**: `json`, `pandas`, `datasets`, `random`  
- **Deep Learning**: `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`  
- **Transformers**: `AutoTokenizer`, `AutoModelForCausalLM`, `transformers.pipeline`  
- **Text Processing**: `nltk`, `nltk.stem.porter`, `re`  
- **Evaluation Metrics**: `nltk.translate.bleu_score`, `rouge_score.rouge_scorer`, `metrics.calculate_bleu`, `metrics.calculate_rouge`  
- **Modeling**: `model.Transformer`, `model.NeuralNet`  
- **Utilities**: `matplotlib.pyplot`, `time`, `math`, `sklearn.model_selection.train_test_split`  
- **Dataset Processing**: `dataset.process_tab_separated_file`, `dataset.ChatDataset`  

---

## ⚙️ Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/Vlad6422/BP_Transformer_PyTorch.git
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

---

## 📜 License  
This project is licensed under the terms of the [MIT License](LICENSE).  

---

## 🙏 Acknowledgments  
- Special thanks to Ing. Martin Kostelník for his invaluable guidance and support throughout my work.

---
## REFERENCES
1. Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. Advances in Neural Information Processing Systems, 33, 1877–1901.

2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 4171–4186). Minneapolis, MN: Association for Computational Linguistics.

3. Hugging Face, Inc. (2020). Transformers: State-of-the-art natural language processing [Computer software]. Retrieved from https://huggingface.co

4. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. Proceedings of the 3rd International Conference on Learning Representations. San Diego, CA: ICLR.

5. Pal, A., & Condon, P. (2021, March 15). Designing a conversational AI chatbot. Towards Data Science. Retrieved from https://towardsdatascience.com/designing-a-conversational-ai-chatbot

6. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training [Technical report]. OpenAI. Retrieved from https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

7. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30, 5998–6008.
---