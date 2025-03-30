# Chatbot Based on Deep Neural Networks

**Author**: Vladyslav Malashchuk  
**Faculty**: Brno University of Technology (VUT), Faculty of Information Technology (FIT)  
**Thesis Title**: Chatbot Based on Deep Neural Networks  

---



## 📝 Project Description  
This bachelor's thesis introduces chatbot development, explaining how different architectures work and how to create your own chatbot using **PyTorch**. The project covers both **retrieval-based models** (simpler, rule-based chatbots) and **transformer-based models** (such as GPT-2 and LLaMA).

By following this work, users will learn how to implement a chatbot using **PyTorch**, train models on different datasets, and experiment with both basic and advanced architectures.  

### Key Features:  
- Implementation of **retrieval-based** and **transformer-based** chatbots using PyTorch.  
- Customizable training pipelines for fine-tuning transformer models.  
- Support for training on different datasets.  
- Pre-provided datasets for both **open-domain** conversations and a **medical chatbot**.  

This project is a great starting point for anyone looking to understand chatbot development and build their own AI-powered assistant using **PyTorch**. 🚀

My thesis : https://www.vut.cz/en/students/final-thesis/detail/164616

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

## 🚀 Usage  
1. **Training**:  
   Configure hyperparameters in `config.json` and run:  
   ```bash
   python train.py --config config.json
   ```
2. **Interactive Chat**:  
   Start the chatbot with a pre-trained model:  
   ```bash
   python chat.py --model gpt2 --temperature 0.7
   ```
   *(Adjust `--model` and `--temperature` as needed.)*  

3. **Evaluation**:  
   Compute metrics using:  
   ```bash
   python metrics.py --dataset test_data.jsonl
   ```

---

## 📜 License  
This project is licensed under the terms of the [MIT License](LICENSE).  

---

## 🙏 Acknowledgments  
- Supervisors and faculty members at VUT FIT.  
- Open-source communities for libraries like Hugging Face Transformers.  
- Dataset providers (e.g., Cornell Movie-Dialogs Corpus, Persona-Chat).  

