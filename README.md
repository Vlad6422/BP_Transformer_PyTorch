# Chatbot Based on Deep Neural Networks

**Author**: Vladyslav Malashchuk  
**Faculty**: Brno University of Technology (VUT), Faculty of Information Technology (FIT)  
**Thesis Title**: Chatbot Based on Deep Neural Networks  

# Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [LICENSE](#license)
- [REFERENCES](#references)
## Introduction 

This is an **Introduction** to my Bachelor's project. My task in this work was to write a chatbot based on deep neural networks. Open sources, as well as articles, documentation for the frameworks used, etc. were used in writing the work. My goal was to write a chat bot, which resulted in writing several different ones at once on different architectures, with varying degrees of complexity and productivity. As a result, it was decided to stay with 2 types, the simplest and fastest, and at the moment of writing this work, the newest and most effective chat bot based on the **Transformer Architecture** [[7]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).


A detailed description of the theory, discussions of these different architectures, metrics, testing with people can be found in my text work, it describes everything that is needed even for a person who has never worked with Machine Learning or AI. In my thesis I describe in detail from the theory of machine learning to creating chatbots, how they work and testing them. You can find the text and my bachelor's thesis at this link [[8]](https://www.vut.cz/en/students/final-thesis/detail/164616)

This document will not discuss or explain the theory, but will strictly describe the tools used in writing the project and their application in it.

I will start with a short list where I will describe the language used, framework, tokenizer, additional libraries used in the project. Do not expect to fully understand the work of the project after reading this small file, you will understand the technical side, in order to understand what chatbots are, their essence and how they work "in words" without technical reasoning, I advise you to read my thesis, the link is above.


## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/Vlad6422/BP_Transformer_PyTorch.git
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
Everyone has different devices, so it is impossible to predict everything exactly, but the project does not use non-standard operating systems or programming languages, etc. Therefore, the project must work on Windows, it was tested on Windows 10 and 11, at least Python must be installed, optionally CUDA [[9]](https://developer.nvidia.com/cuda-toolkit) from Nvidia to speed up training. The remaining libraries are installed either with the help of the script described above. Or manually if necessary. Creating a virtual environment for such tasks is unnecessary and an additional layer greatly slows down training, and this was not part of my assignment.

All training and launch were carried out on my personal laptop with Windows 11 operating system and RTX 3050 ti.


## Language and Framework

The project was written in Python 3.11.9 [[10]](python.org/downloads/release/python-3119/) and PyTorch 2.4.1+cu124 [[11]](https://pytorch.org/). These are the main tools used in writing models and training them. cu124 means support for CUDA 12.4 [[9]](https://developer.nvidia.com/cuda-toolkit).

Python was chosen because of the many ready-made libraries that speed up development, the PyTorch framework was also chosen for Python, since it is one of the most popular frameworks for Machine Learning at the moment, and taking into account these 2 factors, Python and PyTorch cannot be without each other, so they were chosen as the main language and framework. CUDA was chosen because Nvidia at the time of 2025 is almost a monopoly in the field of machine learning and the only one who provides video cards and software that accelerates any training, at the moment there is only one alternative available to the average user, these are AMD video cards and ROCm [[12]](https://www.amd.com/en/products/software/rocm.html) software, but it is only supported on Linux, recently WSL support was added, but even so, at the moment ROCm is very behind CUDA in training speed, as soon as the software is improved, it will be a good reason to try to run training on an AMD video card.

## Libraries  
- **Data Handling**: `json`, `pandas`, `datasets`, `random`  
- **Text Processing**: `nltk`, `nltk.stem.porter`, `re`    
- **Utilities**: `matplotlib.pyplot`, `time`, `math`, `sklearn.model_selection.train_test_split`   

## Metrics




## License  
This project is licensed under the terms of the [MIT License](LICENSE).  



## Acknowledgments  
Special thanks to **Ing. Martin Kostelník** for his invaluable guidance and support throughout my work. Also for advice and help with theory and help with studying machine learning from scratch. Throughout the writing of the work we were in contact and you often helped me in solving my problems.


## REFERENCES

[1] BROWN, T. B. et al. Language models are few-shot learners [online]. In: Advances in Neural Information Processing Systems, 2020, vol. 33, pp. 1877–1901. [accessed 2025-04-17]. Available at: https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf

[2] DEVLIN, J., CHANG, M.-W., LEE, K., and TOUTANOVA, K. BERT: Pre-training of deep bidirectional transformers for language understanding [online]. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Minneapolis, MN: Association for Computational Linguistics, 2019, pp. 4171–4186. [accessed 2025-04-17]. Available at: https://aclanthology.org/N19-1423/

[3] HUGGING FACE, Inc. Transformers: State-of-the-art natural language processing [online]. 2020. [accessed 2025-04-17]. Available at: https://huggingface.co

[4] KINGMA, D. P., and BA, J. Adam: A method for stochastic optimization [online]. In: Proceedings of the 3rd International Conference on Learning Representations, San Diego, CA: ICLR, 2014. [accessed 2025-04-17]. Available at: https://arxiv.org/abs/1412.6980

[5] PAL, A., and CONDON, P. Designing a conversational AI chatbot [online]. Towards Data Science, 2021-03-15. [accessed 2025-04-17]. Available at: https://towardsdatascience.com/designing-a-conversational-ai-chatbot

[6] RADFORD, A., NARASIMHAN, K., SALIMANS, T., and SUTSKEVER, I. Improving language understanding by generative pre-training [online]. OpenAI, 2018. [accessed 2025-04-17]. Available at: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

[7] VASWANI, A. et al. Attention is all you need [online]. In: Advances in Neural Information Processing Systems, 2017, vol. 30, pp. 5998–6008. [accessed 2025-04-17]. Available at: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

[8] MALASHCHUK, Vladyslav. Chatbot based on deep neural networks. Online, bachelor's Thesis. Martin KOSTELNÍK (supervisor). Brno: Brno University of Technology, Faculty of Information Technology, 2025. Available at: https://www.vut.cz/en/students/final-thesis/detail/164616. [accessed 2025-04-18].

[9] NVIDIA Corporation. CUDA Toolkit [online]. [accessed 2025-04-18]. Available at: https://developer.nvidia.com/cuda-toolkit

[10] PYTHON SOFTWARE FOUNDATION. Python 3.11.9 Release [online]. [accessed 2025-04-18]. Available at: https://www.python.org/downloads/release/python-3119/

[11] PYTORCH. PyTorch: An open source machine learning framework [online]. [accessed 2025-04-18]. Available at: https://pytorch.org/

[12] AMD. ROCm™: Open software platform for accelerated computing [online]. [accessed 2025-04-18]. Available at: https://www.amd.com/en/products/software/rocm.html

[13] PAPINENI, Kishore, ROUKOS, Salim, WARD, Todd, and ZHU, Wei-Jing. BLEU [online]. In: Proceedings of the 40th Annual Meeting on Association for Computational Linguistics - ACL '02, Morristown, NJ, USA: Association for Computational Linguistics, 2002, p. 311. [cit. 2025-04-18]. Available at: https://doi.org/10.3115/1073083.1073135

[14] LIN, Chin-Yew. ROUGE: A package for automatic evaluation of summaries [online]. In: Text Summarization Branches Out: Proceedings of the ACL-04 Workshop, Barcelona, Spain, 2004, pp. 74–81. [cit. 2025-04-18]. Available at: https://aclanthology.org/W04-1013.pdf