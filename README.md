# 🧠 Advanced Deep Learning Journey

Welcome to my Advanced Deep Learning journey! This repository tracks my learning, practice, and projects as I worked through the Advanced Deep Learning course at [Phitron](https://phitron.io). It collects the course notes, the instructor's reference notebooks, my own re-implementations, practice-problem solutions, and the mid-term and final exams.

The material runs from image classification with CNNs, through sequence modelling with simple RNNs and LSTMs, to classic NLP text preprocessing and representation. All 16 modules, including the final exam, are present, so the course content in this repository is complete.

Everything is implemented in **Python with PyTorch**, and the notebooks were run in **Google Colab** (T4 GPU for the CNN work).

---

## Overview

The repository is organized by week, and each week is split into numbered modules that mirror the course syllabus:

- **Week 01 – CNN Fundamentals** (Modules 01–04): what an image is, convolution, padding/stride/pooling, a multi-class CNN project on PlantVillage, two brain-tumor MRI classifiers from the conceptual sessions, and the mid-term exam.
- **Week 02 – Simple RNN** (Modules 05–07): the anatomy of an `nn.RNN`, forward/backward propagation through time, and a small question-answering model trained on a GK dataset.
- **Week 03 – LSTM** (Modules 08–11): why plain RNNs fail on long sequences, the three LSTM gates, and a two-part next-word-prediction (language modelling) project.
- **Week 04 – NLP** (Modules 12–16): regex, the NLP pipeline, text cleaning on IMDB reviews, Bag of Words / n-grams / TF-IDF, pre-trained Word2Vec, and the final exam.

Inside most modules you will find three kinds of files:

| File type | What it is |
|---|---|
| `*.pdf` | Course notes exported from OneNote (several are hand-drawn, image-only pages) plus practice-problem sheets |
| `Phitron_*.ipynb` | The instructor's reference notebooks as shared in class |
| Unprefixed `*.ipynb` / `*.md` | My own worked versions of the same material, my practice-problem solutions, and my exam answers |

---

## Learning Roadmap

```
Week 01  CNN Fundamentals
   ├─ Images as arrays → convolution, filters, feature maps
   ├─ Padding, stride, pooling, output-size formula
   ├─ Custom Dataset / DataLoader, CNN training loop in PyTorch
   └─ Mid-term exam: theory + age-group CNN on face images
          │
Week 02  Simple RNN
   ├─ Tokenization, vocabulary, text → indices, nn.Embedding
   ├─ nn.RNN anatomy and parameter counting
   ├─ Forward propagation & BPTT, vanishing / exploding gradients
   └─ GK question-answering with a single-layer RNN
          │
Week 03  LSTM
   ├─ Limits of simple RNNs, cell state vs. hidden state
   ├─ Forget / input / output gates and their formulas
   └─ Next-word prediction (language model) on a short story
          │
Week 04  NLP
   ├─ Regex and the end-to-end NLP pipeline
   ├─ Text cleaning: HTML, URLs, punctuation, stopwords, stemming, lemmatization
   ├─ One-hot, Bag of Words, n-grams, TF-IDF
   ├─ Word2Vec (CBOW / Skip-gram) with pre-trained Google News vectors
   └─ Final exam: 20 fill-in-the-blank coding tasks across all four weeks
```

---

## Repository Structure

```
Advanced Deep Learning/
├── README.md
├── Week 01 CNN Fundamentals/
│   ├── Conceptual Sessions/
│   │   ├── Week 01 CS-01 Image Classification/        cnn_project.ipynb (PlantVillage)
│   │   └── Week 01 CS-02 Complete CNN Project/        Brain_Tumor_CNN.ipynb, Brain_Tumor_CNN_Beginner_100_Per_Class.ipynb
│   ├── Module 01 Introduction to CNN/                 notes (PDF), image.ipynb, flower.jpeg
│   ├── Module 02 Padding, Stride and Pooling in CNNs/ notes (PDF), blog & animation link lists
│   ├── Module 03 CNN Project/                         PlantVillage.ipynb, cnn_project.ipynb, notes (PDF)
│   └── Module 04 ADL Mid Term Exam/                   question + solution notebooks
├── Week 02 Simple RNN/
│   ├── Conceptual Sessions/
│   │   └── Week 02 CS-01 Basic Anatomy of a Recurrent Neural Network (RNN)/  RNN.ipynb
│   ├── Module 05 Simple RNN Basics of NLP/            notes (PDF), RNN_using_pytorch.ipynb
│   ├── Module 06 Forward propagation and backward propgation of RNN/
│   │   └── Practice Problem/                          questions + solution (PDF)
│   └── Module 07 Simple GK Question and Answering System using Simple RNN/
│       ├── simple_gk_answring_system_using_rnn.ipynb, gk_qna_dataset.csv, notes (PDF)
│       └── Practice Problem/                          RNN_Module_3_Solutions.md
├── Week 03 LSTM/
│   ├── Module 08 LSTM Basics/                         notes (PDF), Practice Problem/Solution.md
│   ├── Module 09 LSTM Gates/                          notes (PDF), Practice Problem/Solution.md
│   ├── Module 10 LSTM Project (Part 1)/               LSTM_Project_Part_01.ipynb, notes (PDF)
│   └── Module 11 LSTM Project (Part 2)/               LSTM_Project.ipynb, Practice Problem/ (questions only)
└── Week 04 NLP/
    ├── Module 12 Text Preprocessing in NLP with Regex/ Module__regex.ipynb, notes (PDF), Practice Problem/
    ├── Module 13 Text Preprocessing Fundamentals/     Text_preprocessing.ipynb, Practice Problem/
    ├── Module 14 Text representation in NLP/          tfidf.ipynb, Phitron_bag_of_words.ipynb, notes, Practice Problem/
    ├── Module 15 Word2Vec/                            word2vec.ipynb, notes (PDF), Practice Problem/
    └── Module 16 ADL Final Exam/                      question + solution notebooks
```

Each module that has a `Phitron_*.ipynb` also has my own unprefixed copy next to it; the tree above lists only the unprefixed versions to keep it readable.

---

## Topics Covered

| Area | Theory (notes, practice answers, exams) | Implemented in code |
|---|---|---|
| **Images & CNN basics** | Pixels/channels, RGB shape `(H, W, 3)`, why CNNs beat dense nets on images, local connectivity & weight sharing, filters and feature maps, hierarchical features | Loading an image with PIL, converting to/from a NumPy array; `nn.Conv2d` / `nn.BatchNorm2d` / `nn.MaxPool2d` / `nn.AdaptiveAvgPool2d` stacks |
| **Padding, stride, pooling** | Output-size formula `(N + 2P − F) / S + 1`, valid vs. same padding, what pooling discards and why that is acceptable, max vs. average pooling | Layer-by-layer shape calculations (mid-term Q8, final exam Q16–Q20), NumPy `np.pad`, manual stride-2 convolution, `reshape`-based max/avg pooling |
| **CNN training pipeline** | Train/validation/test splits, augmentation, dropout, class imbalance (discussed, not implemented) | Custom `Dataset` over class folders, `ImageFolder`, `transforms` (resize, normalize, flips, rotation, affine, colour jitter), `CrossEntropyLoss` + Adam, `ReduceLROnPlateau`, best-checkpoint saving, confusion matrix, classification report, single-image inference with softmax confidence |
| **Simple RNN** | Sequential processing and hidden state, shared weights across time steps, one-hot encoding of words, forward propagation, BPTT, vanishing/exploding gradients, gradient clipping (conceptual) | `nn.RNN` with parameter-count breakdown, tokenizer → vocabulary → `<UNK>` handling → `nn.Embedding`, single-token QA model, NumPy RNN step and tanh/BPTT gradients (final exam) |
| **LSTM** | Why simple RNNs forget, cell state vs. hidden state, forget/input/output gate formulas, candidate memory, full cell workflow, applications | `nn.LSTM` language model with pre-padded n-gram sequences, autoregressive text generation, NumPy gate computations, LSTM parameter formula `4·(h·(h+x) + 2h)`, tiny sentiment LSTM with `BCELoss` |
| **Text preprocessing** | The NLP pipeline (acquisition → preparation → cleanup → preprocessing → feature engineering → modelling → evaluation → deployment → monitoring), regex syntax, corpus/tokens/stopwords/stemming vs. lemmatization | `re` (`findall`, `search`, `split`, `sub`, `compile`), HTML/URL/emoji/punctuation removal, chat-word expansion, TextBlob spell-check, NLTK & spaCy tokenization, Porter stemmer, WordNet lemmatizer |
| **Text representation** | One-hot encoding, Bag of Words, n-grams and vocabulary explosion, TF-IDF intuition, embeddings vs. sparse vectors | `CountVectorizer` (binary, unigram/bigram/trigram), `TfidfVectorizer`, IDF inspection, OOV behaviour, cosine similarity in NumPy |
| **Word2Vec** | CBOW vs. Skip-gram, window size, why dense vectors beat one-hot (dimensionality, similarity, memory) | Loading the 300-d Google News vectors with gensim, `most_similar`, `similarity`, `doesnt_match`, vector arithmetic (`king − man + woman`), training a tiny Skip-gram model |

Not covered in this repository: transformers/attention, transfer learning with pre-trained CNNs, GRUs, sequence-to-sequence decoders, and deployment. Some of these are mentioned in the notes as "what comes next", but there is no implementation here.

---

## Technologies & Tools

| Category | Confirmed from the notebooks |
|---|---|
| Language | Python 3 |
| Deep learning | PyTorch (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data`), torchvision (`datasets`, `transforms`) |
| Data handling | NumPy, pandas, `collections.Counter`, Pillow (PIL) |
| Machine learning / metrics | scikit-learn (`train_test_split`, `classification_report`, `confusion_matrix`, `CountVectorizer`, `TfidfVectorizer`) |
| NLP libraries | `re`, NLTK (`word_tokenize`, `stopwords`, `PorterStemmer`, `WordNetLemmatizer`), spaCy (`en_core_web_sm`), TextBlob, gensim (`KeyedVectors`, `Word2Vec`, `gensim.downloader`) |
| Visualization | Matplotlib (training curves, confusion matrices, sample images) |
| Data access | `kagglehub` dataset downloads, `git clone` of a GitHub dataset repo, Google Drive mount |
| Environment | Google Colab notebooks (T4 GPU runtime for the CNN notebooks; the RNN/LSTM/NLP notebooks do not request a GPU accelerator) |

There is no `requirements.txt` or environment file in the repository; the list above is derived from the actual imports and `!pip install` cells.

---

## Learning Modules

### Week 01 – CNN Fundamentals

**Module 01 – Introduction to CNN.** Notes on what an image is (pixel grids, grayscale vs. RGB, `(224, 224, 3)`), CNN intuition (edges → textures → parts → objects) and a survey of CNN applications. [`image.ipynb`](Week%2001%20CNN%20Fundamentals/Module%2001%20Introduction%20to%20CNN/image.ipynb) opens the included `flower.jpeg` with PIL, converts it to a NumPy array, prints the raw pixel values and converts it back. The `cnn architectures.pdf` file is a diagram-only export with no extractable text.

**Module 02 – Padding, Stride and Pooling.** Pure theory: why unpadded convolution shrinks feature maps and under-uses border pixels, how stride down-samples, the output-size formula, and what pooling loses versus what it buys (translation invariance, fewer parameters). Two `.docx` files collect the blog posts and animations used in class.

**Module 03 – CNN Project (PlantVillage).** [`PlantVillage.ipynb`](Week%2001%20CNN%20Fundamentals/Module%2003%20CNN%20Project/PlantVillage.ipynb) and the near-identical [`cnn_project.ipynb`](Week%2001%20CNN%20Fundamentals/Module%2003%20CNN%20Project/cnn_project.ipynb) (also the CS-01 notebook) build the full pipeline from scratch: `kagglehub` download of PlantVillage (38 classes, 43,444 train / 10,861 val images), a custom `Dataset` that walks class folders into `(path, label)` pairs, resize-to-128 + normalize transforms, and `MyCNN` – three `Conv(3×3, same) → ReLU → BatchNorm → MaxPool` blocks (32/64/128 filters) followed by `Linear(32768→128→64→38)` with dropout. Training uses `CrossEntropyLoss` + Adam (lr 1e-3) for 10 epochs. **Important caveat:** the notebooks deliberately train and evaluate on a `Subset` of the first 100 images (a quick smoke test), so the recorded "Test Accuracy: 1.0" is on that 100-image slice and is not a meaningful benchmark. `Module_27.pdf` explains the custom dataset class line by line.

**Conceptual Session CS-02 – Complete CNN Project (Brain Tumor MRI).** Two end-to-end notebooks on the Kaggle Brain Tumor MRI dataset (glioma / meningioma / notumor / pituitary):

- [`Brain_Tumor_CNN.ipynb`](Week%2001%20CNN%20Fundamentals/Conceptual%20Sessions/Week%2001%20CS-02%20Complete%20CNN%20Project/Brain_Tumor_CNN.ipynb) – full dataset (4,000 train / 1,600 val carved out of the Kaggle training split / 1,600 test). `BrainTumorCNN` has four `Conv → BN → ReLU → MaxPool` blocks (32→256), `AdaptiveAvgPool2d`, dropout 0.5 and a single `Linear(256, 4)` – only 390,404 parameters. Trained 30 epochs with augmentation, Adam, `ReduceLROnPlateau`, saving the best checkpoint by validation accuracy. Recorded results: **best validation accuracy 93.62%**, **test accuracy 87.75%** (macro F1 0.873). The confusion matrix shows glioma recall (0.67) as the weak spot, with glioma often predicted as meningioma.
- [`Brain_Tumor_CNN_Beginner_100_Per_Class.ipynb`](Week%2001%20CNN%20Fundamentals/Conceptual%20Sessions/Week%2001%20CS-02%20Complete%20CNN%20Project/Brain_Tumor_CNN_Beginner_100_Per_Class.ipynb) – the "beginner" version reads the dataset from a mounted Google Drive, samples 100 images per class and splits them 70/15/15. `SimpleCNN` (three conv blocks, 2.1 M parameters because of the large flatten layer) trained 15 epochs reaches **test accuracy 78.33%** on 60 images; the validation curve fluctuates, which the notebook uses to discuss small-data behaviour.

**Module 04 – ADL Mid Term Exam.** See [Evaluations / Exams](#evaluations--exams).

### Week 02 – Simple RNN

**Conceptual Session CS-01 – Basic Anatomy of an RNN.** [`RNN.ipynb`](Week%2002%20Simple%20RNN/Conceptual%20Sessions/Week%2002%20CS-01%20Basic%20Anatomy%20of%20a%20Recurrent%20Neural%20Network%20%28RNN%29/RNN.ipynb) walks through the whole text-to-model path on the Kaggle "100 Unique QA" dataset (90 rows): a hand-written tokenizer, vocabulary building with an `UNK` token (326 entries), a `QADataset`, and `SimpleRNN` = `Embedding(vocab, 50) → nn.RNN(50, 64) → Linear(64, vocab)`. Trained 20 epochs with Adam (lr 1e-3), the summed per-epoch loss falls from 524.9 to 11.6. A `predict` helper applies a 0.5 softmax threshold and answers "I do not know" for out-of-distribution questions.

**Module 05 – Simple RNN Basics of NLP.** Notes on sequential data, one-hot encoding of a toy three-sentence corpus, and how an RNN carries a hidden state across time steps. [`RNN_using_pytorch.ipynb`](Week%2002%20Simple%20RNN/Module%2005%20Simple%20RNN%20Basics%20of%20NLP/RNN_using_pytorch.ipynb) builds a minimal `nn.RNN(input_size=5, hidden_size=3) → Linear(3,1) → Sigmoid` and prints the parameter breakdown (`weight_ih` 15, `weight_hh` 9, two biases of 3, FC 4 → 34 total), which is a nice sanity check of the RNN parameter formula.

**Module 06 – Forward and backward propagation of RNN.** The three note PDFs here are hand-drawn, image-only pages (forward pass, BPTT). The practice problem and its [written solution](Week%2002%20Simple%20RNN/Module%2006%20Forward%20propagation%20and%20backward%20propgation%20of%20RNN/Practice%20Problem/Practice%20Problem%20Solution.pdf) cover variable-length sequences, shared parameters when unrolling, exploding gradients (with gradient clipping), vanishing gradients, and why inputs are fed oldest-to-newest. Note: the three note PDF filenames contain a non-breaking space, so they are linked only at the [folder level](Week%2002%20Simple%20RNN/Module%2006%20Forward%20propagation%20and%20backward%20propgation%20of%20RNN/).

**Module 07 – Simple GK Question and Answering System.** [`simple_gk_answring_system_using_rnn.ipynb`](Week%2002%20Simple%20RNN/Module%2007%20Simple%20GK%20Question%20and%20Answering%20System%20using%20Simple%20RNN/simple_gk_answring_system_using_rnn.ipynb) trains on [`gk_qna_dataset.csv`](Week%2002%20Simple%20RNN/Module%2007%20Simple%20GK%20Question%20and%20Answering%20System%20using%20Simple%20RNN/gk_qna_dataset.csv) (171 question/answer pairs, included in the repo). It upgrades the tokenizer to regex, builds a 243-word vocabulary, and uses the same `Embedding(50) → RNN(64) → Linear` architecture, trained 50 epochs with Adam (lr 1e-4); loss drops from 581 (epoch 10) to 120 (epoch 50). Only the **first token** of each answer is used as the target, so it is a single-word classifier rather than a generator; the sample predictions on training questions (Paris, water, Mars, ostrich, Tokyo) are all correct. The practice-problem write-up [`RNN_Module_3_Solutions.md`](Week%2002%20Simple%20RNN/Module%2007%20Simple%20GK%20Question%20and%20Answering%20System%20using%20Simple%20RNN/Practice%20Problem/RNN_Module_3_Solutions.md) answers 15 questions about this notebook, including why the first-token target is insufficient, what a proper encoder–decoder would need, and why building the vocabulary from the full dataset is a leakage problem.

### Week 03 – LSTM

**Module 08 – LSTM Basics.** Notes on the five problems of simple RNNs (vanishing/exploding gradients, short-term memory, no memory control, long-range dependencies), LSTM applications, and the cell components; several pages are image-only diagrams. The folder also includes a PDF copy of Christopher Olah's *Understanding LSTM Networks* blog post. My [`Solution.md`](Week%2003%20LSTM/Module%2008%20LSTM%20Basics/Practice%20Problem/Solution.md) answers five conceptual questions (RNN vs. LSTM on long sequences, cell state vs. hidden state, and the purpose of each gate) without formulas, as the assignment required.

**Module 09 – LSTM Gates.** Gate-by-gate notes (forget, input, output) and a ten-question assignment answered in [`Solution.md`](Week%2003%20LSTM/Module%2009%20LSTM%20Gates/Practice%20Problem/Solution.md): the formulas `f_t = σ(W_f[h_{t−1}, x_t] + b_f)`, `i_t`, `C̃_t = tanh(…)`, `C_t = f_t * C_{t−1} + i_t * C̃_t`, `o_t`, `h_t = o_t * tanh(C_t)`, a symbol table for each, why sigmoid is used for gates and tanh for candidates, and the complete cell workflow.

**Modules 10–11 – LSTM Project (next-word prediction).** The project is a word-level language model trained on a short story ("The Last Train Journey") embedded in the notebook. [Part 1](Week%2003%20LSTM/Module%2010%20LSTM%20Project%20%28Part%201%29/LSTM_Project_Part_01.ipynb) does the data engineering: NLTK `word_tokenize`, a 284-word vocabulary from `Counter`, 555 growing n-gram sequences (every prefix of every sentence), left-padding with zeros to length 25, and splitting into `X` (24 tokens) / `y` (next token) inside a `CustomDataset`. [Part 2](Week%2003%20LSTM/Module%2011%20LSTM%20Project%20%28Part%202%29/LSTM_Project.ipynb) completes `LSTMModel` = `Embedding(284, 100) → nn.LSTM(100, 150) → Linear(150, 284)` (the forward pass reads the final hidden state), inspects every intermediate tensor shape, trains 50 epochs with Adam (lr 1e-3) from loss 101.3 to 2.6, and then generates text token by token – the seed "Daniel checked the ticket" is continued to "in his pocket and walked toward the train" before degenerating into repeated full stops. A `calculate_accuracy` pass reports **96.94% next-word accuracy on the training sequences** (there is no held-out split). The Module 11 practice sheet asks for a train/test split, overfitting analysis and regularisation; it is included as questions only, with no solution in the repo.

### Week 04 – NLP

**Module 12 – Text Preprocessing with Regex.** Notes on what NLP is, the eight-stage NLP pipeline (using a product-review sentiment project as the running example), and regular-expression syntax. [`Module__regex.ipynb`](Week%2004%20NLP/Module%2012%20Text%20Preprocessing%20in%20NLP%20with%20Regex/Module__regex.ipynb) exercises character classes, `\d \w \W \D`, quantifiers (`+ * ? {m,n}`), anchors, wildcards, capture groups, and the `findall / search / split / sub / compile` API on phone-number and e-mail examples. [`NLP_Pipeline_Questions_Solutions.md`](Week%2004%20NLP/Module%2012%20Text%20Preprocessing%20in%20NLP%20with%20Regex/Practice%20Problem/NLP_Pipeline_Questions_Solutions.md) answers all 45 pipeline questions, from data acquisition through evaluation metrics (precision/recall/F1, when accuracy misleads) to deployment, monitoring and model drift.

**Module 13 – Text Preprocessing Fundamentals.** [`Text_preprocessing.ipynb`](Week%2004%20NLP/Module%2013%20Text%20Preprocessing%20Fundamentals/Text_preprocessing.ipynb) applies a full cleaning pipeline to the IMDB 50k movie-review dataset: lowercasing, HTML-tag and URL removal with regex, two ways to strip punctuation, chat-abbreviation expansion, TextBlob spell correction, NLTK stopword removal (applied to the whole column), emoji removal, NLTK and spaCy tokenization, Porter stemming, and WordNet lemmatization (with POS tags). The practice solution [`NLP_Preprocessing_Practice_Solutions.md`](Week%2004%20NLP/Module%2013%20Text%20Preprocessing%20Fundamentals/Practice%20Problem/NLP_Preprocessing_Practice_Solutions.md) explains each regex (`<.*?>`, `https?://\S+|www\.\S+`), `str.maketrans`, stemming vs. lemmatization, and traces one sentence through all eight stages.

**Module 14 – Text representation in NLP.** Notes on one-hot encoding, Bag of Words, n-grams (and their vocabulary explosion), and TF-IDF, all on the same four-document "students / watch / youtube" corpus. [`Phitron_bag_of_words.ipynb`](Week%2004%20NLP/Module%2014%20Text%20representation%20in%20NLP/Phitron_bag_of_words.ipynb) builds a binary `CountVectorizer`; [`tfidf.ipynb`](Week%2004%20NLP/Module%2014%20Text%20representation%20in%20NLP/tfidf.ipynb) fits a `TfidfVectorizer`, prints the matrix, vocabulary and IDF values, and shows how a sentence with an out-of-vocabulary word is transformed. The practice notebook [`Module Practice Solution.ipynb`](Week%2004%20NLP/Module%2014%20Text%20representation%20in%20NLP/Practice%20Problem/Module%20Practice%20Solution.ipynb) is the most complete piece of NLP work in the repo: it runs the whole Module 13 pipeline on the SMS Spam Collection (5,572 → 5,566 non-empty messages), counts 50,910 tokens / 8,413 unique words with plain Python, builds one-hot, BoW, bigram and trigram matrices (vocabulary 8,359 → 30,778 → 31,234 with sparsity above 99.9%), reconciles the `CountVectorizer` token count with `.split()`, lists the most frequent n-grams, and inspects the lowest/highest IDF words.

**Module 15 – Word2Vec.** Notes on embeddings, Word2Vec intuition, CBOW and Skip-gram. [`word2vec.ipynb`](Week%2004%20NLP/Module%2015%20Word2Vec/word2vec.ipynb) installs gensim, downloads the 300-d Google News vectors (≈1.6 GB) and loads the first 500,000 words, then explores them: vector shape `(300,)`, `most_similar` for *man*, *Bangladesh*, *facebook*, *Football*, cosine `similarity` (man/woman 0.766 vs. man/PHP −0.033), `doesnt_match(['PHP','java','monkey']) → monkey`, and the analogy `king − man + woman`, where *queen* is the top result after *king* itself (0.730). [`Solution.md`](Week%2004%20NLP/Module%2015%20Word2Vec/Practice%20Problem/Solution.md) enumerates the Skip-gram pairs for a five-word sentence with window 1, compares Word2Vec with one-hot on dimensionality/similarity/memory, and contrasts CBOW with Skip-gram.

**Module 16 – ADL Final Exam.** See [Evaluations / Exams](#evaluations--exams).

---

## Practical Work & Experiments

| Work | Task / Dataset | Model or Method | Main Learning | Link |
|---|---|---|---|---|
| Brain tumor MRI classifier | 4-class MRI classification; Kaggle *Brain Tumor MRI Dataset* (4,000 / 1,600 / 1,600 split) | 4-block CNN with BatchNorm + global average pooling (390 K params), augmentation, Adam + `ReduceLROnPlateau`, best-checkpoint saving | Full train/val/test workflow with curves, confusion matrix and per-class report; best val acc 93.62%, test acc 87.75% | [notebook](Week%2001%20CNN%20Fundamentals/Conceptual%20Sessions/Week%2001%20CS-02%20Complete%20CNN%20Project/Brain_Tumor_CNN.ipynb) |
| Brain tumor – small-data version | Same task, 100 images per class from Google Drive (280 / 60 / 60) | 3-block `SimpleCNN` with a large flatten layer (2.1 M params), 15 epochs | How augmentation and dropout behave on tiny datasets; noisy validation curve; test acc 78.33% on 60 images | [notebook](Week%2001%20CNN%20Fundamentals/Conceptual%20Sessions/Week%2001%20CS-02%20Complete%20CNN%20Project/Brain_Tumor_CNN_Beginner_100_Per_Class.ipynb) |
| PlantVillage leaf-disease CNN | 38-class plant disease images; Kaggle *PlantVillage* | Custom folder-walking `Dataset`, 3-block CNN (`MyCNN`), `CrossEntropyLoss` + Adam | Writing a `Dataset`/`DataLoader` pipeline from scratch; trained only on a 100-image `Subset` as a smoke test, so the 100% subset accuracy is not a real result | [notebook](Week%2001%20CNN%20Fundamentals/Module%2003%20CNN%20Project/PlantVillage.ipynb) |
| Image → array round-trip | Single JPEG (`flower.jpeg`, included) | PIL + NumPy | Images are integer arrays of shape `(H, W, 3)` | [notebook](Week%2001%20CNN%20Fundamentals/Module%2001%20Introduction%20to%20CNN/image.ipynb) |
| RNN anatomy | Toy input `(batch, 5 steps, 5 features)` | `nn.RNN(5, 3)` + `Linear` + `Sigmoid` | Reading the parameter breakdown (34 params) and the `batch_first` output layout | [notebook](Week%2002%20Simple%20RNN/Module%2005%20Simple%20RNN%20Basics%20of%20NLP/RNN_using_pytorch.ipynb) |
| QA with a simple RNN (v1) | Kaggle *100 Unique QA* (90 pairs) | Tokenizer → vocab → `Embedding(50)` → `RNN(64)` → `Linear`; 20 epochs | Text-to-tensor pipeline, `UNK` handling, confidence threshold at inference | [notebook](Week%2002%20Simple%20RNN/Conceptual%20Sessions/Week%2002%20CS-01%20Basic%20Anatomy%20of%20a%20Recurrent%20Neural%20Network%20%28RNN%29/RNN.ipynb) |
| GK question-answering RNN (v2) | `gk_qna_dataset.csv` (171 pairs, in repo) | Regex tokenizer, 243-word vocab, same RNN architecture; 50 epochs, lr 1e-4 | Why predicting only the first answer token limits the model; `squeeze`/`unsqueeze` shape handling | [notebook](Week%2002%20Simple%20RNN/Module%2007%20Simple%20GK%20Question%20and%20Answering%20System%20using%20Simple%20RNN/simple_gk_answring_system_using_rnn.ipynb) |
| LSTM next-word prediction | Short story (≈40 sentences) embedded in the notebook; 555 prefix sequences | `Embedding(100)` → `LSTM(150)` → `Linear(284)`; pre-padding to 25; 50 epochs | Building language-model training pairs, inspecting LSTM output/hidden/cell shapes, autoregressive generation; 96.94% next-word accuracy on training data | [Part 1](Week%2003%20LSTM/Module%2010%20LSTM%20Project%20%28Part%201%29/LSTM_Project_Part_01.ipynb) · [Part 2](Week%2003%20LSTM/Module%2011%20LSTM%20Project%20%28Part%202%29/LSTM_Project.ipynb) |
| Regex drills | Hand-written strings (phone numbers, e-mails, names/ages) | Python `re` | Character classes, quantifiers, anchors, groups, compiled patterns | [notebook](Week%2004%20NLP/Module%2012%20Text%20Preprocessing%20in%20NLP%20with%20Regex/Module__regex.ipynb) |
| IMDB review cleaning | Kaggle *IMDB Dataset of 50K Movie Reviews* | Regex, NLTK, spaCy, TextBlob | A reusable cleaning pipeline from raw HTML-laden reviews to lemmatized tokens | [notebook](Week%2004%20NLP/Module%2013%20Text%20Preprocessing%20Fundamentals/Text_preprocessing.ipynb) |
| BoW & TF-IDF on a toy corpus | Four short documents | `CountVectorizer(binary=True)`, `TfidfVectorizer` | Reading vocabulary, IDF values and OOV behaviour by hand | [BoW](Week%2004%20NLP/Module%2014%20Text%20representation%20in%20NLP/Phitron_bag_of_words.ipynb) · [TF-IDF](Week%2004%20NLP/Module%2014%20Text%20representation%20in%20NLP/tfidf.ipynb) |
| SMS spam text-representation practice | Kaggle *SMS Spam Collection* (5,566 messages after cleaning) | Full preprocessing pipeline, one-hot, BoW, bigram/trigram, TF-IDF | Vocabulary size vs. n-gram order (8,359 → 30,778 → 31,234), sparsity, IDF extremes | [notebook](Week%2004%20NLP/Module%2014%20Text%20representation%20in%20NLP/Practice%20Problem/Module%20Practice%20Solution.ipynb) |
| Pre-trained Word2Vec exploration | Google News 300-d vectors (first 500 K words) | gensim `KeyedVectors` | Similarity, odd-one-out and analogy queries on real embeddings | [notebook](Week%2004%20NLP/Module%2015%20Word2Vec/word2vec.ipynb) |

Metrics in this table are the values saved in the notebook outputs; nothing was re-run for this README.

---

## Evaluations / Exams

### Mid-term exam (Week 01, Module 04)

[Question notebook](Week%2001%20CNN%20Fundamentals/Module%2004%20ADL%20Mid%20Term%20Exam/Advance_DL_Mid_Term_Exam_Question.ipynb) · [My solution](Week%2001%20CNN%20Fundamentals/Module%2004%20ADL%20Mid%20Term%20Exam/Advance_DL_Mid_Term_Exam_Solution.ipynb)

- **Format:** 10 questions, 100 marks. Q1–Q9 are written theory on CNNs (why CNNs over dense nets, layer roles, filters, RGB shapes, padding/stride and the output-size formula, pooling, a worked numerical convolution, layer-by-layer shape calculation for a 64×64×3 input, and dataset challenges). Q7 and Q8 are answered as embedded photos of handwritten working.
- **Q10 – coding task (20 marks):** age-group classification on the Kaggle *Faces: Age Detection* dataset (19,906 face images labelled YOUNG / MIDDLE / OLD from `train.csv`).
  - Pipeline: `kagglehub` download → pandas label mapping → custom `AgeDataset` → stratified 80/20 split (15,924 / 3,982) with `train_test_split` → `DataLoader`s (batch 16).
  - Preprocessing: resize to 128×128, normalize to `[-1, 1]`; training augmentation with horizontal flip, ±10° rotation and colour jitter.
  - Model: four blocks of **two** `Conv2d(3×3, same) → ReLU` layers each, followed by `BatchNorm2d` and `MaxPool2d` (32 → 64 → 128 → 256 filters), then `Flatten → Linear(16384, 128) → Linear(128, 64) → Linear(64, 3)` with dropout 0.4, exactly as the question specified.
  - Training: `CrossEntropyLoss`, Adam (lr 1e-3), 20 epochs on a Colab T4, with loss/accuracy curves plotted.
  - **Recorded results:** training accuracy rises from 55.7% to 80.1%; **validation accuracy peaks at 79.43% (epoch 19)** and ends at 78.30%, with validation loss 0.52–0.58 in the last epochs. There is no separate test set in the exam setup. `compute_class_weight` is imported but class weighting was not applied.

### Final exam (Week 04, Module 16)

[Question notebook](Week%2004%20NLP/Module%2016%20ADL%20Final%20Exam/Advance_Deep_Learning_Final_Exam.ipynb) · [My solution](Week%2004%20NLP/Module%2016%20ADL%20Final%20Exam/Advance_Deep_Learning_Final_Exam_Solution.ipynb)

- **Format:** 20 fill-in-the-blank coding questions (5 marks each, 100 total), all executed with outputs in the solution notebook.
- **Part 1 – Simple RNN:** a NumPy forward step `h_t = tanh(W_xh·x_t + W_hh·h_{t−1} + b)`, unrolling over a sequence with shared weights, an `nn.RNN(3, 4)` whose 36 parameters match `hidden × (input + hidden + 2)`, the tanh derivative in BPTT, and one-step gradients `dW_hh`, `db`, `dh_prev`.
- **Part 2 – LSTM:** forget gate with concatenated `[h_prev, x_t]`, cell-state update `c_t = f⊙c_prev + i⊙g`, output gate `h_t = o⊙tanh(c_t)`, an `nn.LSTM(5, 8)` whose 480 parameters match `4·(h·(h+x) + 2h)`, and a tiny sentiment classifier `Embedding → LSTM(16) → Linear(16,1) → sigmoid` trained with `BCELoss` on six sentences.
- **Part 3 – NLP:** regex lower-case/punctuation pipeline, bigrams with `zip`, `TfidfVectorizer.fit_transform`, a NumPy cosine-similarity function, and training a gensim Skip-gram `Word2Vec(vector_size=10, window=2, min_count=1, sg=1)`.
- **Part 4 – CNN:** the output-size function `(n − f + 2p) // s + 1`, zero padding with `np.pad`, a manual stride-2 convolution, max vs. average pooling via `reshape(2,2,2,2)`, and an `nn.Sequential` CNN for 28×28 grayscale images (`Conv2d(1,8,3) → ReLU → MaxPool(2) → Flatten → Linear(1352, 10)`, 13,610 parameters).

Grades are not recorded in the repository, so none are reported here.

---

## Skills Developed & Learning Outcomes

What the work in this repository demonstrates:

- **Building PyTorch data pipelines from scratch** – custom `Dataset` classes over image folders, CSV label files and text corpora; `DataLoader`s with batching/shuffling; stratified splits; reproducible seeding.
- **Designing and training CNN classifiers** – choosing conv/BatchNorm/pooling stacks, computing flatten sizes, using augmentation and dropout, learning-rate scheduling, checkpointing the best validation model, and reading training curves, confusion matrices and per-class precision/recall.
- **Reasoning about shapes and parameter counts** – applying the convolution output-size formula and the RNN/LSTM parameter formulas by hand and verifying them against PyTorch.
- **Text-to-tensor preprocessing** – tokenization (manual, regex, NLTK, spaCy), vocabulary construction with `UNK`/padding tokens, index encoding, pre-padding sequences, and `nn.Embedding`.
- **Sequence modelling with RNNs and LSTMs** – implementing single-layer `nn.RNN`/`nn.LSTM` models, reading output vs. final hidden/cell states, training a language model, and generating text autoregressively.
- **Understanding recurrent-network theory** – forward propagation, BPTT, vanishing/exploding gradients, and the role of each LSTM gate, expressed both in formulas and in NumPy.
- **Classic NLP feature engineering** – regex-based cleaning, stopword removal, stemming vs. lemmatization, one-hot, Bag of Words, n-grams, TF-IDF, and interpreting IDF values and sparsity.
- **Working with word embeddings** – loading pre-trained Word2Vec vectors and using similarity, odd-one-out and analogy queries; training a small Skip-gram model.
- **Critical reading of results** – recognising when a recorded accuracy is not meaningful (100-image subset, training-set accuracy, 60-image test set) and documenting the limitation rather than the headline number.
- **Writing technical explanations** – the Markdown practice solutions explain design decisions (e.g. first-token targets, vocabulary leakage, seq2seq requirements) in plain language.

---

## How to Use

### Quick start (Google Colab, recommended)

All notebooks were written for Google Colab and contain Colab-specific cells:

- `!pip install …` and `!git clone …` shell cells.
- Dataset downloads via `kagglehub.dataset_download(...)`; the saved outputs show paths such as `/kaggle/input/...` or `/root/.cache/kagglehub/...`.
- Hard-coded `/content/...` paths (`/content/flower.jpeg`, `/content/Datasets/gk_qna_dataset.csv`, `/content/data/...`).
- `Brain_Tumor_CNN_Beginner_100_Per_Class.ipynb` mounts Google Drive and expects the dataset at `/content/drive/MyDrive/brain_tumor_mri_dataset/Training`.

To reproduce a notebook: open it in Colab, choose a **GPU runtime** for the CNN notebooks (they check `torch.cuda.is_available()` and fall back to CPU, but training on CPU will be slow), run the cells top to bottom, and upload or re-point any `/content/` file the notebook expects.

### Running locally

```bash
git clone https://github.com/ak1bhasan/Advanced_Deep_Learning.git
cd Advanced_Deep_Learning

# core stack used across the notebooks
pip install torch torchvision numpy pandas matplotlib scikit-learn pillow kagglehub jupyter

# NLP extras (Week 03–04 only)
pip install nltk spacy textblob gensim
python -m spacy download en_core_web_sm
python -c "import nltk; [nltk.download(p) for p in ('punkt', 'punkt_tab', 'stopwords', 'wordnet')]"

jupyter notebook
```

No package versions are pinned because the repository has no dependency file; the saved outputs show PyTorch 2.11 (CUDA 12.8) and gensim 4.4 in Colab, but any recent versions should work. These commands have not been separately tested outside Colab. When running locally, replace the `/content/...` and `/kaggle/input/...` paths with your own, and skip the `google.colab` import in the Drive-mounted notebook.

### Datasets and large assets

| Asset | Used by | How it is obtained |
|---|---|---|
| PlantVillage (`mohitsingh1804/plantvillage`, ~818 MB) | Module 03, CS-01 | `kagglehub` download (needs Kaggle access) |
| Brain Tumor MRI (`masoudnickparvar/brain-tumor-mri-dataset`) | CS-02 | `kagglehub` download, or a copy in Google Drive for the beginner notebook |
| Faces: Age Detection (`arashnic/faces-age-detection-dataset`) | Mid-term exam | `kagglehub` download |
| 100 Unique QA (`aadarshvani/100-unique-qa-dataset`) | Week 02 CS-01 | `kagglehub` download |
| `gk_qna_dataset.csv` (171 rows) | Module 07 | **Included in the repo**; the notebook itself clones it from `github.com/tajuar-akash-hub/Datasets` |
| IMDB 50K reviews (`lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`) | Module 13 | `kagglehub` download |
| SMS Spam Collection (`uciml/sms-spam-collection-dataset`) | Module 14 practice | `kagglehub` download |
| GoogleNews-vectors-negative300 (~1.6 GB) | Module 15 | `gensim.downloader` or the `wget` mirror used in the notebook; loaded with `limit=500000` to save RAM |
| `flower.jpeg` | Module 01 | **Included in the repo** |

No trained model checkpoints are stored in the repository. `Brain_Tumor_CNN.ipynb` writes `best_brain_tumor_cnn.pth` to the Colab working directory during training only.

### Suggested reading order

Follow the week and module numbers. Within Week 03, run Module 10's Part 1 notebook before Module 11's Part 2 (Part 2 repeats the data preparation, so it also works standalone). The `Phitron_*` notebooks are the reference versions; the unprefixed notebooks are my own runs and are the ones linked throughout this README.

---

## Acknowledgements

- **[Phitron](https://phitron.io)** – the Advanced Deep Learning course that this repository follows; the `Phitron_*` notebooks, the module notes and the exam papers are course material. The notes identify the instructor as **Akash**, whose GitHub dataset repository (`tajuar-akash-hub/Datasets`) supplies the GK question-answering data.
- **Christopher Olah – [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)** – included as a PDF in Module 08 and the basis for the gate diagrams.
- **Kaggle dataset authors** – PlantVillage (mohitsingh1804), Brain Tumor MRI (masoudnickparvar), Faces Age Detection (arashnic), 100 Unique QA (aadarshvani), IMDB 50K (lakshmi25npathi), SMS Spam Collection (uciml).
- **Google News Word2Vec** vectors via gensim, and the blog posts / animations listed in the Module 02 and Module 05 `.docx` link files (Medium, GeeksforGeeks, deeplizard).
