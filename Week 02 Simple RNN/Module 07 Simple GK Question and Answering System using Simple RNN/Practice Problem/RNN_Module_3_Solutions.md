# RNN (Module 3) – Practice Problem Solutions


## Q1. What is a dataset? Why is it needed before training?

A **dataset** is a collection of examples that a model learns from. Each example usually has:
- an **input** (e.g., a question), and
- a **target/label** (e.g., the correct answer).

**Why it is needed:** A model starts with random parameters and knows nothing. It learns patterns only by looking at many examples and correcting its mistakes. Without data, there is nothing to learn from or to measure performance on. Datasets are usually split into **train** (learn), **validation** (tune), and **test** (final check).

---

## Q2. What is tokenization? Is a whole sentence broken into small tokens?

**Tokenization** means splitting text into small pieces called **tokens**. Tokens can be words, sub-words, or characters.

Example (word-level):

```
"What is the capital of France"
→ ["what", "is", "the", "capital", "of", "france"]
```

**Yes**, a sentence is split into tokens before going into a neural network, because a network cannot read a whole sentence as one thing. It needs small, repeatable units that can each be turned into a number.

---

## Q3. If "What" and "what" are different words, what is the problem?

If the text is *not* lowercased, the model treats `"What"` and `"what"` as two unrelated words. Problems:
- The **vocabulary gets bigger** with duplicates (What, what, WHAT…).
- The model's learning is **split** between them. Each version is seen fewer times, so it learns each one less well.
- It may **fail to generalize** (it learned `"what"`, but a test sentence has `"What"`).

Lowercasing fixes this by merging them into one word.
*Trade-off:* it can lose useful information (e.g., `Apple` the company vs `apple` the fruit), but for simple tasks the benefit is usually larger.

---

## Q4. What is a vocabulary? How do words map to numbers?

A **vocabulary** is a dictionary that maps every known token to a unique integer index (and often back again).

Small example:

```python
vocab = {"<PAD>": 0, "<UNK>": 1, "what": 2, "is": 3, "the": 4, "capital": 5, "of": 6, "france": 7}

sentence = ["what", "is", "capital"]
indices = [vocab[w] for w in sentence]   # [2, 3, 5]
```

So `"what" → 2`, `"is" → 3`, `"capital" → 5`. These numbers can then be turned into a tensor.

---

## Q5. What is the `<UNK>` token? What happens with a word not in the vocabulary?

`<UNK>` (unknown) is a special token used for any word that is **not in the vocabulary**.

If the model sees an unseen word such as `"Dhaka"`:
- Without `<UNK>`, a plain dictionary lookup would give a `KeyError` (crash).
- With `<UNK>`, the word is replaced with the `<UNK>` index (e.g., `0`).

The model keeps running, but all unseen words look the same to it, so their specific meaning is lost.

---

## Q6. Can "What is the capital of France" be fed directly to a PyTorch model?

**No.** Neural networks work on **numerical tensors**, not strings. The text must go through this pipeline:

```
text → tokenize → indices (via vocabulary) → tensor → (embedding) → model
```

```python
import torch
tokens  = "what is the capital of france".split()
indices = [vocab.get(w, vocab["<UNK>"]) for w in tokens]
x = torch.tensor(indices)    # tensor of integers, now usable
```

---

## Q7. Dataset vs DataLoader. What do `batch_size` and `shuffle` do?

| | Dataset | DataLoader |
|---|---|---|
| Role | Stores samples and says **how to get one sample** (`__getitem__`) and **how many** (`__len__`) | Wraps a Dataset and **delivers the data in batches** during training |

- **`batch_size`**: how many samples are processed together in one step (e.g., 32). It is faster than going one-by-one and gives smoother gradient updates than using a single sample.
- **`shuffle=True`**: mixes the order of samples every epoch, so the model does not learn from the order of data, and each batch is different.

Are they needed in training? In practice, yes: batching makes training efficient and memory-friendly, and shuffling usually helps learning. Shuffling is normally turned **off** for validation/test (the order does not matter there).

```python
from torch.utils.data import DataLoader
loader = DataLoader(my_dataset, batch_size=32, shuffle=True)
```

---

## Q8. What does `nn.Embedding()` do? Why is it better than a random integer?

`nn.Embedding(num_embeddings, embedding_dim)` is a **lookup table** (a learnable matrix). Given a word index, it returns that word's **vector** of size `embedding_dim`.

```python
emb = nn.Embedding(num_embeddings=1000, embedding_dim=8)
emb(torch.tensor([2, 3, 5])).shape   # (3, 8)  -> 3 words, each an 8-number vector
```

**Why it is better than one integer per word:**
- An integer implies a **fake order/size** (is word 500 "bigger" than word 5? No meaning).
- One number cannot carry rich meaning; a **vector** can.
- The vectors are **learned during training**, so similar words can end up with similar vectors.

---

## Q9. `squeeze()` vs `unsqueeze()`. Why use `unsqueeze(0)` for a single sequence?

- **`unsqueeze(dim)`** → **adds** a dimension of size 1 at position `dim`.
- **`squeeze()`** → **removes** dimensions of size 1.

RNNs usually expect a **batch dimension**. A single sequence has none, so we add one with `unsqueeze(0)`.

Example (assuming `batch_first=True`; this depends on your notebook's setting):

```python
x = torch.randn(5, 8)      # 5 words, 8-dim embeddings  -> shape (5, 8)
x = x.unsqueeze(0)         # add batch dim              -> shape (1, 5, 8)
# ... after the model ...
y = y.squeeze(0)           # remove batch dim again
```

| Step | Shape |
|---|---|
| Before `unsqueeze(0)` | `(5, 8)` |
| After `unsqueeze(0)` | `(1, 5, 8)` = (batch, seq_len, features) |

*Note:* Without `batch_first=True`, `nn.RNN` expects `(seq_len, batch, features)`. Check your notebook's setting. Also, `target = answer[0][0].unsqueeze(0)` turns a single number into a shape `(1,)`, which is a "batch of 1" for the loss function.

---

## Q10. How does an RNN process a sequence? What is the hidden state?

An RNN reads a sequence **one word at a time**, from left to right. At every step it combines:
1. the **current word** (its embedding), and
2. the **hidden state** from the previous step.

The **hidden state** is a vector acting as the network's **memory**, a summary of what it has read so far.

Basic formula (from the standard RNN definition in the PyTorch docs, with tanh):

```
h_t = tanh( W_ih · x_t + b_ih + W_hh · h_(t-1) + b_hh )
```

**How earlier words affect later ones:** `h_(t-1)` (which contains information about earlier words) is fed into the calculation of `h_t`. So each new hidden state carries a trace of all previous words, passed along step by step.

*Limitation:* plain RNNs struggle to remember things from far back (the "vanishing gradient" problem). LSTM and GRU were designed to handle this better.

---

## Q11. What does `nn.CrossEntropyLoss()` measure? Is it suitable for predicting one word from a large vocabulary?

It measures **how well the predicted probability for the correct class matches the truth**. Roughly, it is `-log(probability the model gave to the correct word)`:
- Correct word given high probability → **small loss**.
- Correct word given low probability → **large loss**.

Details:
- It takes **raw scores (logits)** of shape `(batch, vocab_size)` and the **correct class index** of shape `(batch,)`.
- It internally combines `LogSoftmax` + `NLLLoss`, so you should **not** apply softmax yourself first.

**Suitable?** Yes. Predicting one word out of many is a **multi-class classification** problem (each vocabulary word = one class), and this is the standard loss for it. (With very large vocabularies it can be slow, and other tricks exist, but it is still the normal choice.)

---

## Q12. After `loss.backward()`, what does the Adam optimizer do? Why update many times?

Order in a training step:

```python
optimizer.zero_grad()   # clear old gradients
loss.backward()         # compute gradients (how each parameter affects loss)
optimizer.step()        # UPDATE parameters using those gradients
```

- `loss.backward()` only **calculates gradients**.
- `optimizer.step()` is what actually **changes the parameters**. **Adam** does this using the gradient plus running averages of past gradients, and it adapts the step size for each parameter.

**Why not just once?** One update moves the parameters only a small step (controlled by the learning rate), and one batch only shows a small part of the data. Learning needs **many updates** over many batches and epochs to gradually reduce the loss.

---

## Q13. Is using only the first answer token as the target enough for QA?

**No.** With `target = answer[0][0]`, the model is trained to predict **only the first token** of the answer. It never learns the rest, so it cannot produce a complete answer.

**How the objective should change for full answers:**
1. Add special tokens: `<SOS>` (start) and `<EOS>` (end) to the answer.
2. Train the model to predict the **next token at every position** of the answer.
3. Compute the loss at **every answer position** and average/sum it.
4. Use **teacher forcing**: during training, feed the true previous answer token as the next input.
5. Use `ignore_index` in `CrossEntropyLoss` so padding tokens are not counted.

Example:

```
Answer:   <SOS>  paris  is  the  capital  <EOS>
Input →   <SOS>  paris  is  the  capital
Target →  paris  is     the capital <EOS>
```

---

## Q14. Is "RNN over the question, then predict one token" enough? What changes for a seq2seq QA model?

**Not enough** for full answers; it outputs just one token.

A proper **sequence-to-sequence (encoder–decoder)** model needs:

**Architecture**
- **Encoder** (RNN/GRU/LSTM): reads the whole question; its final hidden state is a summary ("context").
- **Decoder** (RNN/GRU/LSTM): starts from the encoder's hidden state and generates the answer **one token at a time**, using its previous output as the next input.
- An **output layer** (`Linear` to vocabulary size) at **every decoder step**.
- (Optional but common improvement) **Attention**, so the decoder can look back at different parts of the question.

**Training**
- Loss at every decoder step (see Q13), with `<SOS>`/`<EOS>`, teacher forcing and padding handling.

**Inference (generating)**
- Start with `<SOS>`, predict a token, feed it back in, and repeat until `<EOS>` or a maximum length. Greedy or beam search can be used.

---

## Q15. Building the vocabulary from the full dataset: is it a data leakage problem?

**Yes, it is an evaluation-design problem (a mild form of data leakage).** If the vocabulary is built from train + test data together, the test set's information has influenced a part of your pipeline before evaluation.

Why it matters:
- The test set will contain **no truly unseen words** (no `<UNK>`), which is **unrealistic**. In real use, new words will appear.
- This can make results look **better/more optimistic** than they really are.
- *Honest caveat:* words that appear only in the test set would still have **untrained** embeddings, so the effect on accuracy may be small in practice. But the evaluation is still not a fair simulation of real use, and it is considered bad practice.

**Correct way with train/validation/test splits:**
1. **Split the data first.**
2. Build the vocabulary **from the training set only**.
3. Use that same vocabulary to convert validation and test data. Words not found become `<UNK>`.
4. (Optional) Ignore very rare training words (min frequency) so the model also learns to handle `<UNK>`.

```python
from collections import Counter

counter = Counter(w for sent in train_sentences for w in sent)   # TRAIN only
vocab = {"<PAD>": 0, "<UNK>": 1}
for word in counter:
    vocab[word] = len(vocab)

encode = lambda sent: [vocab.get(w, vocab["<UNK>"]) for w in sent]
# use encode() on train, validation AND test
```

---
