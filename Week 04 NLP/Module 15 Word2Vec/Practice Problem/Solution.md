# Word2Vec Practice Solution


## Q1. Skip-gram with window size = 1

**Sentence:** "I love machine learning today"

**Tokens:** `I`, `love`, `machine`, `learning`, `today`

With window size = 1, each word looks one step to its left and one step to its right. In Skip-gram, the **center word is the input** and each **neighbor is a target** it has to predict.

### Input-target pairs (input → target)

| Input (center word) | Targets (neighbors) |
|---|---|
| I | love |
| love | I, machine |
| machine | love, learning |
| learning | machine, today |
| today | learning |

Written out as a flat list, that gives **8 pairs**:

1. (I, love)
2. (love, I)
3. (love, machine)
4. (machine, love)
5. (machine, learning)
6. (learning, machine)
7. (learning, today)
8. (today, learning)

The first and last words only have one neighbor, which is why they only produce one pair each.

### Which word is the input when predicting "I"?

The only pair where "I" is the target is **(love, I)**, so the input is **"love"**.

### Which word is the input when predicting "machine"?

"machine" shows up as a target twice: **(love, machine)** and **(learning, machine)**. So the inputs are **"love"** and **"learning"**.

---

## Q2. Why Word2Vec instead of One-Hot Encoding?

**The core idea:** One-hot encoding gives each word an ID card. Word2Vec gives each word a position in a "meaning space", so words used in similar ways end up near each other.

---

## 1. Dimensionality

With one-hot encoding, every word becomes a vector as long as the whole vocabulary. If your vocabulary has 50,000 words, each word is a 50,000-number vector with a single `1` and 49,999 `0`s. Add more words and every vector gets longer.

Word2Vec instead represents each word as a short, dense vector, commonly around 100 to 300 numbers. That length is a design choice, and it stays the same however big the vocabulary gets. Typical sizes vary by project, so treat 300 as a common example, not a rule.

---

## 2. Similarity Between Words

One-hot vectors carry no meaning. Any two different words are exactly perpendicular, so their cosine similarity is **0**. To a one-hot model, "king" and "queen" are as unrelated as "king" and "banana".

Word2Vec learns from the contexts words appear in, so words used similarly get similar vectors. "King" and "queen" end up close, and "king" and "banana" end up far apart. It's also often shown that vector arithmetic can capture relationships, like `king - man + woman` landing near `queen`. That's a well-known demonstration, but it isn't perfect, and results depend on the training data.

---

## 3. Memory Usage

A rough estimate for 50,000 words stored as 32-bit floats (4 bytes per number). This is back-of-the-envelope math, so please verify it:

| Method | Matrix shape | Approx. size |
|---|---|---|
| One-hot (dense) | 50,000 × 50,000 | about 10 GB |
| Word2Vec (300 dims) | 50,000 × 300 | about 60 MB |

To be fair to one-hot, it can be stored as a sparse matrix, since only one value per row is non-zero. That reduces memory a lot. But even stored sparsely, it still has no notion of similarity, and any layer that uses it still deals with a huge input size.

---

## Summary

| | One-Hot Encoding | Word2Vec |
|---|---|---|
| **Dimensionality** | As large as the vocabulary, grows with it | Small and fixed (e.g., 300) |
| **Similarity** | All words equally unrelated (cosine = 0) | Similar words have similar vectors |
| **Memory** | Very large unless stored sparsely | Compact |

**In one sentence:** Word2Vec is smaller, fixed in size, and meaningful, while one-hot is huge, grows with the vocabulary, and treats every word as equally unrelated.

---

## Q3. CBOW vs Skip-gram

| | **CBOW** (Continuous Bag of Words) | **Skip-gram** |
|---|---|---|
| **Input** | The surrounding context words | One center (target) word |
| **Output** | The center word | The surrounding context words |
| **Direction** | Context → target | Target → context |
| **Typical strengths** | Faster to train, works well for frequent words | Tends to do better on rare words and smaller datasets |

The "typical strengths" row is the commonly reported tendency in the Word2Vec literature, not a guarantee. Results can vary with your data and settings.

**Which one predicts context from a target word?**
**Skip-gram.** CBOW does the opposite: it predicts the target from its context.

A simple way to remember it: CBOW *bags up* the neighbors to guess the middle word, while Skip-gram *skips out* from the middle word to guess its neighbors.
