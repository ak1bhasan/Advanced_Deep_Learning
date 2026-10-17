# NLP Preprocessing Practice Questions and Solutions

Raw text is messy. It contains inconsistent capitalization, HTML tags, links, punctuation, emoji, and many words that carry little meaning on their own. Text preprocessing is the set of cleaning and normalization steps that turn this raw text into a more consistent form before it is given to an NLP model. Good preprocessing usually makes the vocabulary smaller, removes noise, and helps a model focus on useful patterns. However, no single recipe fits every task, so each step should be chosen based on what the model needs to do.

**Environment note.** All code outputs in this document were produced by actually running the code with Python 3 and NLTK 3.10.3. Outputs from NLTK components (tokenizer, stemmer, lemmatizer) can differ between versions, so verify them in your own environment. Some NLTK resources must be downloaded once:

```python
import nltk

nltk.download("punkt_tab")   # tokenizer data (newer NLTK versions)
nltk.download("punkt")       # tokenizer data (older NLTK versions)
nltk.download("wordnet")     # needed by WordNetLemmatizer
```

---

## 1. Corpus

### What is a corpus?

A corpus is a large, organized collection of text (or speech transcripts) that is gathered for studying language or training and testing language-based systems. The plural is "corpora". Examples include a collection of news articles, a set of tweets, or a set of product reviews.

A corpus is different from a single document. One review is one document, while the whole collection of reviews together is the corpus. In practice, a corpus is the raw material that vocabulary building, statistics, and model training are computed from.

### Is the IMDB movie review dataset considered a corpus? Why?

Yes. The IMDB movie review dataset is a collection of many movie reviews written by different people. It was assembled for a specific language task, which is sentiment analysis (deciding whether a review is positive or negative). Since it is a structured collection of text built for analysis and model training, it qualifies as a corpus. A single review from it is just one document inside that corpus.

---

## 2. Lowercasing

### Why do we perform lowercasing?

Computers treat `PHITRON` and `phitron` as different strings, even though they mean the same thing to a human. Lowercasing converts all letters to lowercase so that the same word is represented in one consistent way.

### What will be the output after lowercasing?

```python
text = "PHITRON is Amazing! I love studying NLP at PHITRON."
print(text.lower())
```

```text
phitron is amazing! i love studying nlp at phitron.
```

Input → Output:

`PHITRON is Amazing! I love studying NLP at PHITRON.` → `phitron is amazing! i love studying nlp at phitron.`

### How does lowercasing help reduce vocabulary size?

Without lowercasing, `PHITRON`, `Phitron`, and `phitron` would be three separate vocabulary entries. After lowercasing they all become the same token `phitron`. This removes duplicate entries, reduces sparsity, and lets the model collect more evidence for a single word instead of splitting it across variants.

**Limitation.** Lowercasing is not always appropriate. In named entity recognition, capitalization can signal a proper noun (for example, `Apple` the company versus `apple` the fruit). In such tasks, it may be better to keep the original case.

---

## 3. HTML Tag Removal

### Which regex pattern can remove HTML tags?

The pattern `<.*?>` can remove simple HTML tags.

```python
import re

text = "<p>Welcome to <b>PHITRON</b>!</p>"
clean_text = re.sub(r"<.*?>", "", text)

print(clean_text)
```

```text
Welcome to PHITRON!
```

### Explain why the pattern `<.*?>` works

| Part | Meaning |
|---|---|
| `<` | Matches the opening angle bracket. |
| `.` | Matches any character except a newline (by default). |
| `*` | Zero or more repetitions of the previous item. |
| `?` | After `*`, makes the repetition non-greedy, so it matches as little text as possible while still satisfying the pattern. |
| `>` | Matches the closing angle bracket. |

Together, the pattern means: start at a `<`, take the smallest amount of text possible, and stop at the first `>`.

### What is the purpose of `?` in this regex?

Here `?` switches the quantifier from greedy to non-greedy (lazy). To see the difference, compare both patterns on the same input:

```python
import re

text = "<p>Welcome to <b>PHITRON</b>!</p>"

print(re.findall(r"<.*?>", text))   # non-greedy
print(re.findall(r"<.*>", text))    # greedy

print(repr(re.sub(r"<.*?>", "", text)))
print(repr(re.sub(r"<.*>", "", text)))
```

```text
['<p>', '<b>', '</b>', '</p>']
['<p>Welcome to <b>PHITRON</b>!</p>']
'Welcome to PHITRON!'
''
```

- The greedy pattern `<.*>` starts at the first `<` and keeps going as far as possible, so it ends at the very last `>`. The whole string becomes one big match, and removing it deletes everything.
- The non-greedy pattern `<.*?>` stops at the first `>` it finds. This gives four separate tag matches (`<p>`, `<b>`, `</b>`, `</p>`), and the text between them is preserved.

**Limitation.** Regex is fine for simple examples like this one. For complicated or malformed HTML (nested attributes containing `>`, scripts, comments), a proper HTML parser such as BeautifulSoup is more reliable.

---

## 4. URL Removal

### Explain the regex `r'https?://\S+|www\.\S+'`

The pattern has two alternatives separated by `|`. The first alternative matches URLs that start with `http://` or `https://`. The second matches URLs that start with `www.`.

```python
import re

text = "For more details visit https://phitron.io or www.phitron.io"
clean_text = re.sub(r'https?://\S+|www\.\S+', '', text)

print(repr(clean_text))
```

```text
'For more details visit  or '
```

Both URLs are deleted, but the spaces around them remain (two spaces before `or`, and one space at the end). We can normalize whitespace afterwards:

```python
normalized = re.sub(r'\s+', ' ', clean_text).strip()
print(repr(normalized))
```

```text
'For more details visit or'
```

### What do `https?`, `\S+`, and `www\.` mean?

- `https?` matches `http` or `https`. The `?` makes the preceding `s` optional.
- `://` matches those three characters literally.
- `|` means OR between the two alternative patterns.
- `\S+` matches one or more non-whitespace characters. It keeps consuming characters until it reaches a space, tab, or newline.
- `www\.` matches the literal text `www.`. In regex, a plain `.` means "any character", so the backslash in `\.` makes it a literal period.
- The `r` prefix makes it a raw string, so Python does not interpret backslashes itself. Without it, you would need to write `\\S` and `\\.`, which is harder to read and easier to get wrong. Inside a raw string the correct pattern is `www\.\S+` (not `www\\.\S+`).

**Limitation.** Because `\S+` stops only at whitespace, a trailing punctuation mark attached to a URL (for example `https://phitron.io.` at the end of a sentence) would be removed together with the URL. This simple pattern is not suitable for every URL format, so more robust patterns may be needed for complicated text.

---

## 5. Punctuation Removal

```python
import string

text = "Hello, NLP!"
clean_text = text.translate(
    str.maketrans('', '', string.punctuation)
)

print(clean_text)
```

```text
Hello NLP
```

### What is the role of `string.punctuation`?

`string.punctuation` is a string constant from Python's standard library containing the ASCII punctuation characters (such as `!`, `?`, `.`, `,`, `;`, `:`). It tells `maketrans()` which characters should be deleted. It covers ASCII punctuation only, so it does not include every Unicode punctuation symbol (for example curly quotes or some non-English punctuation), and it does not remove emoji.

### Why are the first two arguments of `maketrans()` empty strings?

`str.maketrans(x, y, z)` builds a translation table.

- The first two arguments (`x` and `y`) are used for character-to-character replacement, where each character in `x` is mapped to the character at the same position in `y`. We do not want to replace anything, so both are empty strings.
- The third argument `z` lists the characters to delete. That is where `string.punctuation` goes.

Then `translate()` applies the table to the string and returns the new string.

### What would happen if punctuation is not removed before tokenization?

If you split on whitespace, punctuation stays attached to words. For example, `hello` and `hello!` become different tokens, which inflates the vocabulary and splits the counts of what is really one word. Tokenizers such as `word_tokenize` separate punctuation into its own tokens, which avoids this issue but adds extra punctuation tokens that you may or may not want.

**Limitation.** Punctuation should not always be removed. It can carry meaning in sentiment analysis (`!!!` can suggest strong emotion), sentence segmentation, and other tasks.

---

## 6. Stopword Removal

### What are stopwords?

Stopwords are very common words such as `i`, `am`, `in`, and `the`. They appear in almost every text and usually carry little topic-specific meaning, so they are sometimes removed to reduce noise and size.

### Which words would likely be removed?

For the sentence `I am studying Machine Learning in the university.`, a typical English stopword list would likely remove `I`, `am`, `in`, and `the`. The exact result depends on the library and the list used, since different libraries ship different stopword lists. The list below is only an illustrative example.

```python
text = "I am studying Machine Learning in the university."

stop_words = {"i", "am", "in", "the"}
tokens = text.lower().split()

filtered_tokens = [
    word for word in tokens
    if word not in stop_words
]

print(filtered_tokens)
```

```text
['studying', 'machine', 'learning', 'university.']
```

Notice that the period is still attached to `university.`. This is because `split()` only splits on whitespace and does not remove punctuation. Punctuation would need to be removed separately.

### Why do we convert stopwords into a set instead of keeping them as a list?

Checking `word in some_list` scans the list item by item, which takes time proportional to the list length. Checking `word in some_set` uses hashing, which is typically much faster on average (roughly constant time). Since we perform this check for every token in a large corpus, a set is usually preferred.

**Limitation.** Stopword removal is task-dependent. Words like `not` can be crucial in sentiment analysis ("not good" versus "good"), so blindly removing every stopword can damage meaning.

---

## 7. Tokenization

### What is tokenization?

Tokenization is the process of splitting text into smaller units called tokens. Tokens are most often words, but they can also be sentences, subwords, or characters. It is usually one of the first steps because most later steps (stopword removal, stemming, counting) work on tokens.

### What is the difference between `text.split()` and `word_tokenize(text)`?

```python
from nltk.tokenize import word_tokenize

text = "I am going to visit Dhaka!"

print(text.split())
print(word_tokenize(text))
```

```python
['I', 'am', 'going', 'to', 'visit', 'Dhaka!']
```

```python
['I', 'am', 'going', 'to', 'visit', 'Dhaka', '!']
```

- `text.split()` splits only on whitespace by default. It knows nothing about punctuation, so `Dhaka!` stays as one token.
- `word_tokenize(text)` applies NLTK's English tokenization rules, which generally separate punctuation into its own tokens (and also handle things like contractions).

Exact NLTK behavior may depend on the installed version and tokenizer resources. If you get a resource error (for example a missing `punkt` or `punkt_tab`), download it once with `nltk.download(...)` as shown in the environment note above. The needed resource name can differ between NLTK versions.

### Which one handles punctuation better?

`word_tokenize()` handles punctuation better for general English text, because it separates `Dhaka` and `!` into different tokens instead of keeping them glued together.

---

## 8. Stemming

### What is stemming?

Stemming is a process that reduces words to an approximate root form (a stem) by chopping off or modifying affixes such as `-ing`, `-es`, and `-ness`, using heuristic rules.

### Apply stemming conceptually to each word

The results below come from NLTK's `PorterStemmer`. Other stemmers (for example Snowball or Lancaster) can give different results.

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["running", "studies", "playing", "happiness"]

for word in words:
    print(word, "->", stemmer.stem(word))
```

```text
running -> run
studies -> studi
playing -> play
happiness -> happi
```

| Word | Porter stem | Comment |
|---|---|---|
| running | run | `-ing` removed and the doubled `n` reduced. |
| studies | studi | Plural ending changed, but `studi` is not a real word. |
| playing | play | `-ing` removed. |
| happiness | happi | `-ness` removed and `y` changed to `i`, so the stem is not a real word. |

### Why might stemming produce words that are not found in a dictionary?

A stemmer applies fixed rules to strings of characters. It does not look words up in a dictionary and does not understand grammar or meaning. Its goal is only to map related word forms to the same stem, not to produce a valid word. That is why you get stems like `studi` and `happi`.

---

## 9. Lemmatization

### What is lemmatization?

Lemmatization reduces a word to its dictionary base form, called a lemma. Unlike stemming, it uses vocabulary and morphological analysis, so the result is normally a real word.

### What would be the lemma of each word?

The expected lemmas depend on the part of speech (POS) assumed. These results were produced with NLTK's `WordNetLemmatizer`:

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

examples = [
    ("running", "v"),
    ("children", "n"),
    ("better", "a"),
    ("studies", "n"),
]

for word, pos in examples:
    print(word, f"(pos={pos})", "->", lemmatizer.lemmatize(word, pos=pos))
```

```text
running (pos=v) -> run
children (pos=n) -> child
better (pos=a) -> good
studies (pos=n) -> study
```

| Word | Assumed POS | Lemma |
|---|---|---|
| running | verb (`v`) | run |
| children | noun (`n`) | child |
| better | adjective (`a`), comparative of "good" | good |
| studies | noun (`n`) or verb (`v`) | study |

The POS assumption matters. With the default noun setting, `WordNetLemmatizer` does not give all the expected lemmas. In the version tested here, the results for different POS values were:

```text
running  -> n: running, v: run,   a: running, r: running
children -> n: child,   v: children
better   -> n: better,  v: better, a: good,   r: well
studies  -> n: study,   v: study,  a: studies, r: studies
```

So calling `lemmatizer.lemmatize("running")` without a POS returns `running`, and `lemmatizer.lemmatize("better")` returns `better`. Different lemmatizers (for example spaCy) may return different results, so do not assume every lemmatizer behaves the same way.

### Why is lemmatization generally more accurate than stemming?

- It uses a dictionary and, when given, POS information, so it returns valid words (`study` instead of `studi`).
- It can handle irregular forms that rule-based stemming cannot, such as `children` → `child` and `better` → `good`.
- The trade-off is that it needs more resources (a lexicon such as WordNet) and more processing time, and it usually needs POS information to be correct.

---

## 10. Complete NLP Pipeline

### Input

The input is one document with two lines:

```text
OMG!!! I was studying NLP at PHITRON 😊.
Visit https://phitron.io for more details.
```

```python
raw = "OMG!!! I was studying NLP at PHITRON 😊.\nVisit https://phitron.io for more details."
```

### Assumptions used throughout

| Item | Choice |
|---|---|
| URL regex | `r'https?://\S+\|www\.\S+'` |
| Emoji removal | A regex over common emoji Unicode ranges (no third-party library) |
| Whitespace | Normalized with `re.sub(r'\s+', ' ', text).strip()` right after emoji removal, which also turns the line break into a space |
| Punctuation | ASCII punctuation from `string.punctuation` |
| Tokenizer | NLTK `word_tokenize` |
| Stopword set | `{"i", "was", "at", "for", "more"}` (a small, explicitly defined set, so that `omg`, `nlp`, and `visit` are kept) |
| Stemmer | NLTK `PorterStemmer` |
| Lemmatizer | NLTK `WordNetLemmatizer`, tried with verb POS and with the default noun POS |

All outputs below were produced by running this code with NLTK 3.10.3.

### Stage 1: Lowercasing

Converts all letters to lowercase. The line break is preserved at this stage (shown as `\n`).

```python
s1 = raw.lower()
print(repr(s1))
```

```text
'omg!!! i was studying nlp at phitron 😊.\nvisit https://phitron.io for more details.'
```

### Stage 2: URL removal

Input: the output of Stage 1.

```python
import re

s2 = re.sub(r'https?://\S+|www\.\S+', '', s1)
print(repr(s2))
```

```text
'omg!!! i was studying nlp at phitron 😊.\nvisit  for more details.'
```

The URL is gone, and two spaces are left between `visit` and `for`.

### Stage 3: Emoji removal

Input: the output of Stage 2. `string.punctuation` does not contain emoji, so a separate regex is used. It removes characters in several common emoji Unicode ranges, plus the variation selector (`\uFE0F`) and the zero-width joiner (`\u200D`) that often accompany emoji.

```python
emoji_pattern = re.compile(
    "[\U0001F300-\U0001FAFF"   # symbols, pictographs, emoticons, and more
    "\U00002600-\U000027BF"    # miscellaneous symbols and dingbats
    "\U0001F1E6-\U0001F1FF"    # regional indicators (flags)
    "\uFE0F\u200D]+"
)

s3 = emoji_pattern.sub('', s2)
print(repr(s3))
```

```text
'omg!!! i was studying nlp at phitron .\nvisit  for more details.'
```

The emoji is removed, leaving a space before the period. Whitespace is then normalized (this also converts the line break to a single space):

```python
s3 = re.sub(r'\s+', ' ', s3).strip()
print(repr(s3))
```

```text
'omg!!! i was studying nlp at phitron . visit for more details.'
```

**Limitation.** The listed ranges cover many common emoji but not every one, and some ranges also contain a few non-emoji symbols. A maintained library such as `emoji` (installed with `pip install emoji`, used with `import emoji`) is more complete, but it is a third-party dependency, so this pipeline uses the plain regex approach.

### Stage 4: Punctuation removal

Input: the output of Stage 3.

```python
import string

s4 = s3.translate(str.maketrans('', '', string.punctuation))
print(repr(s4))
```

```text
'omg i was studying nlp at phitron  visit for more details'
```

The `!!!` and both periods are removed. Because the standalone `.` after `phitron ` was deleted, two spaces now sit between `phitron` and `visit`. This does not matter for the next step, since the tokenizer ignores extra whitespace.

### Stage 5: Tokenization

Input: the output of Stage 4.

```python
from nltk.tokenize import word_tokenize

tokens = word_tokenize(s4)
print(tokens)
```

```text
['omg', 'i', 'was', 'studying', 'nlp', 'at', 'phitron', 'visit', 'for', 'more', 'details']
```

### Stage 6: Stopword removal

Input: the tokens from Stage 5. The stopword set is defined explicitly (it is not a full library list) and removes `i`, `was`, `at`, `for`, and `more`. `nlp` is kept as a meaningful token.

```python
stop_words = {"i", "was", "at", "for", "more"}

filtered = [t for t in tokens if t not in stop_words]
print(filtered)
```

```text
['omg', 'studying', 'nlp', 'phitron', 'visit', 'details']
```

### Stage 7: Stemming

Input: the filtered tokens from Stage 6.

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stems = [stemmer.stem(t) for t in filtered]
print(stems)
```

```text
['omg', 'studi', 'nlp', 'phitron', 'visit', 'detail']
```

`studying` became `studi` and `details` became `detail`. The other tokens are unchanged.

### Stage 8: Lemmatization

Input: the stems from Stage 7 (the requested order puts lemmatization after stemming). The POS assumption is made explicit: we try verb (`v`) and the default noun (`n`).

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

lemmas_verb = [lemmatizer.lemmatize(w, pos="v") for w in stems]
lemmas_noun = [lemmatizer.lemmatize(w) for w in stems]

print(lemmas_verb)
print(lemmas_noun)
```

```text
['omg', 'studi', 'nlp', 'phitron', 'visit', 'detail']
['omg', 'studi', 'nlp', 'phitron', 'visit', 'detail']
```

Final output of the pipeline:

```text
['omg', 'studi', 'nlp', 'phitron', 'visit', 'detail']
```

Lemmatizing the stems changes nothing here, because the stems are already in their final shape and `studi` is not in WordNet, so it cannot be mapped to a lemma. This shows that the order matters. If we lemmatize the original filtered tokens instead (with verb POS assumed), we get a better result for `studying`:

```python
print([lemmatizer.lemmatize(t, pos="v") for t in filtered])
```

```text
['omg', 'study', 'nlp', 'phitron', 'visit', 'detail']
```

Here `studying` becomes the real word `study`. Assigning one POS to every token is a simplification: `details` happened to come out as `detail` even under the verb setting, but in general a single POS for all tokens will be wrong for some words. A real pipeline would use a POS tagger.

**Practical note.** Many real NLP pipelines use lemmatization instead of stemming, rather than applying both one after the other. Stemming and lemmatization are alternative ways to normalize word forms and are not interchangeable.

### Summary table

| Stage | Purpose | Example effect |
|---|---|---|
| 1. Lowercasing | Make word forms consistent | `OMG` → `omg`, `PHITRON` → `phitron` |
| 2. URL removal | Remove links that add noise | `https://phitron.io` removed |
| 3. Emoji removal | Remove non-text symbols and tidy spaces | `😊` removed, line break becomes a space |
| 4. Punctuation removal | Prevent punctuation sticking to words | `omg!!!` → `omg` |
| 5. Tokenization | Split text into word tokens | `'omg i was studying ...'` → list of 11 tokens |
| 6. Stopword removal | Drop very common words | `i`, `was`, `at`, `for`, `more` removed |
| 7. Stemming | Reduce words to approximate stems | `studying` → `studi`, `details` → `detail` |
| 8. Lemmatization | Map words to dictionary base forms | Stems unchanged here (`studi` is not a WordNet word) |
