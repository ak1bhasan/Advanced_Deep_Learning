# NLP Pipeline Questions and Solutions

An NLP pipeline is the sequence of steps used to turn raw human language into a working application, such as a spam filter or a sentiment classifier. The usual steps are collecting data, preparing the text, extracting features, training a model, evaluating it, deploying it, and then monitoring and updating it. These stages are connected because the output of one stage becomes the input of the next. For example, poor data collection leads to poor features, and poor features limit how well any model can perform, no matter how good the algorithm is. The order is also not fixed for every system. A modern transformer-based system, for instance, often skips manual feature engineering and uses a pretrained model's own tokenizer and representations, and teams often go back to earlier stages when evaluation shows a problem.

---

## 1. Data Acquisition

### 1. What do you understand by Data Acquisition?

Data acquisition is the process of finding, collecting, and storing the data that an NLP project needs. For NLP, this data is text (or speech that is later converted to text), often together with labels such as "spam" or "not spam". It is the first stage of the pipeline, and everything after it depends on what is collected here.

### 2. Why do we need to collect data?

Machine learning models learn patterns from examples, so without data there is nothing to learn from. Data is also needed to test whether a model works on examples it has not seen. Even rule-based NLP systems need sample text so that the rules can be checked against real language. The data should also match the problem. A model trained on movie reviews will not necessarily work well on medical notes, because the vocabulary and writing style are very different.

### 3. From where can we collect data for an NLP project?

Common sources include:

- **Public datasets:** Ready-made collections such as those on Kaggle, Hugging Face Datasets, or the UCI repository. These are convenient for learning and benchmarking.
- **APIs:** Some platforms provide programmatic access to their content. The terms of use and rate limits differ for each service.
- **Web pages:** Text can be collected by web scraping, but the site's terms of service, robots.txt rules, and copyright should be checked first.
- **Documents:** PDFs, Word files, emails, reports, and books. These often need extra work to extract clean text.
- **Surveys and forms:** Useful when we need text that answers a specific question, such as customer feedback.
- **User-generated text:** Reviews, comments, chat logs, and support tickets. This is often very relevant but may contain personal information, so privacy rules apply.

### 4. What makes a dataset good for solving a problem?

A good dataset is one that fits the problem, not just one that is large. The main factors are:

- **Relevance:** The text should come from the same kind of source and language as the text the model will see later.
- **Representativeness:** It should cover the variety of cases the system will face, such as different topics, writing styles, and classes.
- **Quantity:** There should be enough examples for the model to learn from. How much is enough depends on the task and model.
- **Label quality:** Labels should be accurate and consistent. Noisy or contradictory labels limit what the model can learn.
- **Completeness and duplication:** Missing values and repeated records should be handled, because duplicates can make evaluation scores look better than they really are.
- **Licensing and privacy:** We should be allowed to use the data, and personal information should be protected or removed.

A large dataset can still be unsuitable. For example, one million tweets are not very useful for building a legal document classifier. A smaller dataset of well-labeled legal text would likely be a better fit.

---

## 2. Text Preparation

### 5. What do you understand by Text Preparation?

Text preparation means getting raw text into a form that a model or an NLP tool can use properly. It covers both text cleaning and text preprocessing. Raw text from the real world is messy, with HTML tags, odd symbols, inconsistent spelling, and different formats, so some preparation is usually needed before feature extraction.

### 6. Why is text cleaning important?

Cleaning removes noise that does not help the task and could even confuse the model. If the same word appears as "Great", "great", and "great!!!", a simple model may treat them as different things. Noise such as leftover HTML or broken encoding characters can also add meaningless features. Cleaner input generally leads to more reliable features and results. However, cleaning should be done carefully, because removing too much can delete useful information. Exclamation marks or emojis, for example, may carry sentiment in some tasks.

### 7. What cleaning steps would you perform on raw text data?

The exact steps depend on the data and the task, but common ones are:

1. Remove unwanted HTML tags and markup.
2. Handle URLs, either by removing them or replacing them with a placeholder such as `URL`.
3. Fix encoding problems, such as broken characters caused by a wrong text encoding.
4. Remove irrelevant symbols or special characters.
5. Normalize repeated whitespace and line breaks.
6. Remove duplicate records.
7. Deal with missing values, either by dropping empty rows or by filling them in if that makes sense.

### 8. What is text preprocessing?

Text preprocessing is the set of steps that transform cleaned text into a more standard form for modeling, such as tokenization or lowercasing. The line between cleaning and preprocessing is not strict, and different books and workflows classify steps differently. A simple way to think about it is that cleaning removes errors and noise from the text, while preprocessing reshapes the text so that the model can work with it.

### 9. What preprocessing techniques have you learned so far?

- **Lowercasing:** Converts all text to lowercase so that "Apple" and "apple" are treated the same. This can hurt tasks where case matters, such as recognizing names.
- **Tokenization:** Splits text into smaller units such as words, sentences, or subwords.
- **Stop-word removal:** Removes very common words like "the" and "is". This can reduce noise for some models, but it can also remove useful words such as "not".
- **Stemming:** Cuts words down to a rough root using rules, for example "running" to "run". The result may not be a real word, such as "studies" becoming "studi".
- **Lemmatization:** Converts a word to its dictionary form (lemma), for example "better" to "good", usually with the help of a vocabulary and part of speech information.

Not every technique is needed for every task. A modern transformer model, for example, normally works best with its own tokenizer and does not usually need stemming or stop-word removal.

### 10. When do we need advanced preprocessing techniques?

Advanced techniques are useful when basic steps are not enough for the task or language. Some examples:

- **Language-specific processing:** Languages like Chinese and Thai have no spaces between words, and languages like Bangla or Arabic have rich morphology, so they need special tokenizers and tools.
- **Named Entity Recognition (NER):** Useful when we need to find names of people, places, or organizations, for example in information extraction.
- **Part-of-Speech (POS) tagging:** Helps with tasks that depend on grammar, and it also improves lemmatization.
- **Handling negation:** In sentiment analysis, "not good" means the opposite of "good", so the negation must be captured in some way.
- **Spelling correction:** Helpful for noisy text such as social media posts or chat messages.
- **Subword tokenization:** Methods such as Byte Pair Encoding or WordPiece break rare words into smaller pieces, which helps with unknown words and is common in transformer models.

These steps add complexity and processing time, so they should be used only when the task actually benefits from them.

---

## 3. Regular Expressions (Regex)

### 11. What is a Regular Expression (Regex)?

A regular expression is a compact way of describing a text pattern so that a program can search for, match, or replace text that follows that pattern. For example, a pattern can describe "any sequence of digits" or "a word that ends with ing" without listing every possible match.

### 12. Why do we use Regex in NLP?

Regex is a fast and simple tool for rule-based work with text. In NLP it is commonly used to extract structured pieces such as emails, dates, phone numbers, and URLs, and to clean text by removing tags, symbols, or extra spaces. It is also used in tokenization rules and for quickly searching a corpus. It works best when the target text follows a predictable format. It is less suitable for tasks that need an understanding of meaning, such as deciding whether a review is positive.

### 13. What is a regex pattern?

A regex pattern is the string of characters and special symbols that defines what we want to match. Some characters match themselves (the pattern `cat` matches the text "cat"), and some have special meanings (the pattern `\d+` matches one or more digits). In Python, patterns are usually written as raw strings, like `r"\d+"`, so that backslashes are not treated as escape characters by Python itself.

### 14. What is the purpose of symbols like `.`, `*`, `+`, `?`, `^`, and `$`?

| Symbol | Meaning | Example |
|--------|---------|---------|
| `.` | Matches any single character (except a newline by default) | `c.t` matches "cat", "cot", "c9t" |
| `*` | Zero or more of the previous item | `ab*` matches "a", "ab", "abbb" |
| `+` | One or more of the previous item | `ab+` matches "ab", "abbb", but not "a" |
| `?` | Zero or one of the previous item (makes it optional) | `colou?r` matches "color" and "colour" |
| `^` | Start of the string (or line, in multiline mode) | `^Hello` matches "Hello" only at the start |
| `$` | End of the string (or line, in multiline mode) | `end$` matches "end" only at the end |

The difference between `*`, `+`, and `?` is how many times the previous item may appear: `*` allows any number including none, `+` requires at least one, and `?` allows at most one.

A few other building blocks are also useful:

- **Character classes** such as `[a-z]` match one character from a set, and shortcuts like `\d` (digit), `\w` (word character), and `\s` (whitespace) exist.
- **Repetition counts** like `{2,4}` mean "between 2 and 4 times".
- **Boundaries** such as `\b` match the edge of a word.
- **Escaping** with a backslash is needed when we want to match a special symbol literally. For example, `\.` matches an actual dot, while `.` matches any character.

### 15. How can Regex help us find email addresses, phone numbers, or URLs?

Emails, phone numbers, and URLs follow fairly regular formats, so patterns can find them in text. Here is a practical example for emails:

```python
import re

text = "Contact us at student@example.com or admin@my-site.org"

emails = re.findall(
    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
    text
)

print(emails)
```

The pattern reads as follows: `[\w.+-]+` matches the part before the `@` (letters, digits, underscores, dots, plus signs, or hyphens), `@` matches the at sign, `[\w.-]+` matches the domain name, `\.` matches the dot before the ending, and `[A-Za-z]{2,}` matches an ending such as "com" or "org". The `\b` at both ends keeps the match at word edges.

Similar ideas work for URLs and phone numbers:

```python
import re

text = "Visit https://www.example.com/page or call +880 1712-345678."

urls = re.findall(r"https?://\S+", text)
phones = re.findall(r"\+?\d[\d\s().-]{7,}\d", text)

print(urls)
print(phones)
```

Here `https?://` matches "http://" or "https://" (the `?` makes the "s" optional), and `\S+` matches the following non-space characters. The phone pattern looks for an optional plus sign, a digit, a run of digits and common separators, and a final digit.

These patterns are practical for common formats, but they are not complete validators. Real email addresses and URLs follow complicated rules, and a simple pattern can accept invalid ones or miss unusual valid ones. For instance, the URL pattern above would also grab a trailing full stop if the URL ends a sentence, and phone number formats differ a lot from country to country. If strict validation is needed, a dedicated library or an actual check (such as sending a confirmation email) is a better choice.

### 16. How can Regex help in text cleaning?

Regex can quickly remove or replace noise. The example below strips HTML tags, removes unwanted symbols, and fixes repeated whitespace:

```python
import re

raw = "<p>Great   product!!!   Visit   https://shop.example.com   #deal</p>"

text = re.sub(r"<[^>]+>", " ", raw)           # remove HTML tags
text = re.sub(r"https?://\S+", " ", text)     # remove URLs
text = re.sub(r"[^A-Za-z0-9\s]", "", text)    # remove symbols
text = re.sub(r"\s+", " ", text).strip()      # collapse extra whitespace

print(text)
```

The output is `Great product deal`. The pattern `<[^>]+>` matches anything between angle brackets, `[^A-Za-z0-9\s]` matches any character that is not a letter, digit, or whitespace, and `\s+` matches one or more whitespace characters, which are replaced by a single space. Whether these steps are appropriate depends on the task. For sentiment analysis, for example, removing "!!!" might lose a useful signal.

---

## 4. Feature Engineering

### 17. What do you understand by Feature Engineering?

Feature engineering is the process of turning text into numerical features that a model can learn from. In traditional NLP this means choosing and building representations such as word counts or TF-IDF values, and sometimes adding hand-made features like text length or the number of exclamation marks. In deep learning, especially with pretrained transformers, much of this work is done automatically by the model, though choices such as tokenization still matter.

### 18. Why can't machine learning models work directly with raw text?

Most machine learning algorithms are built on mathematical operations such as multiplying weights, computing distances, and taking gradients, and these need numbers. A string like "good movie" has no numerical meaning on its own. Text therefore has to be converted into vectors, so that the model can compare documents and find patterns. The way the text is converted decides what information the model can see.

### 19. When would you choose one text representation technique over another?

There is no representation that is best for every case, and the choice depends on the task, the data size, the available computing power, and how much interpretability is needed. The basic ideas are:

- **Bag of Words (BoW):** Represents a document by how many times each word appears, ignoring word order. Word frequency is the simplest form of this.
- **TF-IDF:** Weights each word by how often it appears in a document (term frequency) and reduces the weight of words that appear in many documents (inverse document frequency), so that distinctive words get more importance.
- **Word embeddings:** Represent each word as a dense vector learned from large text collections, so that words used in similar contexts have similar vectors (for example Word2Vec or GloVe).
- **Contextual embeddings:** Produced by transformer models such as BERT, where the vector of a word depends on the sentence around it, so "bank" in "river bank" differs from "bank" in "bank loan".

**Sparse vs dense representations.** BoW and TF-IDF vectors are sparse, meaning they are very long (one position per vocabulary word) and mostly zeros. They are easy to interpret because each position corresponds to a word. Embeddings are dense, meaning they are short (for example a few hundred values) with mostly non-zero numbers, and they can capture similarity between words, but individual values are hard to interpret.

| Representation | Basic idea | Strengths | Limitations | Suitable use case |
|----------------|-----------|-----------|-------------|-------------------|
| Bag of Words | Count of each word in a document | Simple, fast, easy to interpret | Ignores word order and meaning; large sparse vectors | Quick baseline for small text classification tasks |
| TF-IDF | Word counts weighted by how rare a word is across documents | Highlights important words; strong with linear models; interpretable | Still ignores word order and meaning; sparse and high dimensional | Document classification or search with limited data and resources |
| Word Embeddings | Dense vectors learned from word co-occurrence | Capture similarity between words; compact | One vector per word regardless of context; cannot handle word senses well | Tasks where word similarity helps and resources are limited |
| Contextual Embeddings (transformers) | Word vectors that depend on the surrounding sentence | Capture context and word sense; strong performance on many tasks | Need more computing power and memory; harder to interpret | Tasks needing deeper language understanding, such as question answering or nuanced sentiment |

As a practical example, I would choose TF-IDF for a small text classification task with limited computing power, because it is quick, works well with linear models, and is easy to explain. I would consider a pretrained transformer when the task depends on context or subtle meaning and enough computing resources are available.

---

## 5. Modelling

### 20. What do you understand by Modelling?

Modelling is the stage where an algorithm learns patterns from the prepared data so that it can make predictions on new text. In an NLP pipeline this means choosing a model type (such as a classifier), training it on the features, and tuning its settings (hyperparameters) to get good performance.

### 21. How do you choose a machine learning algorithm for a problem?

The choice usually depends on several things:

- **Problem type:** Classification, regression, clustering, sequence labeling, and text generation each need different kinds of models.
- **Dataset:** Size, number of classes, label quality, and whether the text is short or long. Complex models usually need more data.
- **Computational resources:** Training and prediction time and memory limits.
- **Interpretability:** Some applications need explanations for predictions, which favors simpler models.
- **Performance requirements:** How accurate the system must be and how fast it must respond.

In practice, it is common to start with a simple baseline and move to a more complex model only if the simple one is not good enough. Comparing candidates on validation data is better than guessing.

### 22. What algorithm would you use for a text classification task and why?

Common choices for text classification are:

- **Naive Bayes:** Very fast and works well with word counts or TF-IDF, even on small datasets.
- **Logistic Regression:** A reliable linear model that is easy to interpret, because each word gets a weight.
- **Linear Support Vector Machine (SVM):** Often performs well on high-dimensional sparse features.
- **Neural networks or pretrained transformer models:** Can give better results when the task needs context and there is enough data and computing power.

Naive Bayes, Logistic Regression, and linear SVMs are strong baselines for TF-IDF features because text data is high dimensional and often close to linearly separable, and these models handle that setting well while being quick to train. For a typical small project, I would start with TF-IDF and Logistic Regression or a linear SVM, since it is easy to train and explain. If the results were not good enough, I would try fine-tuning a pretrained transformer, which usually needs more data, time, and hardware. I would not say one algorithm is always the best, and the final choice should be based on validation results.

### 23. What is the difference between training and testing a model?

Training is the phase where the model learns from examples by adjusting its parameters. Testing is the phase where the trained model is evaluated on examples it has never seen to estimate how it will perform on new data.

In practice, the data is usually split into three parts:

- **Training set:** Used to fit the model.
- **Validation set:** Used to compare models and tune hyperparameters.
- **Test set:** Kept aside until the end and used for a final evaluation on held-out examples.

The test data should not influence training or model selection. If information from the test set leaks into the training process (called data leakage), the test score becomes too optimistic and no longer reflects real performance. A common mistake is fitting a TF-IDF vectorizer on the whole dataset before splitting, which lets test-set vocabulary statistics leak into training. The vectorizer should be fitted on the training data only.

---

## 6. Evaluation

**Part A: Intrinsic Evaluation**

### 24. What do you understand by model evaluation?

Model evaluation is the process of measuring how well a model performs, using suitable metrics and data that was not used for training. Intrinsic evaluation looks at the model itself, usually by comparing its predictions to known correct labels on a test set.

### 25. Why do we evaluate a model?

We evaluate a model to find out whether it actually works, to compare different models or settings, and to detect problems such as overfitting (doing well on training data but poorly on new data). Evaluation also helps decide whether the model is good enough to deploy and which kinds of mistakes it tends to make.

For the metrics below, consider a binary spam classifier where "spam" is the positive class:

- **TP (true positive):** a spam email correctly predicted as spam.
- **TN (true negative):** a normal email correctly predicted as normal.
- **FP (false positive):** a normal email wrongly predicted as spam.
- **FN (false negative):** a spam email wrongly predicted as normal.

A confusion matrix is a table that shows these four counts together.

### 26. What is Accuracy?

Accuracy is the fraction of all predictions that are correct.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

It is easy to understand, but it can be misleading when the classes are imbalanced.

### 27. What is Precision?

Precision tells us, out of everything the model predicted as positive, how many were actually positive.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

In the spam example, high precision means that when the model says "spam", it is usually right, so few normal emails are lost. Precision matters when false alarms are costly.

### 28. What is Recall?

Recall tells us, out of all the actual positive examples, how many the model found.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

High recall means the model catches most of the spam. Recall matters when missing a positive case is costly, such as detecting harmful content.

### 29. What is F1-Score?

The F1-score combines precision and recall into one number using their harmonic mean.

$$
F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

It is high only when both precision and recall are reasonably high. It is useful when we want a balance between the two, though it does not replace looking at them separately, since the right trade-off depends on the application.

For multiclass problems, the metrics can be averaged in different ways, and these give different perspectives:

- **Macro average:** Computes the metric for each class and takes the simple average, so every class counts equally, including small ones.
- **Micro average:** Combines the counts of all classes first and then computes the metric, so larger classes have more influence.
- **Weighted average:** Takes the average of per-class metrics weighted by how many examples each class has.

### 30. When is Accuracy not a good metric?

Accuracy is not reliable when the classes are imbalanced. As a hypothetical example, suppose only 10 out of 1000 emails are spam. A model that predicts "not spam" for every email would reach 99% accuracy but would never catch any spam, so its recall for spam would be 0. In such cases, precision, recall, F1-score, and the confusion matrix give a more honest picture. Other task-appropriate metrics can also be used, such as the area under the precision-recall curve. Accuracy can also hide different costs of errors, since a false negative and a false positive are not always equally serious.

**Part B: Extrinsic Evaluation**

### 31. What do you understand by Extrinsic Evaluation?

Extrinsic evaluation judges a model by how well it works inside the larger application or workflow where it is used, rather than only by metrics on a test set. Intrinsic evaluation asks "how accurate is the model on this dataset?", while extrinsic evaluation asks "does this model actually make the real system better?"

### 32. How can we determine whether a model is useful in a real-world application?

We look at outcomes in the real application. Some examples:

- **Spam detector:** Do users see less spam without losing important emails? Do users report fewer complaints?
- **Sentiment classifier:** Does it help the business identify unhappy customers early enough to act?
- **Customer support system:** Does it reduce response time or the number of tickets that need a human, and are customers satisfied?
- **Information extraction application:** Does the extracted information make the downstream task (such as filling a database) faster and more accurate?

Ways to check this include A/B testing, pilot launches with a small group of users, user feedback, and tracking business or workflow measures. Latency and cost should also be checked, since a model that is too slow or too expensive may not be practical.

### 33. Can a model have good accuracy but still fail in practice? Why?

Yes. Good test accuracy only shows that the model performed well on that particular test data. It can still fail in practice for several reasons:

- **Distribution changes:** Real users may write differently from the data the model was trained on, for example with new slang, topics, or languages.
- **Latency:** If predictions are too slow, the user experience suffers.
- **Poor user experience:** Even a small number of visible, embarrassing mistakes can reduce trust.
- **Unexpected inputs:** Very long text, typos, mixed languages, or deliberately tricky input may not appear in the test set.
- **Fairness issues:** The model may perform well overall but poorly for certain groups or dialects.
- **Integration problems:** Bugs, formatting differences, or mismatched preprocessing between training and production can cause errors.
- **Misleading metric:** As in the imbalanced example, high accuracy may hide the fact that the model fails on the cases that matter.

---

## 7. Deployment

### 34. What do you understand by Model Deployment?

Model deployment is the process of putting a trained model into a real environment so that it can receive new inputs and return predictions for actual users or other systems. It involves more than saving the model file. It also needs the preprocessing code, a way to call the model, and the infrastructure to run it.

### 35. Why do we deploy a model?

A trained model only creates value when it is used. Deployment turns it from an experiment in a notebook into a usable system, such as a spam filter that works on incoming emails or a classifier that sorts support tickets automatically. A model that is never integrated into a workflow cannot help real users.

### 36. How can users interact with a deployed model?

Common options are:

- **REST API:** The model runs on a server, and other programs send text and receive predictions, usually in JSON format.
- **Web application:** Users type or upload text in a browser, and the page shows the result.
- **Mobile application:** The app either calls an API or runs a smaller model directly on the device.
- **Batch processing:** The model processes a large collection of text at scheduled times, for example classifying all of yesterday's reviews overnight.

As a simple example, a sentiment classifier in a web application could work like this. A user types a review into a text box and clicks a button. The web page sends the text to a backend API. The backend applies the same preprocessing used during training, converts the text into features, runs the model, and returns a label such as "positive" with a confidence score. The page then displays the result to the user.

### 37. What challenges might occur after deployment?

- **Latency:** Predictions must be fast enough for the use case, which can be hard with large models.
- **Memory usage:** Large models need a lot of memory and may not fit on small servers or devices.
- **Scalability:** The system must handle many users at once, especially during peak times.
- **Security:** The system must be protected from misuse, and malicious or unusual inputs must be handled safely.
- **Privacy:** User text may contain personal information, so storage and logging must follow privacy rules.
- **Reliability:** The system should keep working despite failures and should handle errors gracefully.
- **Inference cost:** Running large models for every request can be expensive.
- **Consistency:** The preprocessing used in production must match what was used in training.

---

## 8. Monitoring and Model Update

**Part A: Monitoring**

### 38. What do you understand by Model Monitoring?

Model monitoring means continuously tracking a deployed model and its surrounding system to check that it still works as expected. It looks at both how healthy the system is technically and how good the predictions are.

### 39. Why is monitoring important after deployment?

A model that worked well at launch can slowly get worse, because real-world language and user behavior change over time. Monitoring helps detect problems early so that they can be fixed before they affect many users. Measuring real-world performance often needs production labels or actual outcomes, such as user feedback, manual review of a sample, or later confirmation of what the correct answer was. These are often not available immediately, so teams may need to rely on indirect signals in the meantime.

### 40. What problems can occur if we do not monitor a model?

Without monitoring, the model may keep making worse predictions without anyone noticing. For example, a spam filter may slowly let more spam through as spammers change their wording, or a sentiment model may misread new slang. Technical problems such as slow responses, rising error rates, or high resource use may also go unnoticed until users complain or the system fails. This can lead to lost trust, wrong decisions, and wasted cost.

### 41. What would you monitor in a deployed NLP system?

It helps to separate two kinds of monitoring.

**Technical health:**

- Latency (how long predictions take)
- Error rates and failed requests
- Resource consumption (CPU, GPU, memory)
- Throughput and uptime

**Predictive quality:**

- **Input data:** Changes in text length, vocabulary, language, or topics compared with the training data.
- **Output predictions:** Shifts in the proportion of each predicted class, or in prediction confidence.
- **Performance on labeled samples:** Accuracy, precision, recall, or F1 on production data that has been labeled, using human review or real outcomes.
- **Fairness:** Whether performance differs across user groups.

A system can be technically healthy (fast and with no errors) and still be giving poor predictions, so both types of monitoring are needed.

**Part B: Model Update**

### 42. Why do models need updates?

Models are trained on a snapshot of data from the past, while language, users, and tasks keep changing. New words, topics, and writing styles appear, user behavior changes, and the business goal itself may change. Updating the model helps it stay accurate and useful. New labeled data can also be used to improve it.

### 43. What is model drift?

Model drift is the decline in a model's performance over time because the world has changed compared with when the model was trained. There are two main kinds:

- **Data drift:** The distribution of the input data changes, while the relationship between inputs and correct outputs stays the same. For example, customers start writing reviews about a new product category with vocabulary the model has not seen.
- **Concept drift:** The relationship between the input and the correct output changes. For example, the meaning of a word shifts, or what counts as "spam" changes as spammers adapt, so the same text should now get a different label.

Data drift can sometimes be detected without labels by comparing input statistics, but concept drift usually needs labels or outcome data to confirm.

### 44. When should a model be retrained?

Retraining should be based on evidence and suitable triggers, not on the assumption that every model must be retrained at a fixed interval. Situations that may justify it include:

- New labeled data is available that could improve the model.
- User behavior or language has changed noticeably.
- Monitoring shows that performance has dropped below an acceptable level.
- The task has changed, for example new categories are added.

A regular schedule can still be a reasonable choice for some systems, but it should be checked against evidence, because retraining without need wastes resources and can introduce new problems.

### 45. How would you decide whether a new model is better than the current one?

A new model should not automatically replace the current one just because it scores higher on one metric. I would compare them using several checks:

- **Appropriate evaluation data:** Use a recent, representative test set that neither model was trained on, and ideally include recent production-like examples.
- **Multiple relevant metrics:** Look at precision, recall, F1, and per-class results, not only accuracy, and judge them against what matters for the application.
- **Robustness:** Check performance on difficult cases, noisy text, and unusual inputs.
- **Latency and resource requirements:** A slightly more accurate model that is much slower or more expensive may not be worth it.
- **Fairness:** Where applicable, check that performance across different groups has not become worse.
- **Controlled deployment:** Release the new model gradually, for example to a small share of traffic or in a shadow mode, and compare it with the current model in real use through methods like A/B testing.
- **Rollback plan:** Keep the ability to go back to the previous model quickly if problems appear.

Replacing the model only makes sense if the new one is better overall for the real application and not merely on a single number.
