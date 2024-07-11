# Evaluation Matrices : NLP

Evaluation metrics is used to optimize the model performance to identify the areas for improvement and make iterative adjustments to enhance the accuracy and precision of NLP systems.

#### ➤ F1 score

$$
\text{F1 Score}= 2\times\frac{Precision \times Recall}{Precision + Recall}
$$

 - Precision = TP / (TP+FP)
 - Recall = TP / (TP+FN)

- F1 score commonly used in classification problems, is also applicable to various NLP tasks like Named Entity Recognition, POS-tagging, etc. The F1 score is the harmonic mean of precision and recall, and thus balances the two and prevents extreme cases where one is favored over the other. It ranges from 0 to 1, where 1 signifies perfect precision and recall.
  
- Precision is the number of true positive results divided by the number of all positive results, including those not identified correctly. Recall, on the other hand, is the number of true positive results divided by the number of all samples that should have been identified as positive.

- In NLP, F1 score is often used in Named Entity Recognition, POS-tagging, and other classification tasks. The F1 score is ideal when you need to balance precision and recall, especially in cases where both false positives and false negatives are equally costly.


#### ➤ Perplexity

For a probability distribution p and a sequence of N words w1,w2,...wN

$$
Perplexity = \sqrt[N]{\frac{1}{p(w1, w2, .., wN)}}
$$

- Perplexity is a measure commonly used to assess how well a probability distribution predicts a sample. In the context of language models, it evaluates the uncertainty of a model in predicting the next word in a sequence.

- Perplexity serves as an inverse probability metric. A lower perplexity indicates that the model's predictions are closer to the actual outcomes, meaning the model is more confident (more accurate) in its predictions.

- Higher Perplexity (Human written): Suggests that the text is less predictable or more complex. In the context of language models, a higher perplexity might indicate that the model is less certain about its predictions or that the text has a more complex structure or vocabulary. 

- Lower Perplexity (AI generated): Indicates that the text is more predictable or simpler. For language models, a lower perplexity usually means the model is more confident in its predictions and the text may follow more common linguistic patterns.

#### ➤ BERTScore

- BERTScore is used to evaluate the quality of text. It leverages the contextual embeddings from BERT. This allows for a more nuanced comparison of text, as BERTScore can understand the context in which words are used.

- It computes the **cosine similarity** between the embeddings of words in the candidate text and the reference text, accounting for the deep semantic similarity. The calculation involves finding the best match for each word in the candidate text within the reference text and averaging these scores.

- BERTScore leverages contextual embeddings, offering a sophisticated method to assess semantic similarity between generated and reference texts.


#### ➤ BLEU (Bilingual Evaluation Understudy)

$$
BLEU = BP \times\exp(\sum_{i=1}^{n} w_i \times log(p_i))
$$

 - `BP`: brevity penalty (to penalize short sentences)
 - `w_i`: weights for each gram 
 - `p_i`: precision for each i-gram

- BLEU is predominantly used in machine translation. It quantifies the quality of the machine-generated text by comparing it with a set of reference translations. The crux of the BLEU score calculation is the precision of n-grams in the machine-translated text. However, to prevent the overestimation of precision due to shorter sentences, BLEU includes a brevity penalty factor. Note that BLEU mainly focuses on precision rather than recall.

- BLEU is effective in assessing the closeness of machine generated translations to a set of high quality reference translations. It's suitable when precision of translated text is a priority but it may not capture the fluency or grammatical correctness of the translation, as it focuses on the precision.

#### ➤ ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

- ROUGE is used for evaluating automatic summarization and machine translation. The key feature of ROUGE is its focus on recall, measuring how many of the reference n-grams are found in the system-generated summary. This makes it especially useful where coverage of key points is important. 

- ROUGE-N computes the overlap of n-grams between the system and reference summaries.

- ROUGE-L uses the longest common subsequence to account for sentence-level structure similarity.

- ROUGE-S includes skip-bigram plus unigram-based co-occurrence statistics. Skip-bigram is any pair of words in their sentence order.

#### ➤ METEOR (Metric for Evaluation of Translation with Explicit ORdering)

$$
\text{METEOR} = \frac{10 \bullet P \bullet R}{R + 9 \bullet P} − Penalty
$$

 - `P`: precision (proportion of matched words in the machine translation)
 - `R`: recall (proportion of matched words in the reference translation)
 - `Penalty`: for word order differences.

- METEOR is used for evaluating machine translation. Unlike BLEU, METEOR emphasizes both precision and recall, taking into account the number of matching words between the machine-generated text and reference translations. It's known for using synonyms and stemming to match words, allowing for a more flexible comparison.

- It calculates a score based on the harmonic mean of precision and recall, giving equal importance to both. It also includes a penalty for too many unmatched words, ensuring that translations are not just accurate but also coherent and fluent.
