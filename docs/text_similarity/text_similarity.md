# Text Similarity

Text similarity is useful and one of the active research and application topics in Natural Language Processing. It measures how much the meaning or content of two pieces of text are the same. There are many ways to measure text similarity. 

Wait!

First let's make things clear.

There are many ways and methodology includes algorithms, embedding techniques, matrices and pre-trained models. 

- Embeddings: word2vec, TF-IDF
- Matrices: Cosinse similarity, Jaccard similarity, Euclidean Distance, Levenshtein Distance
- Lexical similarity: for clustering and keyword matching
- Semantic simialrity: for knowledge base, string and statical based
- Model: BERT, RoBERT, ALBERT, FastText

!!! note

    Text similarity is the process to calculate how two words/phrases/documents are close to each other. Semantic similarity is about the meaning closeness while lexical similarity is about the closeness of the word set.


**Example**

    It might not rain today
    It might not work today

It seems very similar according to the lexical similarity, those two phrases are very close and almost identical because they have the same word set('It', 'might', 'not', 'today'). But semantically, they are completely different because they have different meanings ('rain', 'work') despite the similarity of the word set.  

### ➤ Matrices

#### 1. Cosinse similarity

Cosine simiarity is widely used to find similarity between two texts based on the angle between their word vectors. It measures the similarity between two non-zero vectors of an inner product space in general. It is often used to measure the similarity between two documents represented as vectors of word frequencies. 

To find the similarity of documents, a vector representation of each document is constructed where each dimension of the vector corresponds to a word in the document, and the value of the dimension represents the frequency of that word in the document. Followed by then normalization process to have a unit length. 


$$
similarity(A, B) = cos (\emptyset )= \frac{A\bullet B}{\parallel A \parallel \times \parallel B\parallel }
$$


The cosine similarity is calculated as the dot product of the two vectors divided by the product of their lengths. It measures the cosine of the angle between two embeddings and determines whether they are pointing in the same direction or not. 

- When the embeddings are pointing in the same direction the angle between them is zero so their cosine similarity is **1** indicates **identical documents**. 
- When the embeddings are perpendicular to each other the angle between them is 90 degrees and the cosine similarity is **0** indicates **no similarity**.
- When the angle between them is 180 degrees the cosine similarity is **-1** indicates **completely dissimilar**.

Cosine similarity is widely used in natural language processing and information retrieval, particularly in document clustering, classification, and recommendation systems.

**Example**

```py
import numpy as np 

A = np.array([5, 3, 4])
B = np.array([4, 2, 4])

dot_product = np.dot(A, B)
magnitude_A = np.linalg.norm(A)
magnitude_B = np.linalg.norm(B)

cosine_similarity = dot_product / (magnitude_A * magnitude_B)
print(f"{cosine_similarity=}")
```

#### 2. Jaccard similarity

The Jaccard index or the Jaccard similarity, measures the similarity between two sets. It is the intersection of two sentences/texts between which the similarity is being calculated divided by the union of those two which refers to the number of common words over a total number of words. In other words, it is the proportion of common elements between two sets.

$$
J(A,B) = \frac{|A \cap B|}{|A \cup B|}
$$

The Jaccard Similarity score ranges from 0 to 1.
- **1** represents **most similar** 
- **0** represents **least similar**.

The Jaccard index is particularly useful when the presence or absence of elements in the sets is more important than their frequency or order. For example, it can be used to compare the similarity of two documents by considering the sets of words that appear in each document.

It is widely used in applications such as data mining, information retrieval and pattern recognition. It is very useful when dealing with sparse or high-dimensional data, where the presence or absence of features is more important than their actual values.

**Example**

```py
text1 = {"It", "might", "not", "rain", "today" }
text2 = {"It", "might", "not", "work", "today"}
intersection = len(text1.intersection(text2))
union = len(text1.union(text2))

jaccard_similarity = intersection / union
print(f"{jaccard_similarity=}")
```        
#### 3. Euclidean Distance

The Euclidean Distance formula is the most common formula to calculate distance between two points/coordinates in euclidean space. It is calculated as the square root of the sum of the squares of the differences between the corresponding coordinates of the two points.

$$
distance = \sqrt{\sum_{i=1}^{n}({x_i}-{y_i})^2}
$$

The Euclidean distance ranges from 0 to infinity.
- **0** indicates **identical vectors** 
- **larger values** indicate greater **dissimilarity** between the vectors.

In the context of document similarity, the Euclidean distance can be used to compare the frequency of words in two documents represented as vectors of word frequencies. The Euclidean distance can be extended to spaces of any dimension. It is commonly used in machine learning and data analysis to measure the similarity between two vectors in a high-dimensional space.

Euclidean distance is widely used in various applications such as clustering, classification, and anomaly detection. It is particularly useful when dealing with continuous variables or data that can be represented as vectors in a high-dimensional space.

Euclidean Distance takes a bit more time and computation power than other two.

**Example**

```py
point1 = np.array((1, 2, 3))
point2 = np.array((1, 1, 1))

sum_sq = np.sum(np.square(point1 - point2))
euclidean_distance = np.sqrt(sum_sq)
print(f"{euclidean_distance=}")
```

#### 4. Levenshtein distance

Levenshtein distance or also called Edit distance, measures the difference between two strings. It is the minimum number of single character insertions, deletions, or substitutions required to transform one string into another.

For example, the Levenshtein distance between “kitten” and “sitting” is 3, since three single character edits are required to transform “kitten” into “sitting”: substitute “s” for “k”, substitute “i” for “e”, and insert “g” at the end.

Levenshtein distance is used in various applications such as spell correction, string matching, and DNA analysis.

**Example**

```py
A = "cherry"
B = "berry"
def cal_levenshtein_distance(A, B):
    N, M = len(A), len(B)
    # Create an array of size NxM
    dp = [[0 for i in range(M + 1)] for j in range(N + 1)]

    # Base Case: When N = 0
    for j in range(M + 1):
        dp[0][j] = j
    # Base Case: When M = 0
    for i in range(N + 1):
        dp[i][0] = i
    # Transitions
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j], # Insertion
                    dp[i][j-1], # Deletion
                    dp[i-1][j-1] # Replacement
                )
    return dp[N][M]
```


### ➤ Embeddings 

#### 1. word2vec

Word2Vec represents the words as high-dimensional vectors so that we get semantically similar words close to each other in the vector space. There are two main architectures for Word2Vec.

##### Continuous Bag of Words

The CBOW model is a supervised learning neural network model that predicts the center word from the corpus words. It takes one hot encoding of the words as input and the output is the main word that can possibly add some sense to the neighboring words. The objective is to predict the target word based on the context of surrounding words.

| Input (Context)	| Output (Target) |
|:-----------------:|:---------------:|
| (I, to)	    |   like |
| (like, drink)	| to |
| (to, coffee)	| drink |
| (drink, the)	| coffee |
| (coffee, whole)	| the |
| (the, day)	| whole |
| (whole)	    | day |

##### Skip-gram

The Skip-Gram model works in just the opposite way of the CBOW model. It takes the target word as input and predicts the neighbouring words.

| Input     | 	Output |
|:---------:|:--------:|
| like	    | (I, to) |
| to	    | (like, drink) |
| drink	    | (to, coffee) |
| coffee    | (drink, the) |
| the	    | (coffee, whole) |
| whole	    | (the, day) |
| day       | 	(whole) |

#### 2. TF-IDF

TF-IDF is essentially a number that tells you how unique a word or term is across multiple pieces of text. Those numbers are then combined to determine how unique each bit of text is from each other.

- **Term frequency**

 The number of times a term occurs in a document is called its term frequency.
 It measures how often a word/term appears in a bit of text (document). This is computed as the ratio between the number of times the word/term is in the document and the number of words in the document. The weight of a term that occurs in a document is proportional to the term frequency.

 Say there is a query "the red car" to rank among documents. So in order to find the relevant documents we use the combination *['the', 'red', 'car']* that fetches all the documents having these combination but still there are too many documents with this match. Therefore we count the number of times each term occurs in each document (called term frequency).

- **Inverse document frequency** 

 There might be the problem with term frequency for a query like "at the station the" because the term *"the"* is so common, term frequency will tend to incorrectly emphasize documents which happen to use the word "the" more frequently, without giving enough weight to the more meaningful terms "station". The term "the" is very common so it is not a good keyword to distinguish relevant and non-relevant documents and terms, unlike the least common words "station". Hence an inverse document frequency factor is incorporated which diminishes the weight of terms that occur very frequently in the document set and increases the weight of terms that occur rarely.

 > The specificity of a term can be quantified as an inverse function of the number of documents in which it occurs.

The formula for inverse document frequency is a bit more complicated and many software implementations use their own tweaks. That being said, IDF ratio is just the ratio between the number of documents in your corpus and the number of documents with the word you’re evaluating.

The TF and IDF parts are multiplied together to get the actual TF-IDF value. This gives us a metric for how much each word makes a document in the corpus unique.


### ➤ Lexical similarity

Lexical similarity is the similarity between words or set of words in both senteces or text.

> the cat ate the mouse \
> the mouse ate the cat

Both sounds similar according to the word sets ['the', 'cat', 'ate', 'mouse']. The similarity score will be very high as the words in both the sentences are almost same.

It is used for Clustering to group similar texts together based on their similarity and Keywords matching for selecting texts based on given keywords like finding resumes with similar skills set keywords.

### ➤ Semantic simialrity

Semantic Similarity refers to the degree of similarity between the words. The focus is on the structure and lexical resemblance of words and phrases. Semantic similarity delves into the understanding and meaning of the content. It measures how close or how different the two pieces of word or text are in terms of their meaning and context.

**Types of Semantic similarity:**

- Knowledge based: similarity between the concept of corpus.
- Statistical based: similarity based on learning features vectors from the corpus.
String based: combines the above two approaches to find the similarity between non-zero vectors.

Semantic similarity is often used in nlp for question answer, recommandation system, paraphrase identification.

## 👩🏻‍💻 Implementation

#### with nltk

```py
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def text_similarity(text1, text2):
    # Tokenize and lemmatize the texts
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    lemmatizer = WordNetLemmatizer()
    tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
    tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    # Remove stopwords
    stop_words = stopwords.words('english')
    tokens1 = [token for token in tokens1 if token not in stop_words]
    tokens2 = [token for token in tokens2 if token not in stop_words]

    # Create the TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vector1 = vectorizer.fit_transform(tokens1)
    vector2 = vectorizer.transform(tokens2)

    # Calculate the cosine similarity
    similarity = cosine_similarity(vector1, vector2)

    return similarity
```

#### with sklearn

```py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def text_similarity(text1, text2):
    # Convert the texts into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])

    # Calculate the cosine similarity
    similarity = cosine_similarity(vectors)
    return similarity
```

#### with bert

```py
from transformers import BertTokenizer, BertModel
from torch import tensor, no_grad
from sklearn.metrics.pairwise import cosine_similarity

def text_similarity(text1, text2):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize the sentences
    tokens1 = tokenizer.tokenize(text1)
    tokens2 = tokenizer.tokenize(text2)

    # Add [CLS] and [SEP] tokens for separating two texts
    tokens = ['[CLS]'] + tokens1 + ['[SEP]'] + tokens2 + ['[SEP]']
    # print(f"{tokens=}")

    # Convert tokens to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print(f"{input_ids=}")

    # Convert tokens to input IDs
    input_ids1 = tensor(tokenizer.convert_tokens_to_ids(tokens1)).unsqueeze(0)  # batchsize 1
    input_ids2 = tensor(tokenizer.convert_tokens_to_ids(tokens2)).unsqueeze(0)  # batchsize 1

    # embeddings
    with no_grad():
        outputs1 = model(input_ids1)
        outputs2 = model(input_ids2)
        embeddings1 = outputs1.last_hidden_state[:, 0, :]  # [CLS] token
        embeddings2 = outputs2.last_hidden_state[:, 0, :]  # [CLS] token

    # Calculate similarity
    similarity_score = cosine_similarity(embeddings1, embeddings2)
    return similarity_score
```

#### with torch

```py
import torch
from torch.nn import CosineSimilarity

def text_similarity(text1, text2):
    # Custom encoding
    encoding_dict1 = {char: i for i, char in enumerate(set(''.join([text1])))}
    encoding_dict2 = {char: i for i, char in enumerate(set(''.join([text2])))}

    tensor1 = torch.tensor([[encoding_dict1[char] for char in string] for string in [text1]], dtype=float)
    tensor2 = torch.tensor([[encoding_dict2[char] for char in string] for string in [text2]], dtype=float)

    # Size of tensors must be match
    if tensor1.shape == tensor2.shape:
        # Calculate the cosine similarity
        cosine_similarity = CosineSimilarity(dim=0)
        similarity = cosine_similarity(tensor1, tensor2)
    else:
        raise RuntimeError(f"{tensor1.shape} and {tensor2.shape} are mismatched")

    return similarity
```