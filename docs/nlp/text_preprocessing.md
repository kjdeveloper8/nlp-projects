# Text Preprocessing

Text preprocessing is an essential first step in natural language processing (nlp) that involves cleaning and transforming unstructured text data to prepare it for analysis. It includes tokenization, stemming, lemmatization, stop-word removal.

Text preprocessing steps for a problem depend mainly on the domain and the problem itself so it can be vary depend on requirements.

Text data often contains punctuation, emojis, special characters, and fuzzy symbols. These noisy data must be removed for model building. Preprocessing helps remove these elements, making the text cleaner and easier to analyze.

## Preprocessing

#### Convert to lowercase

Text lowercase helps to reduce the size of the vocabulary and also complexity of our text data.

```py
text = "Do TEXT processing in Natural Language Processing"
text.lower()
```

#### Remove stop words

Stopwords like *do, an, him, it, did, so, wasn't, had, once* are words that do not contribute to the meaning of a sentence. Hence, they can safely be removed without causing any change in the meaning of the sentence. The NLTK library has a set of stopwords and we can use these to remove stopwords from our text and return a list of word tokens. But it is not necessary to use the provided list as stopwords as they should be chosen differently based on the requirements.


```py

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# add custom stop words to remove
stop_words.add('hello')

```

#### Tokenization

Split the text into smaller units.

```py
from nltk.tokenize import word_tokenize
word_tokenize(text)
```

#### Remove matched expression

Remove numbers, punctuations or any other particular occurance using regex expression.

```py
import re
expression = '\d+' # digits
remove_digit = re.sub(expression, '', text)
```

#### Stemming

Stemming is the process of getting the root form of a word. Stem or root is the part to which inflectional affixes (-ed, -ize, -de, -s, etc.) are added. The stem of a word is created by removing the prefix or suffix of a word. So, stemming a word may not result in actual words.

- books      --->    book
- looked     --->    look
- denied     --->    deni
- science    --->    scienc


```py
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
print([porter_stemmer.stem(word) for word in text.split()])
```        

#### Lemmatization

Like stemming, lemmatization also converts a word to its root form. The only difference is that lemmatization ensures that the root word belongs to the language. We will get valid words if we use lemmatization. In NLTK, we use the WordNetLemmatizer to get the lemmas of words. We also need to provide a context for the lemmatization. So, we add the part-of-speech as a parameter. 

- goose    -->   goose
- geese    -->   goose
- science  -->   science

```py
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
print([wordnet_lemmatizer.lemmatize(word) for word in text.split()])
```        