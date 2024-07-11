# Named Entity Recognition

Named Entity Recognition (NER) is a technique in nlp that focuses on identifying and classifying entities from text. NER is the component of information extraction that aims to identify and categorize named entities within unstructured text. NER involves the identification of key information in the text and classification into a set of predefined categories.

**An entity** is the thing that is consistently talked about or refer to in the text, such as person names, organizations, locations, time expressions, quantities, percentages and more predefined categories.

    Text: "Argentine captain Lionel Messi won Golden Ball at FIFA world cup 2022"

    - Argentine     LOCATION
    - Lionel Messi  PERSON
    - Golden Ball   OTHER
    - 2022          YEAR
  
NER enabling machines to understand and categorize entities in a meaningful manner for various applications like question answering, information retrival, knowledge graph construction, machine translation and text summarization. NER plays important role in part-of-speech (POS) tagging and parsing.

### NER methods

#### ➤ Lexicon Based Method

In lexical based approach uses a dictionary with a list of words or terms. The process involves checking if any of these words are present in a given text. However, this approach isn't commonly used because it requires constant updating and careful maintenance of the dictionary to stay accurate and effective.

#### ➤ Rule Based Method

The Rule Based NER method uses a set of predefined rules guides the extraction of information. These rules are based on patterns and context. Pattern-based rules focus on the structure and form of words, looking at their morphological patterns. On the other hand, context-based rules consider the surrounding words or the context in which a word appears within the text document. This combination of pattern-based and context-based rules enhances the precision of information extraction in NER.

#### ➤ Machine Learning-Based Method

This approach includes to train the model for multi-class classification using different machine learning algorithms that requires a lot of labelling. Other way is to Conditional random field (CRF) that is implemented by both NLP Speech Tagger and NLTK. It is a probabilistic model that can be used to model sequential data such as words.

#### ➤ Deep Learning Based Method

Deep learning NER system is much more accurate than other methods as it is capable to assemble words cause it used word embedding, that is capable of understanding the semantic and syntactic relationship between various words. With this approach it is also able to learn analyzes topic specific as well as high level words automatically.


#### 👩🏻‍💻 Implementation

With Spacy

```py
from spacy import load

text = "Tim Cook is the CEO of Apple company founded by Steve Jobs and Steve Wozniak in April 1976 at California."

nlp = load("en_core_web_sm")
doc = nlp(text)
print([(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents])
```

**Result**

```shell
[('Tim Cook', 0, 8, 'PERSON'), ('Apple', 23, 28, 'ORG'), ('Steve Jobs', 48, 58, 'PERSON'), ('Steve Wozniak', 63, 76, 'PERSON'), ('April 1976', 80, 90, 'DATE'), ('California', 94, 104, 'GPE')]
```

With bert

```py
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
result = nlp(text)

print(result)
```

**Result**

```shell
[{'entity': 'B-PER', 'score': 0.9997967, 'index': 1, 'word': 'Tim', 'start': 0, 'end': 3}, {'entity': 'I-PER', 'score': 0.99976486, 'index': 2, 'word': 'Cook', 'start': 4, 'end': 8}, {'entity': 'B-ORG', 'score': 0.9981711, 'index': 7, 'word': 'Apple', 'start': 23, 'end': 28}, {'entity': 'B-PER', 'score': 0.9997534, 'index': 11, 'word': 'Steve', 'start': 48, 'end': 53}, {'entity': 'I-PER', 'score': 0.9996791, 'index': 12, 'word': 'Job', 'start': 54, 'end': 57}, {'entity': 'I-PER', 'score': 0.98890764, 'index': 13, 'word': '##s', 'start': 57, 'end': 58}, {'entity': 'B-PER', 'score': 0.99980503, 'index': 15, 'word': 'Steve', 'start': 63, 'end': 68}, {'entity': 'I-PER', 'score': 0.99975544, 'index': 16, 'word': 'W', 'start': 69, 'end': 70}, {'entity': 'I-PER', 'score': 0.99936086, 'index': 17, 'word': '##oz', 'start': 70, 'end': 72}, {'entity': 'I-PER', 'score': 0.9995459, 'index': 18, 'word': '##nia', 'start': 72, 'end': 75}, {'entity': 'I-PER', 'score': 0.98740005, 'index': 19, 'word': '##k', 'start': 75, 'end': 76}, {'entity': 'B-LOC', 'score': 0.9994041, 'index': 24, 'word': 'California', 'start': 94, 'end': 104}]
```