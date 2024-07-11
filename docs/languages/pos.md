# Part Of Speech: POS Tagging

Part-of-Speech (POS) tagging is a natural language processing technique that involves assigning specific grammatical categories or labels (such as nouns, verbs, adjectives, adverbs, pronouns, etc.) to individual words within a sentence. This process provides insights into the syntactic structure of the text, aiding in understanding word relationships, disambiguating word meanings, and facilitating various linguistic and computational analyses of textual data.

POS tagging is essential for comprehending a language’s syntactic structure, named entity recognition, information retrieval, and machine translation.

POS are language dependent as diffrernt languages have different rules and grammers. Even though there are universal POS tagsets, it can be difficult to develop completely language-independent models because different languages have different rules and difficulties.


##### 👩🏻‍💻 Implementation

```py
from spacy import load
# Load model 
nlp = load("en_core_web_sm") 

text = "Tim Cook is the CEO of Apple company founded by Steve Jobs and Steve Wozniak in April 1976."

doc = nlp(text) 
for token in doc: 
    print(token, token.pos_) 

print("Verbs:", [token.text for token in doc if token.pos_ == "VERB"])
```

**Result**

```shell
Tim PROPN
Cook PROPN
is AUX
the DET
CEO PROPN
of ADP
Apple PROPN
company NOUN
founded VERB
by ADP
Steve PROPN
Jobs PROPN
and CCONJ
Steve PROPN
Wozniak PROPN
in ADP
April PROPN
1976 NUM
. PUNCT
Verbs: ['founded']
```
