# Language Detection

Language comes from Old French *langage*, based on Latin word lingua means 'tongue'. Language is a structured system of communication that consists of grammar and vocabulary but it convey more than just communication. It has a very long evolutionary history of itself. Languages are such a amazing thing! isn't it ^^


So let's identify these languages.

#### Language Detection with Spacy

```py
import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector

def get_lang_detector(nlp, name):
    return LanguageDetector(seed=42) 


nlp_model = spacy.load("en_core_web_sm")
# Language.factory("language_detector", func=get_lang_detector)
nlp_model.add_pipe('language_detector', last=True)

# Document level language detection
job_title = "Senior NLP Research Engineer"
doc = nlp_model(job_title)
language = doc._.language
print(language)

# Sentence level language detection
text = "こんにちは お元気ですか.  This is English text. नमस्ते, आप कैसे हैं. Er lebt mit seinen Eltern und seiner Schwester in Berlin. Yo me divierto todos los días en el parque. Je m'appelle Angélica Summer, j'ai 12 ans et je suis canadienne. హలో, ఎలా ఉన్నారు."
doc = nlp_model(text)
for i, sent in enumerate(doc.sents):
    print(sent, sent._.language)

```

**Result**

```shell
{'language': 'en', 'score': 0.9999944616311092}
こんにちは お元気ですか.   {'language': 'ja', 'score': 0.9999999999820187}
This is English text. {'language': 'en', 'score': 0.9999987929307772}
नमस्ते, आप कैसे हैं. {'language': 'hi', 'score': 0.9999969329463939}
Er lebt mit seinen Eltern und seiner Schwester in Berlin. {'language': 'de', 'score': 0.999996045846908}
Yo me divierto todos los días en el parque. {'language': 'es', 'score': 0.9999960751128255}
Je m'appelle Angélica Summer, j'ai 12 ans et je suis canadienne. {'language': 'fr', 'score': 0.9999960488878061}
హలో, ఎలా ఉన్నారు. {'language': 'te', 'score': 0.9999999998909419}
```


#### Language Detection with HF

```py
from transformers import pipeline

text = [
    "Brevity is the soul of wit.",
    "こんにちは お元気ですか.",
    "காலை வணக்கம்.",
    "Oh, ¿viste ese vestido colorido?",
    "Θα πάω εκεί να δω λουλούδια",
    "J'aime manger du chocolat",
    "Ich werde dorthin gehen, um Blumen zu sehen",
]

model = "papluca/xlm-roberta-base-language-detection"
pipe = pipeline("text-classification", model=model)
pipe(text, top_k=1, truncation=True)
```

**Result**

```shell
[[{'label': 'en', 'score': 0.8889274001121521}],
 [{'label': 'ja', 'score': 0.9445705413818359}],
 [{'label': 'hi', 'score': 0.9658094048500061}],
 [{'label': 'es', 'score': 0.8788681030273438}],
 [{'label': 'el', 'score': 0.9942275285720825}],
 [{'label': 'fr', 'score': 0.9696151614189148}],
 [{'label': 'de', 'score': 0.9949468970298767}]]
```