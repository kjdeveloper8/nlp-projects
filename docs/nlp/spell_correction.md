# Spell correction
Spell correction with dictionary based approach using pyspellchecker.

---

## Introduction

As we made many mistakes in typoes and writing correcting spelling mistakes is necessary 
while texting a phone, sending an email, writing large documents or searching for information on the web.

There are many approaches for correcting spelling.
- Norvig's approach
- Language model based
- Phonetic matching based on pronunciation
- Rule based approach with pattern matching
- Dictionary based
  
Let's implement using dictionary based approach with pyspellchecker.

Pyspellchecker is based on Norvig’s algorithm with a Levenshtein Distance algorithm to find permutations within an edit distance of 2 from the original word. It then compares all permutations (insertions, deletions, replacements, and transpositions) to known words in a word frequency list. Those words that are found more often in the frequency list are more likely the correct results.

Pyspellchecker also supports (lang): en, es, fr, it, pt, de, ru, ar, lv, eu, nl 

🔗 [pyspell docs](https://pyspellchecker.readthedocs.io/en/latest/index.html)

### Installation
```shell
pip install pyspellchecker
```

#### Create a pyspellchecker object
```python
from spellchecker import SpellChecker
correct = SpellChecker()
```

Load word frequency dictionary
```py
correct.word_frequency.dictionary
```

word frequency dictionary 
```shell
   { "a": 48779620,
    "aah": 50,
    "aalii": 50,
    "aardvark": 106, ...}
```

#### Now add corpus words in this dictionary
```py
with open('corpus.txt', 'r', encoding='utf-8') as rfile:
    data = rfile.read()
words = list(set([word for word in data.split()])) # list of unique words
# add this words to dict
correct.word_frequency.load_words(words)
```

#### Load a dictionary if have one already
```py
correct.word_frequency.load_dictionary('corpus.json')
```

#### Get the corrected word
```py
correct.correction('tomorow') # tommorrow
```

#### Get the suggestion for word
```py
correct.candidates('wirld') # {'wild', 'wired', 'world', 'wield'}
```

#### Add custom word to dictinary
```py
correct.word_frequency.add('chococolato')
```

#### Remove words from dictionary
```py
correct.word_frequency.remove('chococolato')
correct.word_frequency.remove_words(['apple', 'banana', 'berry']) # list
```

#### Remove words by threshould value
Removes words at or below threshould value
```py
# removes words having freq value 25 or below
correct.word_frequency.remove_by_threshold(25)
```

#### Check if word exists in dictionary or not
```py
correct.known('apple')
```

#### Calculate word frequency count
```py
correct.word_usage_frequency(['apple'])
```

#### 👩🏻‍💻 Implementation
Now putting all this together

```py title='spell_correction.py'

from spellchecker import SpellChecker
from pathlib import Path
from typing import Union

DIR = "path/to/corpus"
class Colors:
    YELLOW = '\033[33m'
    ENDC = '\033[m'

class CorrectSpell:
    """ Custom spell correction with pyspellchecker.
        docs: https://pyspellchecker.readthedocs.io/en/latest/index.html
    """
    def __init__(self,
                 corpus:Path,
                 lang:str = "en"):
        # supported lang(by pyspellchecker): [en, es, fr, it, pt, de, ru, ar, lv, eu, nl]
        try:
            self.correct = SpellChecker()
        except ImportError:
            print("!pip install pyspellchecker")
        
        self.corpus = corpus
        self.load()

    def load(self):
        """ Load data with all frequency count.
            corpus: data filepath to load (format: json or txt)
        """
        if self.corpus.suffix == '.json': 
            self.correct.word_frequency.load_dictionary(self.corpus)

        elif self.corpus.suffix == '.txt':
            with open(self.corpus, 'r', encoding='utf-8') as rfile:
                data = rfile.read()
            words = list(set([word for word in data.split()])) # list of unique words
            # add this words to dict
            self.correct.word_frequency.load_words(words)
            # self.correct.export('corpus_freq.gz')
        else:
            raise ValueError("Please provide valid file format!")
        
    def build_dict(self, store:Union[str, Path]="word_freq.json"):
        """ Build dict with words and freq count.
            store (str, Path): filename/path to store
        """
        import json
        with open(store, 'w', encoding='utf-8') as wfile:
            json.dump(self.correct.word_frequency.dictionary, wfile, sort_keys=True, indent=4)

    def word_freq(self, words:Union[list[str]]):
        """ Returns word frequency."""
        return [(w,self.correct.word_usage_frequency(w)) for w in words]

    def add_to_dict(self, word:str):
        """ Add words to existing dictionary.
            word (str): word to add
        """
        self.correct.word_frequency.add(word)
    
    def remove_from_dict(self, word:Union[Union[str,list[str]],int]):
        """ Add words to existing dictionary.
            word (str, list): word or words to remove (int if wanted to remove from certain threshould)
        """
        if isinstance(word, str):
            # remove word from dict
            self.correct.word_frequency.remove(word)

        elif isinstance(word, list):
            # remove list of words from dict
            self.correct.word_frequency.remove_words(word)

        elif isinstance(word, int):
            # remove words from dict <= threshold value
            self.correct.word_frequency.remove_by_threshold(word)
        
        else:
            raise ValueError("Unable to remove!")

    def spell_corr(self, word:str):
        """ Returns corrected word.
            word (str): word to be corrected 
        """
        # correct the word if found in dict
        corrected = self.correct.correction(word)
        # set color highlighter for word (incorrected)
        # that is not found or added in dict
        word = str(Colors.YELLOW + word + Colors.ENDC)
        return word if corrected is None else corrected
    
    def check_word_in_dict(self, word):
        """ Check if word is in dict or not."""
        return True if self.correct.known(word) else False

    def suggestions(self, word:str):
        """ Returns suggested words matched from dict.
            word (str): get suggestions for word
        """
        return self.correct.candidates(word)


if __name__ == "__main__":
    fpath = DIR.joinpath("spell_correction/corpus.txt")
    query = "Tomorow i am ging to Disney parkkk at 10 o\'clock !Yahooooo"
    
    c = CorrectSpell(fpath)
    c.add_to_dict('Disney')
    result = " ".join([c.spell_corr(q) for q in query.split()])
    print(f"corrected: {result}")

    suggestions = c.suggestions('cheet')
    print(f"suggestions: {suggestions}")
```