from spellchecker import SpellChecker
from helper.util import DIR, Colors
from pathlib import Path
from typing import Union

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
    