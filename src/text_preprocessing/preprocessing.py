import re
from string import punctuation
from typing import Union, Optional

class Preprocess:
    def __init__(self,
                 text:str,
                 case_sensitive:bool = False):
        self.text = text
        self.extracted = dict
        self.words = self.split_text() # use as token
        self.case_sensitive = case_sensitive

    def case_lower(self):
        """ convert text to lower case. """
        if not self.case_sensitive:
            return self.text.lower()

    def num_to_text(self, word):
        """ Convert numbers to words."""
        import inflect
        e = inflect.engine()
        # check if word(splitted text) is digit or numeric 
        return e.number_to_words(word)

    def split_text(self, num2text:bool = False):
        """ Splits text into words with removed extra space. """
        words = self.text.split()

        # convert numbers to text (numerics and fractional numbers)
        if num2text:
            words_ = []
            re_float = re.compile(r"(^\d+\.\d+$|^\.\d+$)") # match numbers but with exactly 1 dot followed by number
            for w in words:
                words_.append(self.num_to_text(w) if w.isnumeric() or re_float.match(w) else w)
            return words_
        return words
    
    def tokens(self):
        """ Split text into tokens with nltk.tokenize. """
        from nltk.tokenize import word_tokenize
        return word_tokenize(self.text)
    
    def remove_stop_words(self, remove_this:Union[str, list[str]]):
        """ Removes stop_words from text. """
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        # add custom stop words to remove
        if isinstance(remove_this, list):
            for w in remove_this:
                stop_words.add(w) 
        else:
            stop_words.add(remove_this)
        return " ".join([word for word in self.words if word not in stop_words])

    def stemming(self):
        """ Stemming text. 
            ex: programmer, programming, program --> stemmed to(reverrt to / root): program.
            ex: science --> scie
        """
        from nltk.stem.porter import PorterStemmer
        porter_stemmer = PorterStemmer()
        return [porter_stemmer.stem(word) for word in self.words]

    def lemmatize(self):
        """ Lemmatize text. 
            ex: goose, geese --> lemmatize to(root) : goose 
            It ensures that the root word belongs to the language
        """
        from nltk.stem import WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()
        return [wordnet_lemmatizer.lemmatize(word) for word in self.words]
    
    def remove_expression(self, expression:str, repl=''):
        """ Removes text matches in expression.
            expression (str): regex
        """
        return re.sub(expression, repl, self.text)



if __name__ == "__main__":
    text = "Hey, did you know, that the Eiffel Tower can be 15 cm taller in the summer. Heat causes the metal in the structure to expand!!"
    text2 = "hey, that Eiffel tower is so tall ang high like sky"
    p = Preprocess(text=text2)
    rem = ['hey,', 'so', 'ang']
    # result = p.stemming()
    result = p.remove_stop_words(rem)
    print(f"{result=}")


