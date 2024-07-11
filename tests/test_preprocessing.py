import os
import sys
sys.path.insert(1, os.getcwd())

import pytest
from src.text_preprocessing.preprocessing import Preprocess

@pytest.fixture(scope='function')
def test_data(request):
    """ Create object and pass arg."""
    params, expected = request.param
    p = Preprocess(params)
    return p, expected

@pytest.mark.parametrize(
        "test_data",
        [("HEllo", "hello"),
         ("NOW IS GOOD", "now is good"),
         ("Well Good Day To You!", "well good day to you!")], 
         indirect=True)
def test_case_lower(test_data):
    """ test text lowercase."""
    dataobj, expected = test_data
    assert (dataobj.case_lower()) == expected


@pytest.mark.parametrize(
    "input_data, expected",
    [("10", "ten"),
     ("81", "eighty-one"),
     ("1234", "one thousand, two hundred and thirty-four"),
     ("0056", "fifty-six"),
     ("8.26", "eight point two six"),
     ("12.78", "twelve point seven eight"),
     ("001.30", "one point three zero")],
)
def test_num_to_text(input_data, expected):
    """ test num2text. """
    dataobj = Preprocess(input_data)
    assert (dataobj.num_to_text(input_data)) == expected

@pytest.mark.parametrize(
        "test_data",
        [("I love choco.", ['I', 'love', 'choco.']),
         ("oh! is that you?", ['oh!', 'is', 'that', 'you?']),
         ("Well, Good Day To You! on 34", ['Well,', 'Good', 'Day', 'To', 'You!', 'on', '34'])], 
         indirect=True)
def test_split_text(test_data):
    """ test splitting text. """
    dataobj, expected = test_data
    assert (dataobj.split_text() == expected)

@pytest.mark.parametrize(
        "test_data",
        [("I love choco.", ['I', 'love', 'choco', '.']),
         ("oh! is that you?", ['oh', '!', 'is', 'that', 'you', '?']),
         ("Well, Good Day To You! on 34", ['Well', ',', 'Good', 'Day', 'To', 'You', '!', 'on', '34'])], 
         indirect=True)
def test_tokens(test_data):
    """ test tokenization. """    
    dataobj, expected = test_data
    assert (dataobj.tokens()) == expected


@pytest.mark.parametrize(
        "test_data, remove_this",
        [(("oh! this is cool", "cool"), "oh!"), # 'this' 'is' : already in stopwords 
         (("good day!", "good"), "day!"),       # 'good' : already in stopwords
         (("my id is 123 for account", "id 123 account"), ""),  # 'my' 'is' 'for': already in stopwords
         (("oh! did you see that tower", "see"), ['oh!', 'tower']),  # list to remove
         (("hey, that Eiffel tower is so tall ang high like sky", "Eiffel tower tall high like sky"),['hey,', 'so', 'ang'])],  # list to remove
         indirect=['test_data'])
def test_remove_stop_words(test_data, remove_this):
    """ test stop words. """
    dataobj, expected = test_data
    if isinstance(remove_this, list):
        assert isinstance(remove_this, list)
    else:
        assert isinstance(remove_this, str)
    assert (dataobj.remove_stop_words(remove_this)) == expected

@pytest.mark.parametrize(
        "test_data",
        [("Data science", ['data', 'scienc']),
         ("programming the computer", ['program', 'the', 'comput']),
         ("wonderful", ['wonder'])], 
         indirect=True)
def test_stemming(test_data):
    """ test stemming. """
    dataobj, expected = test_data
    assert (dataobj.stemming()) == expected

@pytest.mark.parametrize(
        "test_data",
        [("Data science", ['Data', 'science']),
         ("programming the computer", ['programming', 'the', 'computer']),
         ("wonderful", ['wonderful'])], 
         indirect=True)
def test_lemmatize(test_data):
    """ test lemmatization. """    
    dataobj, expected = test_data
    assert (dataobj.lemmatize()) == expected

@pytest.mark.parametrize(
        "test_data, expression, repl",
        [(("is this right123 or 34not", "is this right or not"), r"\d+", ''), # remove digit
         (("http//example.com is weburl.", " is weburl."), r"http\S+", ''),    # remove http
        (("oh! this-is..punct#u@ation?", "oh  this is  punct u ation "), r'[^\w\s]', ' ')], # remove punctuations
        indirect=['test_data'])
def test_remove_expression(test_data, expression, repl):
    """ test regex with expression. """
    dataobj, expected = test_data
    assert (dataobj.remove_expression(expression, repl)) == expected
