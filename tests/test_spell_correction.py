import os
import sys
sys.path.insert(1, os.getcwd())

import pytest
from src.spell_correction.spell_correction import CorrectSpell

@pytest.fixture
def input_word():
    return "disney"

@pytest.fixture
def data(tmp_path):
    """A fixture that returns CorrectSpell object with 
    created temp path with text data file and content.""" 
    # temp path
    temp_dir = tmp_path / "temp_dir"  
    temp_dir.mkdir()  
  
    # temp directory  
    temp_file = temp_dir / "test_file.txt"  
    # write data
    temp_file.write_text("Tomorrow i am going to Disney park!")  
  
    # check if the file exists  
    assert temp_file.is_file()  
  
    # read data  
    assert temp_file.read_text() == "Tomorrow i am going to Disney park!"
    corr = CorrectSpell(temp_file)
    return corr

def test_input_word(input_word):  
    assert input_word in "disney"


@pytest.mark.parametrize(
        "test_input, expected_value, expected_freq", 
        [("today", "today", 0.00017942648950299852), 
         ("hello", "hello", 3.723237640377041e-05),])
def test_word_freq(data, test_input, expected_value, expected_freq):
    """test word frequency."""
    test_input = ["today"]
    expected = data.word_freq(test_input)
    for input, expected_value in zip(test_input, expected):
        assert input == expected_value[0]
    for expected_value, expected_freq in expected:
        assert isinstance(expected_value, str) and isinstance(expected_freq, float)

def test_add_to_dict(data):
    """ test add word."""
    assert not data.check_word_in_dict("himalayas") == 0 # False == False : True (so add not)
    data.add_to_dict("himalayas")
    assert data.check_word_in_dict("himalayas") == 1

def test_remove_from_dict(data):
    """ test remove word."""
    assert data.check_word_in_dict("morning") == 1
    data.remove_from_dict("morning")
    assert not data.check_word_in_dict("morning") == 0

@pytest.mark.parametrize(
        "test_input, expected",
        [("tomorow", "tomorrow"),
         ("ging", "going"),
         ("park", "park")])
def test_spell_corr(data, test_input, expected):
    """test spell correction."""
    assert (data.spell_corr(test_input)) == expected

def test_check_word_in_dict(data):
    """test if word is known or not."""
    assert data.check_word_in_dict("hello") == 1
    assert data.check_word_in_dict("alphabet") == 1
    assert not data.check_word_in_dict("yaay") == 0 
    assert not data.check_word_in_dict("disney") == 0
    assert data.check_word_in_dict("morning") == 1


@pytest.mark.parametrize(
        "test_input, expected",
        [("wirld", {'wild', 'wired', 'world', 'wield'}),
         ("looloooooo", None),
         ("cheery", {'cheery'}),
         ("nnnnnn", None),
         ("cheet", {'cheep', 'chevet', 'chert', 'cheek', 'chest', 'cheat', 'sheet', 'cheer'})])
def test_suggestions(data, test_input, expected):
    """test word suggestions."""
    assert (data.suggestions(test_input)) == expected

def test_build_dict(data, tmp_path):
    """test build dict."""
    data.build_dict()
    assert data.check_word_in_dict("disney") == 1