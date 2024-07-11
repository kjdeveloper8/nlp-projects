import json
from pathlib import Path
from helper.util import DIR
from typing import Union, Optional
from collections.abc import Callable
from jsonpath_ng import parse
from jsonpath_ng.jsonpath import Fields, Slice

class Load:
    """ Loads Json file."""
    def __init__(self, filepath: Path) -> None:
        self.file_path = filepath
        self.data = self.load_file()

    def load_file(self):
        """ Load json data."""
        if self.file_path.suffix == '.json':
            with open(self.file_path, "r") as rfile:
                data = json.load(rfile)
                return data
        else:
            raise ValueError("Please provide valid json file format!")

class ParseJson(Load):     
    """ Parse Json with nested lookup. """
    def __init__(self, filepath: Path) -> None:
        super().__init__(filepath)
    
    def lookup(self, key:str):
        """ Returns key value in nested json. 
            key (str): key to search
        """
        from nested_lookup import nested_lookup
        return nested_lookup(key, self.data)

    def update(self, key:str, updated_val:Union[str, list, dict]):
        """ Update the value of key.
            key (str): key to be updated
            updated_val (str, list, dict): value to be updated
            By default, returns a copy of the document. To mutate the original specify the in_place=True argument
        """
        from nested_lookup import nested_update
        return nested_update(self.data, key, updated_val, in_place=False)
    
    def delete(self, key:str):
        """ Delete the value of key.
            key (str): key to be deleted
            By default, returns a copy of the document. To mutate the original specify the in_place=True argument
        """
        from nested_lookup import nested_delete
        return nested_delete(self.data, key, in_place=False)
    
    def callback_to_alter(self, something:int):
        """ Callback for alter."""
        return something + 500

    def alter(self, key:str, callback:Callable[[int], int]):
        """ Alters the value of key.
            key (str): key to be altered
            callback (Callable): callback function
            By default, returns a copy of the document. To mutate the original specify the in_place=True argument
        """
        from nested_lookup import nested_alter
        return nested_alter(self.data, key, callback, in_place=True)
    
    def get_keys(self):
        """ Returns all the key. """
        from nested_lookup import get_all_keys
        return get_all_keys(self.data)

    def get_occurrences(self, key:str, value:Union[str, int]):
        """ Returns all the occurrences of key and value. 
            key (str): occurrences of key to be counted
            value (str, int): occurrences of value to be counted
        """
        from nested_lookup import get_occurrence_of_key, get_occurrence_of_value
        key_occurence = get_occurrence_of_key(self.data, key)
        val_occurence = get_occurrence_of_value(self.data, value)
        return key_occurence, val_occurence


class ParseJson2(Load):
    """ Parse Json with jsonpath_ng. """
    def __init__(self, filepath: Path, expression:str) -> None:
        super().__init__(filepath)
        self.expression = expression
        self.parsed_data = self.get_parsed()

    def get_parsed(self):
        """ Return parsed data. """
        return parse(self.expression)

    def lookup_with_expr(self):
        """ Returns data matched with given expression."""
        match_val = []
        match_val_path = []
        for match in self.parsed_data.find(self.data):
            match_val.append(match.value)
            match_val_path.append(str(match.full_path))
        return match_val, match_val_path

if __name__ == "__main__":
    file = DIR.joinpath("parser/sample.json")
    pj = ParseJson(file)
    result = pj.lookup('color')
    print(f"{result=}")
    expression = 'product[*].Dress.property[*].color'
    # convert custom_path to str 
    # to pass as expression 
    # ex: ParseJson2(file, str(custom_path))
    custom_path = Fields('product').child(Slice('*')).child(Fields('Dress'))

    p = ParseJson2(file, expression)
    result2 = p.lookup_with_expr()
    print(f"{result2=}")

