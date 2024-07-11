# JSON Parser

JSON  is a lightweight data-interchange format which is easy to read and write.

JSON is built on two structures:


<div class="grid" markdown>

1. A collection of name/value pairs. In various languages, this is realized as an object, record, struct, dictionary, hash table, keyed list, or associative array.

2. An ordered list of values. In most languages, this is realized as an array, vector, list, or sequence.

</div>


JSON is very popular format for data exchange between web applications and APIs.

Json's text-based format is used to store data in a key-value format which is very similar to python dictionaries and javaScript objects.


## Parse Json

Json data is readble but not searchable!

As they made up of many many nested layers with lots of keys and arrays. It's not uncommon to see Json datasets with hundreds of keys and values so it becomes very complex.

Therefore parsing is needed to make data accessability faster and more flexible.

### Nested-lookup

The nested_lookup package provides many Python functions for working with deeply nested documents. A document in this case is a a mixture of Python dictionary and list objects typically derived from YAML or JSON.

**sample data**
```json
data = {
    "product":[{
        "T-shirt":{
            "property": [
                {"size": "L", "color": "blue"},
                {"size": "S", "color": "pink"},
                {"size": "M", "color": "black"}
                ],
            "category":[
                {"officewear": [
                    {"type": "polo shirt", "price": 100}
                    ]
                },
                {"casualwear":[
                    {"type": "long sleeves", "price": 200}
                ]}
            ]
        },
        "Dress":{
            "property":[
                {"color":"red"},
                {"color":"white"}
            ]
        }
    }]
} 
```

Install 

```shell
pip install nested-lookup 
```

##### lookup
`nested_lookup` returns list of matched key lookups
```py 
from nested_lookup import nested_lookup

print(nested_lookup('color', data))
# ['blue', 'pink', 'black', 'red', 'white']
```

##### update
`nested_update` returns all occurences of the given key and update the value. By default, returns a copy of the document. To mutate the original specify the in_place=True argument.

```py
from nested_lookup import nested_update

# before
print(nested_lookup('price', data)) # [100, 200]

nested_update(data, 'price', 300, in_place=True)

# after
print(nested_lookup('price', data)) # [300, 300]
```

##### delete

`nested_delete` returns all occurrences of the given key and delete it. By default, returns a copy of the document. To mutate the original specify the in_place=True argument.

```py
from nested_lookup import nested_delete
# before
print(nested_lookup('price', data)) # [100, 200]

nested_delete(data, 'price', in_place=True)

# after
print(nested_lookup('price', data)) # []
```

##### alter

`nested_alter` returns all occurrences of the given key and alter it with a callback function. By default, returns a copy of the document. To mutate the original specify the in_place=True argument.

```py
from nested_lookup import nested_alter
# before
print(nested_lookup('price', data)) # [100, 200]

def callback_to_alter(self, something:int):
    return something + 500

nested_alter(data, 'price', callback_to_alter, in_place=True)

# after
print(nested_lookup('price', data)) # [600, 700]
```

##### get keys

`get_all_keys` returns a list of keys

```py
from nested_lookup import get_all_keys
get_all_keys(data)
# ['product', 'T-shirt', 'property', 'size', 'color', 'size', 'color', 'size', 'color', 'category', 'officewear', 'type', 'price', 'casualwear', 'type', 'price', 'Dress', 'property', 'color', 'color']
```

##### get occurrences

`get_occurrence_of_key` and `get_occurrence_of_value` returns the number of occurrences of a key/value from a nested dictionary

```py
from nested_lookup import get_occurrence_of_key, get_occurrence_of_value

key_occurence = get_occurrence_of_key(data, key="color") # 5
val_occurence = get_occurrence_of_value(data, value="red") # 1
```

### Jsonpath

This library differs from other JSONPath implementations in that it is a full _language_ implementation, meaning the JSONPath expressions are first class objects, easy to analyze, transform, parse, print, and extend.

Install 

```shell
pip install jsonpath-ng 
```

```py
from jsonpath_ng import parse
from jsonpath_ng.jsonpath import Fields, Slice

expression = 'product[*].Dress.property[*].color'
# convert custom_path to str to pass as expression 
# ex: parse(str(custom_path))
custom_path = Fields('product').child(Slice('*')).child(Fields('Dress'))

parsed_data = parse(expression)
for match in parsed_data.find(data):
    print(match.value)
    print(str(match.full_path))

# match_value: ['red', 'white']
# match_full_path: ['product.[0].Dress.property.[0].color', 'product.[0].Dress.property.[1].color']
```