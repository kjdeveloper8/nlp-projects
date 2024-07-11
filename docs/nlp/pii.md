# PII Detection

PII stands for Personal Identifialbe Information. Any information that can uniquely identify people as individuals is called PII. Such as name, email, credit card number, social security number (usa), aadhaar number (india), fiscal code (italian), NIF number (spanish), driver license number, bank details, passport etc.

PII contains sensitive and non sensitive information therefore it is good practice to encrypt them.

For that we will used microsoft presidio a toolkit that provides fast identification and anonymization modules for private entities in text and images such as credit card numbers, names, locations, social security numbers, bitcoin wallets, US phone numbers, financial data and more.

#### 👩🏻‍💻 Implementation

With [MS Presidio](https://microsoft.github.io/presidio/)

Installation 

```py
pip install presidio_analyzer presidio_anonymizer -q
python -m spacy download en_core_web_lg -q
```

##### Detect PII


Initialize

```py
analyzer = AnalyzerEngine()
engine = AnonymizerEngine()
registry = RecognizerRegistry()
```

```py
text = "Mr. Alvin lives in Berlin and his id is 453612345678 with alvin@gmail.com at 4 AM"
analyzer_results = analyzer.analyze(text=text, language="en")
print(analyzer_results)
```

result

```shell
[type: EMAIL_ADDRESS, start: 58, end: 73, score: 1.0, type: PERSON, start: 4, end: 9, score: 0.85, type: LOCATION, start: 19, end: 25, score: 0.85, type: PERSON, start: 58, end: 73, score: 0.85, type: DATE_TIME, start: 77, end: 81, score: 0.85, type: URL, start: 64, end: 73, score: 0.5, type: US_BANK_NUMBER, start: 40, end: 52, score: 0.05, type: US_DRIVER_LICENSE, start: 40, end: 52, score: 0.01]
```

Get encrypted text

```py
result = engine.anonymize(
    text=text,
    analyzer_results = analyzer_results,
    operators={"PERSON": OperatorConfig("replace", {"new_value": "SECRET"}), 
               "DATE_TIME":OperatorConfig("replace", {"new_value": "00 00 0000"}),
               "EMAIL_ADDRESS":OperatorConfig("replace", {"new_value": "dummay@email.com"}),
               "LOCATION":OperatorConfig("replace", {"new_value": "LOC"}),
               "US_BANK_NUMBER":OperatorConfig("replace", {"new_value": "BA12345"}),
               },
)

print(result.text)
print(result.items)
```

result

```shell
text: Mr. SECRET lives in LOC and his id is BA12345 with dummay@email.com at 00 00 0000
items:
[
    {'start': 71, 'end': 81, 'entity_type': 'DATE_TIME', 'text': '00 00 0000', 'operator': 'replace'},
    {'start': 51, 'end': 67, 'entity_type': 'EMAIL_ADDRESS', 'text': 'dummay@email.com', 'operator': 'replace'},
    {'start': 38, 'end': 45, 'entity_type': 'US_BANK_NUMBER', 'text': 'BA12345', 'operator': 'replace'},
    {'start': 20, 'end': 23, 'entity_type': 'LOCATION', 'text': 'LOC', 'operator': 'replace'},
    {'start': 4, 'end': 10, 'entity_type': 'PERSON', 'text': 'SECRET', 'operator': 'replace'}
]
```