from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.recognizer_registry import RecognizerRegistry

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from typing import List, Optional, Union

class PiiEntity:
    """ PII detection.
        doc: https://microsoft.github.io/presidio/
        
        !pip install presidio_analyzer presidio_anonymizer -q
        !python -m spacy download en_core_web_lg -q
    """
    def __init__(self) -> None:
        # Init the objects
        self.analyzer = AnalyzerEngine()
        self.engine = AnonymizerEngine()
        self.registry = RecognizerRegistry()

        
    def custom_pii_operator(self, operator_performed: str = "replace", replaced_value:str = "screte_001"):
        """ Returns custom generated value.
            operator_performed: operation to perform (supported:hash, replace, redact, encrypt, mask)
            replaced_value: value to replce
        """
        new_value = {"new_value": replaced_value}
        return OperatorConfig(operator_name=operator_performed, params=new_value)

    def operator(self):
        """ Custom masking method for pii entity."""
        custom_operator = {"DEFAULT": self.custom_pii_operator(),
                           "US_SSN": self.custom_pii_operator(operator_performed="replace",replaced_value="SSN_CODE"),
                           "TITLE": self.custom_pii_operator(replaced_value="Sir"),
                           "US_PASSPORT": self.custom_pii_operator(replaced_value="PASS0102"),
                           "US_BANK_NUMBER": self.custom_pii_operator(replaced_value="DUMMY1234"),
                           "PHONE_NUMBER": self.custom_pii_operator(replaced_value="0123456789"),
                           "CREDIT_CARD": self.custom_pii_operator(replaced_value="CREDIT1212"),
                           "URL": self.custom_pii_operator(replaced_value="dummy/url.com"),
                           "DATE_TIME": self.custom_pii_operator(replaced_value="00 00 0000"),
                           "DATE": self.custom_pii_operator(replaced_value="12-12-12"),
                           "TIME": self.custom_pii_operator(replaced_value="00:00"),
                           "NUMBER": self.custom_pii_operator(replaced_value="00"),
                           "LOCATION": self.custom_pii_operator(replaced_value="LOC"),
                           "EMAIL_ADDRESS": self.custom_pii_operator(replaced_value="dummy@email"),
                           }
        return custom_operator

      
    def pii_entity_extraction(self, text):
        """PII extraction.
            returns
            analyzer_result: extracted entity type, score, start and end index
            anonymizer_result: encrypted text and extracted items
        """
        analyzer_result = self.analyzer.analyze(text=text, language="en", allow_list=['Alvin']) # allow_list: list of custom words that cannot considered during pii
        anonymizer_result = self.engine.anonymize(text=text, analyzer_results=analyzer_result, operators=self.operator())
        return analyzer_result, anonymizer_result
    
if __name__ == "__main__":
    text = "Mr. Alvin lives in Berlin and his id is 453612345678 with alvin@gmail.com at 4 AM"
    p = PiiEntity()
    analyzer_score, encryted = p.pii_entity_extraction(text=text)
    print(f"{analyzer_score=}")
    print(f"{encryted.text}") # Mr. Alvin lives in LOC and his id is DUMMY1234 with dummy@email at 00 00 0000
    print(f"{encryted.items}")