from helper.util import Estimate

@Estimate.timer
def ner_spacy(text):
    from spacy import load
    # load spacy model
    nlp = load("en_core_web_sm")
    doc = nlp(text)
 
    return [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]


@Estimate.timer
def ner_bert(text):
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from transformers import pipeline

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    return nlp(text)



if __name__ == "__main__":
    text = "Tim Cook is the CEO of Apple company founded by Steve Jobs and Steve Wozniak in April 1976 at California."
    result = ner_spacy(text)
    # result = ner_bert(text)
    print(f"{result}")