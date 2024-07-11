# Question Answer in NLP

Question Answering (QA) in AI refers to the capability of a machine to respond to questions asked in natural language. The main objective of this technology is to extract relevant information from vast amounts of data and present it in the form of a concise answer. 

QA models take in a question and then process a large amount of text data to determine the most accurate answer. For example, if the question is "What is the color of banana" the QA model will scan its database and return the answer "Yellow". 

Nowadays, the demand for conversational AI systems and virtual assistants has grown, which has driven the development of Question Answering in NLP. These systems rely on NLP techniques like text classification, sentiment analysis, information retrieval, machine translation, and generate answers.

### ➤ QA system flow

#### 1. Data Collection and Preprocessing

First is to collect a large corpus of text data from sources like books, online news articles, or databases. After that cleaning and preprocessing techniques like tokenization, steeming and lemmatization is done in order to remove irrelevant information.

#### 2. Information Retrieval

Information retrival can be done by algorithms that can extract relevant information from the text corpus to answer questions. This includes ner, semantic search, keyword search, text classification. 

#### 3. Question Analysis

It's important to analyze the question to understand its intent and identify keywords or phrases that will guide the information retrieval process. This can involve using techniques like POS tagging, dependency parsing, and named entity recognition to identify important words and phrases in the question.

#### 4. Answer Generation

This often involves techniques like text generation and summarization for answer. For example, a text generation algorithm can generate a response based on the most relevant information retrieved in the previous step.

#### 5. Model Training and Evaluation

Model is trained to identify patterns in the data and improve the accuracy of the answers generated. For evaluation metrics like precision, recall, and F1 score is used to evaluates the performance of the QA system.


### 👩🏻‍💻 Implementation

QA with Huggingface model `BertForQuestionAnswering` which is fine tuned on Stanford Question Answering Dataset (SQuAD) dataset

##### Import 

```py
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
```

##### Load model 

```py
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
```

##### Sample qa 

```py
question = "Where is the Great Barrier Reef located?"
answer_text = "The Great Barrier Reef is located in the Coral Sea, off the coast of Australia. It is the largest coral reef system in the world, stretching over 2,300 km and covering an area of approximately 344,400 km². The Great Barrier Reef is home to a diverse range of marine life and is considered one of the seven natural wonders of the world. It is also a UNESCO World Heritage Site threatened by climate change and other environmental factors."
```

##### Tokenization and attention masking

```py
input_ids = tokenizer.encode(question, answer_text)
attention_mask = [1] * len(input_ids)
```

##### Get the logits 

```py
output = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
start_index = torch.argmax(output[0][0, :len(input_ids) - input_ids.index(tokenizer.sep_token_id)])
end_index = torch.argmax(output[1][0, :len(input_ids) - input_ids.index(tokenizer.sep_token_id)])
```

##### Decode the answer 

```py
answer = tokenizer.decode(input_ids[start_index:end_index + 1], skip_special_tokens=True)
# answer: 'coral sea'
```
