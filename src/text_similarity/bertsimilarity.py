from transformers import BertTokenizer, BertModel
from torch import tensor, no_grad
from sklearn.metrics.pairwise import cosine_similarity

class BERTSimilarity:
    def text_similarity(text1, text2):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        # Tokenize the sentences
        tokens1 = tokenizer.tokenize(text1)
        tokens2 = tokenizer.tokenize(text2)

        # Add [CLS] and [SEP] tokens for separating two texts
        tokens = ['[CLS]'] + tokens1 + ['[SEP]'] + tokens2 + ['[SEP]']
        # print(f"{tokens=}")

        # Convert tokens to input IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # print(f"{input_ids=}")

        # Convert tokens to input IDs
        input_ids1 = tensor(tokenizer.convert_tokens_to_ids(tokens1)).unsqueeze(0)  # batchsize 1
        input_ids2 = tensor(tokenizer.convert_tokens_to_ids(tokens2)).unsqueeze(0)  # batchsize 1

        # embeddings
        with no_grad():
            outputs1 = model(input_ids1)
            outputs2 = model(input_ids2)
            embeddings1 = outputs1.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings2 = outputs2.last_hidden_state[:, 0, :]  # [CLS] token

        # Calculate similarity
        similarity_score = cosine_similarity(embeddings1, embeddings2)
        return similarity_score
