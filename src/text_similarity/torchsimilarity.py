from torch import tensor
from torch.nn import CosineSimilarity

class TorchSimilarity:
    def text_similarity(text1, text2):
        # Custom encoding
        encoding_dict1 = {char: i for i, char in enumerate(set(''.join([text1])))}
        encoding_dict2 = {char: i for i, char in enumerate(set(''.join([text2])))}

        tensor1 = tensor([[encoding_dict1[char] for char in string] for string in [text1]], dtype=float)
        tensor2 = tensor([[encoding_dict2[char] for char in string] for string in [text2]], dtype=float)

        # Size of tensors must be match
        if tensor1.shape == tensor2.shape:
            # Calculate the cosine similarity
            cosine_similarity = CosineSimilarity(dim=0)
            similarity = cosine_similarity(tensor1, tensor2)
        else:
            raise RuntimeError(f"{tensor1.shape} and {tensor2.shape} are mismatched")

        return similarity
