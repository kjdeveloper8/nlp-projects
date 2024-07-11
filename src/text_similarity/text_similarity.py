# plug and play 
import numpy as np

from helper.util import Estimate
from nltksimilarity import NLTKSimilarity
from sklearnsimilarity import SklearnSimilarity
from bertsimilarity import BERTSimilarity
from torchsimilarity import TorchSimilarity

class Similarity:
    @Estimate.timer
    def get_similarity(self, text1, text2, similarity="nltk"):
        if similarity == "nltk":
            sim = NLTKSimilarity
        elif similarity == "sklearn":
            sim = SklearnSimilarity
        elif similarity == "bert":
            sim = BERTSimilarity
        elif similarity == "torch":
            sim = TorchSimilarity
        else:
            raise ValueError(f"{similarity} does not supported or invalid!")

        result = sim.text_similarity(text1, text2)
        return result

class Matrices:
    @Estimate.timer
    @staticmethod
    def cal_cosine_similarity():
        A = np.array([5, 3, 4])
        B = np.array([4, 2, 4])

        dot_product = np.dot(A, B)
        magnitude_A = np.linalg.norm(A)
        magnitude_B = np.linalg.norm(B)

        cosine_similarity = dot_product / (magnitude_A * magnitude_B)
        print(f"{cosine_similarity=}")

    @Estimate.timer
    @staticmethod
    def cal_jaccard_simialrity():
        text1 = {"It", "might", "not", "rain", "today" }
        text2 = {"It", "might", "not", "work", "today"}
        intersection = len(text1.intersection(text2))
        union = len(text1.union(text2))
        
        jaccard_similarity = intersection / union
        print(f"{jaccard_similarity=}")

    @Estimate.timer
    @staticmethod
    def cal_euclidean_distance():
        point1 = np.array((1, 2, 3))
        point2 = np.array((1, 1, 1))
        
        sum_sq = np.sum(np.square(point1 - point2))
        euclidean_distance = np.sqrt(sum_sq)
        print(f"{euclidean_distance=}")

    @Estimate.timer
    @staticmethod
    def cal_levenshtein_distance(A, B):
        N, M = len(A), len(B)
        # Create an array of size NxM
        dp = [[0 for i in range(M + 1)] for j in range(N + 1)]

        # Base Case: When N = 0
        for j in range(M + 1):
            dp[0][j] = j
        # Base Case: When M = 0
        for i in range(N + 1):
            dp[i][0] = i
        # Transitions
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                if A[i - 1] == B[j - 1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j], # Insertion
                        dp[i][j-1], # Deletion
                        dp[i-1][j-1] # Replacement
                    )
        return dp[N][M]

if __name__ == "__main__":
    t3= "It might not rain today in this afternoon."
    t4= "It might not work today in this afternoon."

    s = Similarity()
    similarity_score = s.get_similarity(t3, t4, similarity="torch")
    print(f"{similarity_score=}")

    m = Matrices.cal_cosine_similarity()
