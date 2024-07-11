# Text Summarization

Text summarization refers to a group of methods that employ algorithms to compress a certain amount of text while preserving the it's key points.  Systems capable of extracting the key concepts from the text while maintaining the overall meaning have the potential to revolutionize a variety of industries, including banking, law, and even healthcare.

➤ Approach 

 - Extractive Summarization 
 - Abstractive Summarization

#### ➤ Extractive Summarization

Extractive approach is simple and uses traditional algorithms. For example, If we want to summarize our text on the basis of the frequency method, for that we will store all the unique words and frequency of all those words in the dictionary. On the basis of high frequency words, we store the sentences containing that word in our final summary. This means the words which are in our summary confirm that they are part of the given text.


#### ➤ Abstractive Summarization

Abstractive summarization techniques emulate human writing by generating entirely new sentences to convey key concepts from the source text, rather than rephrasing portions of it. These fresh sentences distill the vital information while eliminating irrelevant details, often incorporating novel vocabulary absent in the original text. It understands the meaning and context of the text and then generates the summary. It requires a deeper understanding of the content and the ability to generate new text without changing the meaning of the source information.


#### 👩🏻‍💻 Implementation

With pytextrank

```py
!pip install pytextrank -q
```

Import and load model

```py
import spacy
import pytextrank
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")
```

Sample text

```py
sample_text = """ Deep learning (also known as deep structured learning) is part of a 
broader family of machine learning methods based on artificial neural networks with 
representation learning. Learning can be supervised, semi-supervised or unsupervised. 
Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, 
recurrent neural networks and convolutional neural networks have been applied to
fields including computer vision, speech recognition, natural language processing, 
machine translation, bioinformatics, drug design, medical image analysis, material
inspection and board game programs, where they have produced results comparable to 
and in some cases surpassing human expert performance. Artificial neural networks
(ANNs) were inspired by information processing and distributed communication nodes
in biological systems. ANNs have various differences from biological brains. Specifically, 
neural networks tend to be static and symbolic, while the biological brain of most living organisms
is dynamic (plastic) and analogue. The adjective "deep" in deep learning refers to the use of multiple
layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, 
but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can.
Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, 
which permits practical application and optimized implementation, while retaining theoretical universality 
under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely 
from biologically informed connectionist models, for the sake of efficiency, trainability and understandability, 
whence the structured part.
"""
```

Summary

```py
for text in doc._.textrank.summary(limit_phrases=2, limit_sentences=2):
    print(text)
    print('Summary Length:',len(text))
```

**Result**

```shell
recurrent neural networks and convolutional neural networks have been applied to
fields including computer vision, speech recognition, natural language processing, 
machine translation, bioinformatics, drug design, medical image analysis, material
inspection and board game programs, where they have produced results comparable to 
and in some cases surpassing human expert performance.
Summary Length: 81
The adjective "deep" in deep learning refers to the use of multiple
layers in the network.
Summary Length: 20
```