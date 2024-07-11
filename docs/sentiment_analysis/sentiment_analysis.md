# Sentiment Analysis

Sentiment analysis is the process of analyzing text to determine if the emotional tone of the message is positive, negative, or neutral. Today, companies have large volumes of text data like emails, customer support chat transcripts, social media comments, and reviews. Sentiment analysis tools can scan this text to automatically determine the customer attitude towards a product. Companies use the insights from sentiment analysis to improve customer service and increase brand reputation.

#### ➤ Sentiment analysis in real world

For instance, consider a movie reviews.


!!! Example

    User Review 1: “I love this movie, it's fun to watch” --> positive. 
    
    User Review 2: “I have never seen a worst movie like these” --> negative.
     
    User Review 3: “I watch this movie” --> neutral.


As there are lots of data, a sentiment analysis model is crucial for identifying patterns in user reviews, as initial customer preferences may lead to a skewed perception of positive feedback. By processing a large corpus of user reviews, the model provides substantial evidence, allowing for more accurate conclusions than assumptions from a small sample of data.

➤ Use cases

- Social Media Monitoring 
  
  As businesses are growing with social media, it allows businesses to gain insights about how customers feel about their product and services or to have any issues. 

- Brand Monitoring
  
  Brand monitoring offers a wealth of insights from conversations happening about your brand from all over the internet like news articles, blogs, forums, and more to gauge brand sentiment, and target certain demographics or regions, as desired. Understanding customer feelings and opinions about how your brand image evolves over time.

- Voice of customer (VoC)
  
  Open-ended survey responses can be classified into positive and negative (and everywhere in between) offering further insights on customer support interactions, to understand the emotions and opinions of your customers. Tracking customer sentiment over time adds depth to help understand individual aspects of your business.


- Customer Service
  
  Sentiment analysis can be used to automatically organize incoming support queries by topic and urgency to route them to the correct department and make sure the most urgent are handled right away.

- Market Research
  
  To analyze online reviews of company products and compare them to market  competition. Uncover trends just as they emerge, or follow long-term market leanings through analysis of formal market reports and business journals.

  
#### ➤ Types of Sentiment  Analysis

- Fine-grained scoring

    Fine-grained sentiment analysis refers to categorizing the text intent into multiple levels of emotion. Typically, the method involves rating user sentiment on a scale of 0 to 100, with each equal segment representing very positive, positive, neutral, negative, and very negative. Ecommerce stores use a 5-star rating system as a fine-grained scoring method to gauge purchase experience. 

- Aspect-based

    Aspect-based analysis focuses on particular aspects of a product or service. For example, laptop manufacturers survey customers on their experience with sound, graphics, keyboard, and touchpad. They use sentiment analysis tools to connect customer intent with hardware-related keywords. 

- Intent-based

    Intent-based analysis helps understand customer sentiment when conducting market research. Marketers use opinion mining to understand the position of a specific group of customers in the purchase cycle. They run targeted campaigns on customers interested in buying after picking up words like discounts, deals, and reviews in monitored conversations. 

- Emotional detection

    Emotional detection involves analyzing the psychological state of a person when they are writing the text. Emotional detection is a more complex discipline of sentiment analysis, as it goes deeper than merely sorting into categories. In this approach, sentiment analysis models attempt to interpret various emotions, such as joy, anger, sadness, and regret, through the person's choice of words. 


#### ➤ Importance of Sentiment  Analysis

- Provide objective insights

    Businesses can avoid personal bias associated with human reviewers by using artificial intelligence (AI)–based sentiment analysis tools. As a result, companies get consistent and objective results when analyzing customer's opinions.

- Build better products and services

    A sentiment analysis system helps companies improve their products and services based on genuine and specific customer feedback. AI technologies identify real-world objects or entities that customers associate with negative sentiment. 

- Analyze at scale

    Businesses constantly mine information from a vast amount of unstructured data, such as emails, chatbot transcripts, surveys, customer relationship management records, and product feedback. Cloud-based sentiment analysis tools allow businesses to scale the process of uncovering customer emotions in textual data. 

- Real-time results

    Businesses must be quick to respond to potential crises or market trends in today's fast-changing landscape. Marketers rely on sentiment analysis software to learn what customers feel about the company's brand, products, and services in real time and take immediate actions based on their findings.

#### ➤ Challenges

- Sarcasm and Irony
  
  These linguistic features can completely reverse the sentiment of a statement. Detecting sarcasm and irony is a complex task even for humans, and it's even more challenging for AI systems.

- Contextual Understanding
  
  The sentiment of certain words can change based on the context in which they're used. For example, the word "sick" can have a negative connotation in a health-related context ("I'm feeling sick") but can be positive in a different context ("That's a sick beat!").

- Negations and Double Negatives
  
  Phrases like "not bad" or "not unimpressive" can be difficult to interpret correctly because they require understanding of double negatives and other linguistic nuances.

- Emojis and Slang
  
  Text data, especially from social media, often contains emojis and slang. The sentiment of these can be hard to determine as their meanings can be subjective and vary across different cultures and communities.

- Multilingual Sentiment Analysis
  
  Sentiment analysis becomes significantly more difficult when applied to multiple languages. Direct translation might not carry the same sentiment, and cultural differences can further complicate the analysis.

- Aspect-Based Sentiment Analysis
  
  Determining sentiment towards specific aspects within a text can be challenging. For instance, a restaurant review might have a positive sentiment towards the food, but a negative sentiment towards the service.


#### 👩🏻‍💻 Implementation

Simple sentiment analysis on sentences with textblob which returns polarity and subjectivity as output.

**Polarity** determines the sentiment of the text. Its values lie in [-1,1] where -1 denotes a highly negative sentiment and 1 denotes a highly positive sentiment.


**Subjectivity** determines whether a text input is factual information or a personal opinion. Its value lies between [0,1] where a value closer to 0 denotes a piece of factual information and a value closer to 1 denotes a personal opinion.


##### With Textblob

```py
!pip install textblob

from textblob import TextBlob

text_1 = "It was really good and delicious"
text_2 = "I have never seen a worst movie like these"

def sentiment(text):
    pol =  TextBlob(text).sentiment.polarity
    sub =  TextBlob(text).sentiment.subjectivity
    return [pol, sub]

result1 = sentiment(text_1)
result2 = sentiment(text_2)

print(f"{result1=}")
print(f"{result2=}")
```

**Result**

```shell
result1=[0.85, 0.8]
result2=[-1.0, 1.0]
```

##### With NLTK

```py
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sia.polarity_scores("Wow, NLTK is really powerful!")
```

**Result**

```shell
{'neg': 0.0, 'neu': 0.295, 'pos': 0.705, 'compound': 0.8012}
```