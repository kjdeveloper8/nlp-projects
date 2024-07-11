from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from helper.util import Estimate

training_data = [
     ('i want to buy a jeans pent', 'Buy_a_product'),
     ('i want to purchase a pair of shoes', 'Buy_a_product'),
     ('are you selling laptops', 'Buy_a_product'),
     ('i need an apple jam', 'Buy_a_product'),
     ('can you please tell me the price of this product', 'Buy_a_product'),
     ('please give me some discount.', 'negotition'),
     ("i cannot afford such price", 'negotition'),
     ("could you negotiate", "negotition"),
     ("i agree on your offer", "success"),
     ("yes i accepcted your offer", "success"),
     ("offer accepted", "success"),
     ("agreed", "success"),
     ("what is the price of this watch", "ask_for_price"),
     ("How much it's cost", "ask_for_price"),
     ("i will only give you 3000 for this product", "counter_offer"),
     ("Its too costly i can only pay 1500 for it", "counter_offer"),
]

test_data = [
    'your offer',
    'it will cost you 10000',
    'i need chocolate',
    'i want to buy t-shirts',
    'does it cost 30',
]

@Estimate.timer
def intent_detection_nb(training_data):
    """ Intent Detection with Naivebayes. """
    vectorizer = TfidfVectorizer()
    data = [t[0] for t in training_data]
    intent = [t[1] for t in training_data]
    # print(data,"-->", intent)

    # fit_transform(): learn the params and applied transformation on data
    X = vectorizer.fit_transform(data)     
    # transform(): apply learned transformation on data
    Y = vectorizer.transform(test_data)    

    clf = MultinomialNB()
    # fit(): learn and estimate(calculate mean and std) params of data
    clf.fit(X, intent)              
    result = clf.predict(Y) # test
    # prob = clf.predict_proba(Y)
    acc = clf.score(X, intent) # train
    return result, acc

if __name__ == "__main__":
    prediction, accuracy = intent_detection_nb(training_data=training_data)
    print(f"{prediction=}")
    print(f"{accuracy=}")
