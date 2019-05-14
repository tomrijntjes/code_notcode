import pickle
from nltk.tokenize import word_tokenize

with open('/code/pickled_models/NaiveBayesClassifier_200.pickle', 'rb') as f:
    classifier = pickle.load(f)

print("loaded classifier")

with open('/code/pickled_models/tokens_200.pickle', 'rb') as f:
    all_words = pickle.load(f)

print("loaded features")

with open('/landing/0430.txt') as f:
    for line in f.readlines():
        test_sent_features = {word: (word in word_tokenize(line.lower())) for word in all_words}
        print("{0}: {1}".format(classifier.classify(test_sent_features), line))
