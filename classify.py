import pickle
from nltk.tokenize import word_tokenize

with open('pickled_models/NaiveBayesClassifier_100.pickle', 'rb') as f:
    classifier = pickle.load(f)

with open('pickled_models/tokens_100.pickle', 'rb') as f:
    all_words = pickle.load(f)


test_sentence = "This is the best band I've ever heard!"
test_code = "args = util.parse_args(MODULES)"
test_sent_features = {word: (word in word_tokenize(test_sentence.lower())) for word in all_words}
test_code_features = {word: (word in word_tokenize(test_code.lower())) for word in all_words}
print(classifier.classify(test_sent_features))
print(classifier.classify(test_code_features))
