import random
import glob
import pickle
from nltk.tokenize import word_tokenize
from nltk import NaiveBayesClassifier


with open('/data/notcode/webtext/overheard.txt',encoding="utf-8") as f:
    overheard = f.read().splitlines()


overheard = [line.split(':')[1] for line in overheard if len(line)>0 and ':' in line]
notcode = [ (line,'notcode') for line in overheard]



filenames = glob.iglob('/usr/lib/python3/dist-packages/nltk/*.py')
files = (open(f,encoding="utf-8") for f in filenames)
lines = (line.rstrip('\n') for f in files for line in f)
code = [(line,'code') for line in lines if len(line)>0]
print(len(code))
print(len(notcode))


train = random.sample(code, 500)+random.sample(notcode, 500)
print(len(train))

all_words = set(word.lower() for passage in train for word in word_tokenize(passage[0]))
t = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in train]
print("done tokenizing")


classifier = NaiveBayesClassifier.train(t)
classifier.show_most_informative_features()

with open('/code/pickled_models/tokens_{0}.pickle'.format(len(train)), 'wb') as f:
    pickle.dump(all_words, f)

with open('/code/pickled_models/NaiveBayesClassifier_{0}.pickle'.format(len(train)), 'wb') as f:
    pickle.dump(classifier, f)
