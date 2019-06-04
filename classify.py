import pickle
import glob
from nltk.tokenize import word_tokenize

with open('/code/pickled_models/NaiveBayesClassifier_200.pickle', 'rb') as f:
    classifier = pickle.load(f)

print("loaded classifier")

with open('/code/pickled_models/tokens_200.pickle', 'rb') as f:
    all_words = pickle.load(f)

print("loaded features")

filenames = glob.iglob('/landing/*.txt')
for fn in filenames:
    print("[+] Parsing {0}".format(fn))
    f_in = open(fn, encoding="utf-8")
    f_code_out = open("/processed/{0}-code".format(fn.split('/')[-1]),mode='w+',encoding='utf-8')
    f_notcode_out = open("/processed/{0}-notcode".format(fn.split('/')[-1]),mode='w+',encoding='utf-8')
    for line in f_in.readlines():
        test_sent_features = {word: (word in word_tokenize(line.lower())) for word in all_words}
        label = classifier.classify(test_sent_features)
        #print("{0}: {1}".format(label, line))
        if label =='code':
            f_code_out.write(line)
        elif label == 'notcode':
            f_notcode_out.write(line)
        else:
            raise ValueError(line)
    f_in.close()
    f_code_out.close()
    f_notcode_out.close()
