import pandas as pd
import nltk
import sklearn
import sklearn_crfsuite
import scipy.stats
import math

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from itertools import chain
from sklearn.preprocessing import MultiLabelBinarizer

punctuation = ["!", "\"", "#", "$", "%", "&",
"\'", "(", ")", "*", "+", ",", "-", ".", "/", 
":", ";", "<", "=", ">", "?", "@", "[", "\\",
"]", "^", "_", "`", "{", "|", "}", "~", "–",
"—", "«", "»"]

def word2features(sent, i):
    word = sent[i][0]
    features = {
        "bias": 1.0,
        "word": word,
        "length": len(word), 
        "prefix4": word[:4],
        "prefix3": word[:3],
        "prefix2": word[:2],
        "suffix2": word[-2:],
        "suffix3": word[-3:],
        "suffix4": word[-4:],
        "lowercase": word.lower(),
        "title": word.istitle(),
        "punctuation": (word in punctuation),
        "digit": word.isdigit(),
    }

    if i > 0:
        word1 = sent[i-1][0] # info of the preceding word
        features.update({
            "-1:word": word1,
            "-1:length": len(word1),
            "-1:prefix3": word1[:3],
            "-1:prefix2": word1[:2],
            "-1:suffix2": word1[-2:],
            "-1:suffix3": word1[-3:],
            "-1:lowercase": word1.lower(),
            "-1:title": word1.istitle(),
            "-1:punctuation": (word1 in punctuation),
            "-1:digit": word.isdigit(),
        })
    else:
        features["BOS"] = True
    
    if i > 2:
        word2 = sent[i-2][0] # trigram
        features.update({
            "-2:word": word2,
            "-2:length": len(word2),
            "-2:prefix3": word2[:3],
            "-2:prefix2": word2[:2],
            "-2:suffix2": word2[-2:],
            "-2:suffix3": word2[-3:],
            "-2:lowercase": word2.lower(),
            "-2:title": word2.istitle(),
            "-2:punctuation": (word2 in punctuation),
            "-2:digit": word2.isdigit(),
        })
    
    if i < len(sent) - 1: # if not the last word
        word1 = sent[i+1][0]
        features.update({
            "+1:word": word1,
            "+1:length": len(word1),
            "+1:prefix3": word1[:3],
            "+1:prefix2": word1[:2],
            "+1:suffix2": word1[-2:],
            "+1:suffix3": word1[-3:],
            "+1:lowercase": word1.lower(),
            "+1:title": word1.istitle(),
            "+1:punctuation": (word1 in punctuation),
            "+1:digit": word1.isdigit(),
        })
    else:
        features["EOS"] = True
    
    if i < len(sent) - 2: # trigram
        word2 = sent[i+2][0]
        features.update({
            "+2:word": word2,
            "+2:length": len(word2),
            "+2:prefix3": word2[:3],
            "+2:prefix2": word2[:2],
            "+2:suffix2": word2[-2:],
            "+2:suffix3": word2[-3:],
            "+2:lowercase": word2.lower(),
            "+2:title": word2.istitle(),
            "+2:punctuation": (word2 in punctuation),
            "+2:digit": word2.isdigit(),
        })
    
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [word[1] for word in sent]

def sent2tokens(sent):
    return [word[0] for word in sent]

def format_data(csv_data):
    sents = []
    for i in range(len(csv_data)):
        if math.isnan(csv_data.iloc[i, 0]):
            continue
        elif int(csv_data.iloc[i, 0]) == 1:
            sents.append([[csv_data.iloc[i, 1], csv_data.iloc[i, 2]]])
        else:
            sents[-1].append([csv_data.iloc[i, 1], csv_data.iloc[i, 2]])
    for sent in sents:
        for i, word in enumerate(sent):
            if type(word[0]) != str:
                del sent[i]
    return sents

# convert to readable data
train = input("Specify the training data:\n")
assert train.endswith(".csv"), "Invalid file name."
test = input("Specify the test data. Enter if the file name is default:\n")
if test == "":
    test = train[:-9] + "test.csv"
assert train.endswith("csv"), "Invalid file name."
data = {}
data["train"] = pd.read_csv(train, index_col=0)
data["test"] = pd.read_csv(test, index_col=0)

train_sents = format_data(data["train"])
test_sents = format_data(data["test"])

Xtrain = [sent2features(s) for s in train_sents]
ytrain = [sent2labels(s) for s in train_sents]

Xtest = [sent2features(s) for s in test_sents]
ytest = [sent2labels(s) for s in test_sents]

def train():
    crf = sklearn_crfsuite.CRF(
        algorithm = "lbfgs", # Limited-memory BFGS
        c1 = 0.25, # L1 regularization parameter
        c2 = 0.3, # L2 regularization parameter
        max_iterations = 100,
        all_possible_transitions = True
    )
    crf.fit(Xtrain, ytrain)
    return crf

def train_metrics(crf):
    ypred = crf.predict(Xtrain)
    print("F1 score on the train set = {}\n".format(
        metrics.flat_f1_score(ytrain, ypred,
        average="weighted", labels=labels)))
    print("Accuracy on the train set = {}\n".format(
        metrics.flat_accuracy_score(ytrain, ypred)))
    
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )

    print("Train set classification report:\n\n{}".format(
        metrics.flat_classification_report(
            ytrain, ypred, labels=sorted_labels, digits=3)))

def test_metrics(crf):
    ypred = crf.predict(Xtest)
    print("F1 score on the test set = {}\n".format(
        metrics.flat_f1_score(ytest, ypred,
        average="weighted", labels=labels)))
    print("Accuracy on the test set = {}\n".format(
        metrics.flat_accuracy_score(ytest, ypred)))
    
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )

    print("Test set classification report:\n\n{}".format(
        metrics.flat_classification_report(ytest, ypred,
        labels=sorted_labels, digits=3)))

if __name__ == "__main__":
    crf = train()
    labels = list(crf.classes_)
    # labels.remove("SCONJ")
    # labels.remove("DET")
    train_metrics(crf)
    test_metrics(crf)
