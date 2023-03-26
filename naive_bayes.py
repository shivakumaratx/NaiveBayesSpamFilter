# naive_bayes.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 3 for CS6375: Machine Learning.
# Nikhilesh Prabhakar (nikhilesh.prabhakar@utdallas.edu),
# Athresh Karanam (athresh.karanam@utdallas.edu),
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing a simple version of the
# Logistic Regression algorithm. Insert your code into the various functions
# that have the comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results.
#
# 4. Make sure to save your model in a pickle file after you fit your Naive
# Bayes algorithm.
#

'''
Members:
Erik Hale (emh170004)
Shiva Kumar(sak220007)
'''
import numpy as np
from collections import defaultdict
from collections import Counter
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string
import pprint
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import pickle


class Simple_NB():
    """
    A class for fitting the classical Multinomial Naive Bayes model that is especially suitable
    for text classifcation problems. Calculates the priors of the classes and the likelihood of each word
    occurring given a class of documents.
    """

    def __init__(self):
        # Instance variables for the class.
        self.priors = defaultdict(dict)
        self.likelihood = defaultdict(dict)
        self.columns = None

    def partition(self, x):
        """
        Partition the column vector into subsets indexed by its unique values (v1, ... vk)

        Returns a dictionary of the form
        { v1: indices of y == v1,
        v2: indices of y == v2,
        ...
        vk: indices of y == vk }, where [v1, ... vk] are all the unique values in the vector z.
        """
        # INSERT YOUR CODE HERE
        uniqueColumn = set()

        # Contains all indexs for each column
        indexDictionary = {}
        for column in range(len(x)):
            # Unique value has not been found
            if x[column] not in uniqueColumn:
                uniqueColumn.add(x[column])
                indexDictionary[x[column]] = np.array([column])
            # Add the index to the unique value
            else:
                indexDictionary[x[column]] = np.append(
                    indexDictionary[x[column]], column)

        return indexDictionary
        raise Exception('Function not yet implemented!')

    def fit(self, X, y, column_names, alpha=1):
        """
        Compute the priors P(y=k), and the class-conditional probability (likelihood param) of each feature
        given the class=k. P(y=k) is the the prior probability of each class k. It can be calculated by computing
        the percentage of samples belonging to each class. P(x_i|y=k) is the number of counts feature x_i occured
        divided by the total frequency of terms in class y=k.

        The parameters after performing smooothing will be represented as follows.
            P(x_i|y=k) = (count(x_i|y=k) + alpha)/(count(x|y=k) + |V|*alpha)
            where |V| is the vocabulary of text classification problem or
            the size of the feature set

        :param x: features
        :param y: labels
        :param alpha: smoothing parameter

        Compute the two class instance variable
        :param self.priors: = Dictionary | self.priors[label]
        :param self.likelihood: = Dictionary | self.likelihood[label][feature]

        """
        # INSERT CODE HERE
        # Last column is prediction which is empty
        self.columns = column_names[:-1]

        # Creates Dictionary of all y_labels and its indexs
        y_labels = self.partition(y)  # Replace None

        # Tip: Add an extra key in your likelihood dictionary for unseen data. This will be used when testing sample texts
        # that contain words not present in feature set
        for label in y_labels:
            self.likelihood[label]["__unseen__"] = None  # Enter Code Here

            for column in self.columns:
                self.likelihood[label][column] = 0

        # INSERT YOUR CODE HERE

        # Priors
        for index, label in enumerate(y_labels):
            # P(y=k)
            self.priors[label] = len(y_labels[index]) / len(y)

        # Likelihood
        # For every feature of X compute the likihood probability which is
        # P(x_i | y=k)* P(y=k)
        for label in y_labels:

            # y_Values represents all the indexes for current label
            y_Values = y_labels[label]

            # Represents all counts of every column for given label
            totallabelCount = 0
            for i in range(len(self.columns)):
                column = self.columns[i]
                # Puts all the column values in an array

                LikelihoodProbabilityDictionary = defaultdict(list)
                # Updating total count of each column given the label
                totalColumnCount = 0
                for index in y_Values:

                    totalColumnCount += X[index][i]
                # Update totallabelCount
                totallabelCount += totalColumnCount
                # Updates [label][column] with count
                self.likelihood[label][column] = totalColumnCount

            for i in range(len(self.columns)):
                column = self.columns[i]
                # Smoothing Probability
                self.likelihood[label][column] = (
                    (self.likelihood[label][column] + alpha) / (totallabelCount + (len(self.columns)*alpha) + len(y_labels)))

            # Unseen Probability with Smoothing
            '''Unseen data with is equally likely to be in any column'''
            self.likelihood[label]["__unseen__"] = 1 / \
                (len(y_Values) + len(self.columns)*alpha + 2)

        return
        raise Exception('Function not yet implemented!')

    def predict_example(self, x, sample_text=False, return_likelihood=False):
        """
        Predicts the classification label for a single example x by computing the posterior probability
        for each class value, P(y=k|x) = P(x_i|y=k)*P(y=k).
        The predicted class will be the argmax of P(y=k|x) over all the different k's,
        i.e. the class that gives the highest posterior probability
        NOTE: Converting the probabilities into log-space would help with any underflow errors.

        :param x: example to predict
        :return: predicted label for x
        :return: posterior log_likelihood of all classes if return_likelihood=True
        """

        if sample_text:

            # Convert list of words to a term frequency dictionary
            frequencyDictionary = {}

            for word in x:

                # Updating frequency
                if word is not frequencyDictionary.keys():
                    frequencyDictionary[word] = 1
                else:
                    frequencyDictionary[word] += 1

            posteriorLikelihoodNoSpam = 0
            NoSpamLogLikelihood = 0
            posteriorLikelihoodSpam = 0
            SpamLogLikelihood = 0
            for word in frequencyDictionary.keys():
                if word not in self.likelihood[1].keys():
                    posteriorLikelihoodSpam += self.likelihood[1]["__unseen__"]
                    SpamLogLikelihood += np.log(
                        self.likelihood[1]["__unseen__"])
                    posteriorLikelihoodNoSpam += self.likelihood[0]["__unseen__"]
                    NoSpamLogLikelihood += np.log(
                        self.likelihood[0]["__unseen__"])
                else:
                    posteriorLikelihoodSpam += self.likelihood[1][word]
                    SpamLogLikelihood += np.log(
                        self.likelihood[1][word])

                    posteriorLikelihoodNoSpam += self.likelihood[0][word]
                    NoSpamLogLikelihood += np.log(
                        self.likelihood[0][word])

            # Updating Posterior with prior Probability
            posteriorLikelihoodNoSpam *= self.priors[0]
            NoSpamLogLikelihood += np.log(self.priors[0])
            posteriorLikelihoodSpam *= self.priors[1]
            SpamLogLikelihood += np.log(self.priors[1])

        # INSERT YOUR CODE HERE

        # If sample Text Equals False
        else:

            y_pred = []

            # Will keep track of the class-conditional probabilities for spam
            Spam = {}
            # Will keep track of the class-conditional probabilities for no spam
            NoSpam = {}

            # Initialize all column probabilities
            for column in self.columns:
                Spam[column] = 0
                NoSpam[column] = 0

            for row in x:
                # Will represent the classConditional for No Spam
                classConditionalLabel0 = 0
                # Will represent the classConditional for Spam
                classConditionalLabel1 = 0

                for index in range(len(self.columns)):
                    column = self.columns[index]
                    if row[index] != 0:
                        NoSpamvalue = (self.likelihood[0][column])

                        NoSpam[column] = NoSpamvalue
                        classConditionalLabel0 += np.log(NoSpamvalue)

                        SpamValue = self.likelihood[1][column]
                        Spam[column] = SpamValue
                        classConditionalLabel1 += np.log(SpamValue)

                for column in self.columns:
                    NoSpam[column] *= self.priors[0]
                    Spam[column] *= self.priors[1]
                # Add prior probabilities
                classConditionalLabel0 += np.log(self.priors[0])
                classConditionalLabel1 += np.log(self.priors[1])
                # If log odd Ratio is greather than or equal to 0 then we classify as Spam
                # Else: Classify as Not Spam

                if (classConditionalLabel1) - (classConditionalLabel0) >= 0:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

        # Return posterior log_likelihood of all classes
        if (return_likelihood == True):
            # Find top three words that have the highest class-conditional likelihood for Spam
            ThreeMostCommonSpam = Counter(Spam).most_common(3)

            # Find top three words that have the highest class-conditional likelihood for No Spam
            ThreeMostCommonNoSpam = Counter(NoSpam).most_common(3)

            print("For Spam, the Top three Most Words are: ")
            for likelihood in ThreeMostCommonSpam:
                print("Word: ", likelihood[0])
                print("Class-conditional likelihoods", likelihood[1])
                print("Log-Likelihood", np.log(likelihood[1]))
                print("\n")
            print("For No-Spam, the Top three Most Words are: ")
            for Spamlikelihood in ThreeMostCommonNoSpam:
                print("Word: ", Spamlikelihood[0])
                print("Class-conditional likelihoods", Spamlikelihood[1])
                print("Log-Likelihood", np.log(Spamlikelihood[1]))
                print("\n")
            return y_pred

        # If return_likelihood == False then return predicted label for x

        else:
            # Print Results
            print("Posterior likelihood for Spam: ", posteriorLikelihoodSpam)

            print("Posterior likelihood for Not Spam: ",
                  posteriorLikelihoodNoSpam)
            print("Log Posterior Likelihood for Spam ", SpamLogLikelihood)
            print("Log Posterior Likelihood for Not Spam ", NoSpamLogLikelihood)
            # If Spam Log Posterior is greather, then predict Spam
            if (SpamLogLikelihood) > (NoSpamLogLikelihood):
                return 1
            # If noSpam log Posterior is greather than predict No Spam
            else:
                return 0
        raise Exception('Function not yet implemented!')


def compute_accuracy(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)
    :param y_true: true label
    :param y_pred: predicted label
    :return: error rate = (1/n) * sum(y_true!=y_pred)
    """
    # INSERT YOUR CODE HERE
    return (1/len(y_true)) * sum(y_true != y_pred)
    raise Exception('Function not yet implemented!')


def compute_precision(y_true, y_pred):
    """
    Computes the precision for the given set of predictions.
    Precision gives the proportion of positive predictions that are actually correct.
    :param y_true: true label
    :param y_pred: predicted label
    :return: precision
    """
    # INSERT YOUR CODE HERE
    # Precision = TP / (TP + FP)
    TruePositiveCount = 0
    FalsePositiveCount = 0
    for i in range(len(y_pred)):
        if (y_pred[i] == 1):
            # y_pred and y_true are both 1
            if (y_pred[i] == y_true[i]):
                TruePositiveCount += 1
            # y_pred = 1 and y_true = 0
            else:
                FalsePositiveCount += 1
    TP = TruePositiveCount / len(y_true)
    FP = FalsePositiveCount / len(y_true)

    # There are no predicted 1
    if (TP+FP) == 0:
        return 0
    return TP / (TP + FP)

    raise Exception('Function not yet implemented!')


def compute_recall(y_true, y_pred):
    """
    Computes the recall for the given set of predictions.
    Recall measures the proportion of actual positives that were predicted correctly.
    :param y_true: true label
    :param y_pred: predicted label
    :return: recall

    """
    # INSERT YOUR CODE HERE
    # Recall: TP = TP + FN

    TruePositiveCount = 0
    FalseNegativeCount = 0
    for i in range(len(y_pred)):
        if (y_pred[i] == 1):
            # y_pred[i] and y_true[i] are both 1
            if (y_pred[i] == y_true[i]):
                TruePositiveCount += 1
        # y_pred[i]= 0
        else:
            # y_pred[i]= 0 and y_true[i] = 1
            if (y_pred[i] != y_true[i]):
                FalseNegativeCount += 1
    TP = TruePositiveCount / len(y_true)
    FN = FalseNegativeCount / len(y_true)

    return TP/(TP + FN)
    raise Exception('Function not yet implemented!')


def compute_f1(y_true, y_pred):
    """
    Computes the f1 score for a given set of predictions.
    F1 score is defined as the harmonic mean of precision and recall.
    :param y_true: true label
    :param y_pred: predicted label
    :return: f1 = 2 * (P*R)/(P+R)
    """
    # INSERT YOUR CODE HERE

    # Call Compute_Precision()
    Precision = compute_precision(y_true, y_pred)

    # Calculate compute_recall()
    Recall = compute_recall(y_true, y_pred)

    # P*R
    numerator = Precision * Recall

    # P+R
    denominator = Precision + Recall

    # 2 * (P*R)/(P+R)
    return 2 * (numerator/denominator)
    raise Exception('Function not yet implemented!')


if __name__ == '__main__':

    df_train = pd.read_csv("data/train_email.csv")
    df_train.drop(df_train.columns[0], inplace=True, axis=1)

    df_test = pd.read_csv("data/test_email.csv")
    df_test.drop(df_test.columns[0], inplace=True, axis=1)

    X_columns = df_train.columns
    print(len(X_columns))
    print(df_train.shape)

    Xtrn = np.array(df_train.iloc[:, :-1])
    ytrn = np.array(df_train.iloc[:, -1])

    Xtst = np.array(df_test.iloc[:, :-1])
    ytst = np.array(df_test.iloc[:, -1])
    results = {}  # To Store All Results
    
    # PART A
    NB = Simple_NB()
    NB.fit(Xtrn, ytrn, column_names=X_columns, alpha=1)

    # Prediction on Test Set

    # INSERT CODE HERE
    test_pred = NB.predict_example(Xtst, False, True)
    tst_acc = compute_accuracy(ytst, test_pred)
    tst_precision = compute_precision(ytst, test_pred)
    tst_recall = compute_recall(ytst, test_pred)
    tst_f1 = compute_f1(ytst, test_pred)
    results["Simple Naive Bayes"] = {"accuracy": tst_acc,
                                     "precision": tst_precision,
                                     "recall": tst_recall,
                                     "f1_score": tst_f1,
                                     }
    print(results["Simple Naive Bayes"])
    print("\n")
    # PART B - Testing on Sample Text

    sample_email = ["Congratulations! Your raffle ticket has won yourself a house. Click on the link to avail prize",  # SAMPLE EMAIL 1
                    "Hello. This email is to remind you that your project needs to be submitted this week",  # SAMPLE EMAIL 2
                    # " TRY OUT A SAMPLE EMAIL(S) HERE AND REPORT RESULTS"
                    "Congrats! Click Link for a New Car ! ",  # Sample Email 3
                    "Thanks! I forwarded the attached file to my boss. "  # Sample Email 4
                    ]

    for sample in sample_email:
        words = None  # INSERT CODE HERE (USE NLTK LIBRARY)
        # INSERT CODE HERE (MAKE SURE TO LOWERCASE THE WORDS AND REMOVE NUMBERS)

        # Lower case each word
        words = sample.lower()
        # Tokenize the sentence
        words = word_tokenize(words)

        # Creates a list of tokens based of the lower case sample sentence
        for word in words:
            # If the word is not a letter than remove the word
            if word.isalpha() == False:
                word = words.remove(word)

        y_sent_pred = NB.predict_example(words, sample_text=True)
        print("Sample Email: {} \nIs Spam".format(sample)) if y_sent_pred else print(
            "Sample Text: {} \n Is Not Spam".format(sample))
        print("\n")
    
    # PART C - Compare with Sklearn's NB Models
    # Replace Nones with the respective Sklearn library.
    models = {
        # MODEL NAME : MODEL_SKLEARN_LIBRARY
        "Gaussian NB": GaussianNB(),
        "Multinomial NB": MultinomialNB(),
        "Bernoulli NB": BernoulliNB(),
        "Our MNB": Simple_NB()
    }

    for model_name, sk_lib in models.items():

        model = sk_lib
        if (model_name != "Our MNB"):
            model.fit(Xtrn, ytrn)
        else:
            model.fit(Xtrn, ytrn, column_names=X_columns, alpha=1)

        # Predict the target values for test set
        if (model_name != "Our MNB"):
            y_pred = model.predict(Xtst)
        else:
            y_pred = model.predict_example(Xtst, False, True)

        # Evaluate the Models with the different metrics
        # INSERT CODE HERE
        accuracy = compute_accuracy(ytst, y_pred)
        precision = compute_precision(ytst, y_pred)
        recall = compute_recall(ytst, y_pred)
        f1 = compute_f1(ytst, y_pred)
        print(model_name, ": a=", accuracy, " p=",
              precision, " r=", recall, " f1=", f1)

        results[model_name] = {"accuracy": accuracy,
                               "precision": precision,
                               "recall": recall,
                               "f1_score": f1
                               }

    pprint.pprint(results)

    # PART D - Visualize the model using bar charts
    # INSERT  CODE HERE
    # We need to add a
    barWidth = 0.15
    fig = plt.subplots()

    acc = []
    pre = []
    rec = []
    f1s = []
    for k in list(results.keys()):
        acc.append(results[k]["accuracy"])
        pre.append(results[k]["precision"])
        rec.append(results[k]["recall"])
        f1s.append(results[k]["f1_score"])

    # Set the position of the bar on the x axis (for accuracy, precision, recall, and f1)
    bar1 = np.arange(len(acc))
    #print(len(acc))
    bar2 = [x + barWidth for x in bar1]
    bar3 = [x + barWidth for x in bar2]
    bar4 = [x + barWidth for x in bar3]

    # Plot the bar
    plt.bar(bar1, acc, color='r', width=barWidth,
            edgecolor='black', label="Accuracy")
    plt.bar(bar2, pre, color='b', width=barWidth,
            edgecolor='black', label="Precision")
    plt.bar(bar3, rec, color='g', width=barWidth,
            edgecolor='black', label="Recall")
    plt.bar(bar4, f1s, color='y', width=barWidth,
            edgecolor='black', label="F1 Score")

    plt.xticks([r + barWidth for r in range(len(acc))],
               list(results.keys()))

    plt.legend()
    plt.title("Naive Bayes Evaluation")
    plt.xlabel("Model")
    plt.ylabel("Precentage")
    plt.show()
    # Members:
    # Erik Hale (emh170004)
    # Shiva Kumar(sak220007)
    # PART E - Save the Model
    # Code to store as pickle file
    netid = 'emh170004_sak220007'
    # file_pi = open('{}_model_3.obj', format(netid), 'wb')  # Use your NETID
    file_pi = open('emh170004_sak220007_model_3.obj', 'wb')
    pickle.dump(NB, file_pi)
