import numpy as np
import regex
from nltk.corpus import brown
from operator import itemgetter
import nltk
import json
import re

NUMER_REGEX = r"\d+"

MONEY_REGEX = r"\d+[€$¥₪₹]"

TIME_REGEX = r"^(?:[01]?\d|2[0-3])(?::[0-5]\d){1,2}$"

DATE_REGEX = r"\d{1,2}-\d{1,2}-\d{2,4}|\d{1,2}\/\d{1,2}\/\d{2,4}|\d{1,2}\.\d{1,2}\.\d{2,4}"

SPARSE_WORD_THRESHOLD = 5

nltk.download('brown')

FILTER_COMPLEX_TAG = re.compile(r"[^*+\-]*")


def filter_tag(tag):
    return FILTER_COMPLEX_TAG.match(tag).group(0)


def split_train_test():
    sents = list(brown.tagged_sents(categories="news"))
    split_size = (len(sents) // 10) + 1
    train = sents[:-split_size]
    test = sents[-split_size:]
    return train, test


class Bigram_HMM:
    def __init__(self, training, test, smoothing=False):
        self.known_words = set()
        self.unknown_words = set()
        self.train_emission = self.init_emission(training)
        self.train_transition = self.init_transition(training)
        self.all_w = self.known_words.copy()
        self.smoothing = smoothing
        self.posTagger = posTagger(train, test)

        if self.smoothing:
            self.add_test(test)
            self.smooth()

    def init_transition(self, data):
        transition = dict()
        for sent in data:
            last_tag = "START"
            for w, tag in sent:
                t = filter_tag(tag)
                inner = transition.setdefault(last_tag, dict())
                inner[t] = inner.get(t, 0) + 1
                last_tag = t
                self.known_words.add(w)
            inner = transition.setdefault(last_tag, dict())
            inner["STOP"] = inner.get("STOP", 0) + 1
        return transition

    def init_emission(self, data):
        words_tag = dict()
        for sent in data:
            for word in sent:
                w_tag = filter_tag(word[1])
                if word[0] in words_tag:
                    if w_tag in words_tag[word[0]]:
                        words_tag[word[0]][w_tag] += 1
                    else:
                        words_tag[word[0]][w_tag] = 1
                else:
                    words_tag[word[0]] = dict()
                    words_tag[word[0]][w_tag] = 1
        return words_tag

    def add_test(self, test):
        for sentence in test:
            for word, tag in sentence:
                self.all_w.add(word)

    def smooth(self):
        for w in self.all_w:
            inner = self.train_emission.setdefault(w, dict())
            for t in self.train_transition.keys():
                inner[t] = inner.get(t, 0) + 1

    def check_known_word(self, word):
        return word in self.known_words

    def calc_transition(self, tag1, tag2):
        if tag1 not in self.train_transition:
            return 0
        count_tag1_tag2 = self.train_transition[tag1].get(tag2, 0)
        return count_tag1_tag2 / sum(self.train_transition[tag1].values())

    def calc_emission(self, t, w):
        if not self.check_known_word(w):
            return -1
        c = self.train_emission[w].get(t, 0)  # amount of occurrences the word w is tagged to t
        d = len(self.all_w) if self.smoothing else 0
        return c / (sum(self.train_transition[t].values()) + d)

    def calc_prev_max_tag(self, all_tags, matrix, i, tag):
        max_prob = 0
        index = 0
        for j, u in enumerate(all_tags):
            p = matrix[i][j][0] * self.calc_transition(u, tag)
            if p > max_prob:
                max_prob = p
                index = j
        list_of_tags = matrix[i][index][1].copy()
        list_of_tags.append(tag)
        return max_prob, list_of_tags

    def calc_Viterbi(self, sent):
        k = len(sent) + 1
        num_of_tags = len(self.train_transition)
        all_tags = list(self.train_transition.keys())
        viterbi_matrix = []
        text = ['']
        text.extend(sent)
        for i, w in enumerate(text):
            viterbi_matrix.append([(0, [])] * num_of_tags)
            if i == 0:
                viterbi_matrix[i][0] = (1.0, ["START"])
            elif not self.check_known_word(w):  # unknown word
                self.unknown_words.add(w)
                p, tags = self.calc_prev_max_tag(all_tags, viterbi_matrix, i - 1, "NN")
                viterbi_matrix[i][3] = (p, tags)
            else:  # known word
                for j, t in enumerate(all_tags):
                    e = self.calc_emission(t, w)
                    if e == 0: continue
                    p, tags = self.calc_prev_max_tag(all_tags, viterbi_matrix, i - 1, t)
                    viterbi_matrix[i][j] = (e * p, tags)
            if sum([g[0] for g in viterbi_matrix[i]]) == 0:
                tags = max(viterbi_matrix[i - 1], key=itemgetter(0))[1]
                tags.extend(["NN"] * (k - i))
                return tags[1:]
        predicted_tags = max(viterbi_matrix[k - 1], key=lambda item: item[0])
        return predicted_tags[1][1:]

    def viterbi_error(self, data, realData = None):
        test_size = 0
        correct = 0
        unknown_correct = 0
        unknown_words = 0
        known_words = 0
        for j, sentence in enumerate(data):
            sent, tags = [list(i) for i in zip(*sentence)]
            tags = [filter_tag(t) for t in tags]
            predicted_tags = self.calc_Viterbi(sent)
            for i, word in enumerate(sent):

                if realData:  #case of pseudo words - we need to check if a word is unknown or not with the real data instead of pseudo data
                    if not (self.check_known_word(realData[j][i][0])):
                        unknown_words += 1
                        if tags[i] == predicted_tags[i]:  # correct guess for unknown word
                            unknown_correct += 1
                    else:
                        known_words += 1
                        if tags[i] == predicted_tags[i]:
                            correct += 1

                elif not self.check_known_word(word):
                    unknown_words += 1
                    if tags[i] == predicted_tags[i]:  # correct guess for unknown word
                        unknown_correct += 1
                else:
                    known_words += 1
                    if tags[i] == predicted_tags[i]:
                        correct += 1
            test_size += len(sent)
        accuracy = (correct + unknown_correct) / test_size
        unknown_accuracy = unknown_correct / unknown_words
        known_accuracy = correct / known_words
        print(f"Overall error: {1 - accuracy}\n"
              f"Unknown words error: {1 - unknown_accuracy}\n"
              f"Known words error: {1 - known_accuracy}")
        return 1 - accuracy


class posTagger:

    def __init__(self, training, test, smoothing=False, file=None):
        self.train_words_tags = self.init_word_tags(training)
        self.test_words_tags = self.init_word_tags(test)
        self.train_words_mle = self.init_mle(self.train_words_tags) if file is None else self.get_mle_from_json(file)

    def smoothing(self):
        pass

    def get_words_tag_count(self, word, tag):
        if not self.check_known_word(word):
            return -1
        return self.train_words_tags[word].get(tag, 0)

    def check_known_word(self, word):
        return word in self.train_words_tags
        # if word not in self.train_words_tags:
        #     return False
        # return True

    def init_word_tags(self, data):
        """
        create a dict of all the unique words in the data set along with a count of each of their tags
        :param data: a list of tuples of words and their tag.
        :return: a dict of words and the value is dict of each tag and the count in the parm 'data'
        """
        words_tag = dict()
        for sent in data:
            for word in sent:
                w_tag = filter_tag(word[1])
                if word[0] in words_tag:
                    if w_tag in words_tag[word[0]]:
                        words_tag[word[0]][w_tag] += 1
                    else:
                        words_tag[word[0]][w_tag] = 1
                else:
                    words_tag[word[0]] = dict()
                    words_tag[word[0]][w_tag] = 1
        return words_tag

    def init_mle(self, data):
        words_mle = dict()
        for word in data:
            words_mle[word] = max(data[word], key=data[word].get)
        return words_mle

    def save_mle(self):
        with open("mle_train_data.json", 'w') as f:
            json.dump(self.train_words_mle, f)

    def get_mle_from_json(self, file):
        with open(file, "r") as f:
            self.train_words_mle = json.load(f)

    def get_mle(self, word):
        return self.train_words_mle.get(word, "NN")

    def mle_error(self):
        test_size = 0
        unknown_correct = 0
        correct = 0
        unknown_words = 0
        known_words = 0
        for word, tags in self.test_words_tags.items():
            for tag, count in tags.items():
                test_size += count
                if word not in self.train_words_mle:
                    unknown_words += count
                    if tag == self.get_mle(word):
                        unknown_correct += count

                elif word in self.train_words_mle:
                    known_words += count
                    if tag == self.get_mle(word):
                        correct += count

        accuracy = (correct + unknown_correct) / test_size
        unknown_acc = unknown_correct / unknown_words
        known_acc = correct / known_words
        print(f"Overall error: {1 - accuracy}\n"
              f"Unknown error: {1 - unknown_acc}\n"
              f"Known error: {1 - known_acc}")
        return 1 - accuracy


class pseudoTagger:
    def __init__(self, training, test):
        self.bigHMM = Bigram_HMM(training, test)
        self.trainData = training
        self.testData = test
        self.pseudoTrainingData = []
        self.numOfUnknown = 0
        # self.pseudoDict = dict()
        self.pseudoWords = set()
        self.pseudoTestData = []
        # self.pseudoDict =


    def get_word_count(self, word):
        """
        given a word - counts how many times it apears in the training data according to the word tags dict
        :param word:
        :return:
        """
        count = sum(self.bigHMM.posTagger.train_words_tags[word].values())
        return count

    def getPseudoWord(self, word):
        """
        given a sparse word categorise it into a pseudo category and return the category along with its original tag
        :param word:
        :return: pseudo word and original tag
        """

        if word.isupper() and not word == word.capitalize():                #capitalised word - name / term
            return "PWcapName"

        if word == word.capitalize() and not word.isnumeric():     #all caps
            return "PWallCaps"

        if re.compile(DATE_REGEX).fullmatch(word):      #date
            return "PWdate"

        if re.compile(TIME_REGEX).fullmatch(word):            #time with :
            return "PWtime"

        if re.compile(NUMER_REGEX).fullmatch(word):              #just a number
            return "PWnumber"

        if re.compile(MONEY_REGEX).fullmatch(word):     #money
            return "PWmoney"

        if '-' in word:                                     #has dash
            return "PWdash"

        if '/' in word:                                     #has slash
            return "PWslash"

        return "PWother"                                      #uncategorised



    def getPseudoDataTrain(self):
        """
        creates a new data that replaces sparse words with pseudo words along with matching psuedo words-tags dictionary
        :param data: train data
        :return:   pseudoDict: words-tag dictionary which replaces sparse words with pseudo words
                   pseudoData: tagged text data which replaces sparse words with pseudo words with the original tag
        """
        pseudoDict = dict()
        pseudoData = []
        for sent in self.trainData:
            pseudoSent = []
            for word, tag in sent:
                if self.get_word_count(word) <= SPARSE_WORD_THRESHOLD:            #sparse word - turn into a pseudo word
                    self.pseudoWords.add(word)
                    pseudoWord = self.getPseudoWord(word)
                    pseudoSent.append((pseudoWord, tag))

                    pDict = pseudoDict.setdefault(pseudoWord, dict())       #create a dictionary for a pseudoword
                    pDict[pseudoWord] = pDict.get(tag, 0) + 1

                else:
                    pseudoSent.append((word, tag))

            pseudoData.append(pseudoSent)  # add sentence with pseudo words

        self.pseudoTrainingData = pseudoData
        # self.pseudoDict = pseudoDict


    def getPseudoDataTest(self):
        """
        creates a new data that replaces sparse words and unknown words with pseudo words along with matching psuedo
        words-tags dictionary
        :param data: test data
        :return:   pseudoDict: words-tag dictionary which replaces sparse words with pseudo words
                   pseudoData: tagged text data which replaces sparse words with pseudo words with the original tag
        """
        pseudoData = []
        for sent in self.testData:
            pseudoSent = []
            for word, tag in sent:
                if not(self.bigHMM.check_known_word(word)):   #unknown word - insructed to replace with pseudo word
                    self.numOfUnknown += 1
                    pseudoWord = self.getPseudoWord(word)
                    pseudoSent.append((pseudoWord, tag))

                elif word in self.pseudoWords:              #was replaced by a pseudo word in training and therefore needs to be replaced by pseudo in test as well
                    pseudoWord = self.getPseudoWord(word)
                    pseudoSent.append((pseudoWord, tag))

                else:
                    pseudoSent.append((word, tag))

            pseudoData.append(pseudoSent)  # add sentence with pseudo words

        self.pseudoTestData = pseudoData

    def retrainWithPseudo(self, smoothing=False):
        """
        train the bigram and posTagger with the pseudo data
        :return:
        """
        self.bigHMM = Bigram_HMM(self.pseudoTrainingData, self.pseudoTestData, smoothing)


def B():
    train, test = split_train_test()
    p = posTagger(train, test)
    p.mle_error()


def C(train, test):
    hmm = Bigram_HMM(train, test)
    hmm.viterbi_error(test)


def D(train, test):
    hmm = Bigram_HMM(train, test, True)
    hmm.viterbi_error(test)

def E(train, test):
    pt = pseudoTagger(train, test)
    pt.getPseudoDataTrain()
    pt.getPseudoDataTest()
    # pt.retrainWithPseudo()
    # pt.bigHMM.viterbi_error(pt.pseudoTestData, pt.testData)
    pt.retrainWithPseudo(True)
    pt.bigHMM.viterbi_error(pt.pseudoTestData, pt.testData)

if __name__ == '__main__':
    train, test = split_train_test()
    E(train, test)
