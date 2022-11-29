import numpy as np
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
import nltk
import json
import re

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
    def __init__(self, training, test):
        self.pos_tagger = posTagger(training, test)
        self.train_transition = self.init_transition(training)

    def init_transition(self, data):
        transition = dict()
        for sent in data:
            last_tag = "START"
            for w, tag in sent:
                t = filter_tag(tag)
                inner = transition.setdefault(last_tag, dict())
                inner[t] = inner.get(t, 0) + 1
                last_tag = t
            inner = transition.setdefault(last_tag, dict())
            inner["STOP"] = inner.get("STOP", 0) + 1
        return transition

    def calc_transition(self, tag1, tag2):
        if tag1 not in self.train_transition:
            return 0
        count_tag1_tag2 = self.train_transition[tag1].get(tag2, 0)
        return count_tag1_tag2 / sum(self.train_transition[tag1].values())

    def calc_emission(self, t, w):
        c = self.pos_tagger.get_words_tag_count(w,t)  # amount of occurrences the word w is tagged to t
        if c == -1:
            return -1
        return c / sum(self.train_transition[t].values())

    def calc_prev_max_tag(self, all_tags, matrix, i, tag):
        max_prob = 0
        index = 0
        selected_tag = all_tags[0]
        for j, u in enumerate(all_tags):
            p = matrix[i][j][0] * self.calc_transition(u, tag)
            if p > max_prob:
                max_prob = p
                selected_tag = u
                index = j
        list_of_tags = matrix[i][index][1]
        list_of_tags.append(selected_tag)
        return max_prob, list_of_tags

    def calc_Viterbi(self, sent):
        k = len(sent)
        num_of_tags = len(self.train_transition)
        all_tags = list(self.train_transition.keys())
        all_tags.remove("START")
        viterbi_matrix = []
        text = ['']
        text.extend(sent)
        for i, w in enumerate(text):
            if i == 0:
                viterbi_matrix.append([(1.0, ["START"])] * num_of_tags)
                continue
            viterbi_matrix.append([(0, [])] * num_of_tags)
            if not self.pos_tagger.check_known_word(w):  # unknown word
                p, tags = self.calc_prev_max_tag(all_tags, viterbi_matrix, i - 1, all_tags[0])
                viterbi_matrix[i][0] = (p, tags)
            else:  # known word
                for j, t in enumerate(all_tags):
                    e = self.calc_emission(t, w)
                    p, tags = self.calc_prev_max_tag(all_tags, viterbi_matrix, i, t)
                    viterbi_matrix[i][j] = (e * p, tags)
            if sum([g[0] for g in viterbi_matrix[i]]) == 0:
                tags = viterbi_matrix[i][0][1]
                tags.extend([all_tags[0] * (k-i)])
                return 0, tags
        print(1)
        return max(viterbi_matrix[k], key=lambda item: item[0])

    def viterbi_error(self, data):
        test_size = 0
        correct = 0
        unknown_correct = 0
        unknown_words = 0
        known_words = 0
        for sentence in data:
            sent, tags = [list(i) for i in zip(*sentence)]
            tags = [filter_tag(t) for t in tags]
            predicted_tags = self.calc_Viterbi(sent)[1]
            for i, word in enumerate(sent):
                if not self.pos_tagger.check_known_word(word):
                    unknown_words += 1
                    if tags[i] == predicted_tags[i]: #correct guess for unknown word
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
              f"Unknown words error: {1 - unknown_accuracy}"
              f"Known words error: {1 - known_accuracy}")
        return 1 - accuracy




class posTagger:

    def __init__(self, training, test, file=None):
        self.train_words_tags = self.init_word_tags(training)
        self.test_words_tags = self.init_word_tags(test)
        self.train_words_mle = self.init_mle(self.train_words_tags) if file is None else self.get_mle_from_json(file)

    def get_words_tag_count(self, word, tag):
        if not self.check_known_word(word):
            return -1
        return self.train_words_tags[word].get(tag, 0)

    def check_known_word(self, word):
        if word not in self.train_words_tags:
            return False
        return True

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


def B():
    train, test = split_train_test()
    p = posTagger(train, test)
    p.mle_error()

def C():
    train, test = split_train_test()
    hmm = Bigram_HMM(train, test)
    hmm.viterbi_error(test)



if __name__ == '__main__':
    C()
