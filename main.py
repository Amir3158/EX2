from nltk.corpus import brown
from sklearn.model_selection import train_test_split
import nltk
import json
import re

nltk.download('brown')

FILTER_COMPLEX_TAG = re.compile(r"[^*+\-]*")


class posTagger:

    def __init__(self, training, test, file=None):
        self.train_words_tags = self.get_word_tags(training)
        self.test_words_tags = self.get_word_tags(test)
        self.train_words_mle = self.get_mle(self.train_words_tags) if file is None else self.get_mle_from_json(file)

    def filter_tag(self, tag):
        return FILTER_COMPLEX_TAG.match(tag).group(0)

    def get_word_tags(self, data):
        """
        create a dict of all the unique words in the data set along with a count of each of their tags
        :param data: a list of tuples of words and their tag.
        :return: a dict of words and the value is dict of each tag and the count in the parm 'data'
        """
        words_tag = dict()
        for sent in data:
            for word in sent:
                w_tag = self.filter_tag(word[1])
                if word[0] in words_tag:
                    if w_tag in words_tag[word[0]]:
                        words_tag[word[0]][w_tag] += 1
                    else:
                        words_tag[word[0]][w_tag] = 1
                else:
                    words_tag[word[0]] = dict()
                    words_tag[word[0]][w_tag] = 1
        return words_tag

    def get_mle(self, data):
        words_mle = dict()
        for word in data:
            words_mle[word] = max(data[word], key=data[word].get)
        return words_mle

    def save_mle(self):
        with open("mle_train_data.json", 'w') as f:
            json.dump(self.train_words_mle, f)

    def get_mle_from_json(self, file):
        with open(file,"r") as f:
            self.train_words_mle = json.load(f)

    def predict_mle(self, word):
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
                    if tag == self.predict_mle(word):
                        unknown_correct += count

                elif word in self.train_words_mle:
                    known_words += count
                    if tag == self.predict_mle(word):
                        correct += count

        accuracy = (correct + unknown_correct) / test_size
        unknown_acc = unknown_correct / unknown_words
        known_acc = correct / known_words
        print(f"Overall error: {1 - accuracy}\n"
              f"Unknown error: {1 - unknown_acc}\n"
              f"Known error: {1 - known_acc}")
        return 1 - accuracy


def split_train_test():
    sents = list(brown.tagged_sents(categories="news"))
    split_size = (len(sents) // 10) + 1
    train = sents[:-split_size]
    test = sents[-split_size:]
    return train, test



if __name__ == '__main__':
    train, test = split_train_test()
    p = posTagger(train, test)

    p.mle_error()