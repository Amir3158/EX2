from nltk.corpus import brown
from sklearn.model_selection import train_test_split
import nltk
import json

nltk.download('brown')


class posTagger:

    def __init__(self, training, test, file=None):
        self.training_data = training
        self.test_data = test
        self.train_words_tags = self.get_word_tags(training)
        self.train_words_mle = self.get_mle(self.train_words_tags) if file is None else self.get_mle_from_json(file)

    def get_word_tags(self, data):
        """
        create a dict of all the unique words in the data set along with a count of each of their tags
        :param data: a list of tuples of words and their tag.
        :return: a dict of words and the value is dict of each tag and the count in the parm 'data'
        """
        words_tag = dict()
        for sent in data:
            for word in sent:
                if word[0] in words_tag:
                    if word[1] in words_tag[word[0]]:
                        words_tag[word[0]][word[1]] += 1
                    else:
                        words_tag[word[0]][word[1]] = 1
                else:
                    words_tag[word[0]] = dict()
                    words_tag[word[0]][word[1]] = 1
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
        test_size = len(self.test_data)
        correct_guess = 0
        correct = 0

        for sentence in self.test_data:
            for word in sentence:
                if word[0] not in self.train_words_mle and word[1] == "NN":
                    correct_guess += 1

                elif word[0] in self.train_words_mle and word[1] == self.get_mle(word):
                    correct += 1

        accuracy = (correct + correct_guess) / test_size
        return 1 - accuracy


def split_train_test():
    sents = brown.tagged_sents(categories="news")
    return train_test_split(sents, test_size=0.1, random_state=False)


if __name__ == '__main__':
    train, test = split_train_test()
    p = posTagger(train, test)

