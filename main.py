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
        test_size = 0
        unknown_correct = 0
        correct = 0
        unknown_words = 0
        known_words = 0
        for sent in self.test_data:
            for word in sent:
                test_size += 1
                if word[0] not in self.train_words_mle:
                    unknown_words += 1
                    if word[1] == "NN":
                        unknown_correct += 1

                elif word[0] in self.train_words_mle:
                    known_words += 1
                    if word[1] == self.train_words_mle[word[0]]:
                        correct += 1

        accuracy = (correct + unknown_correct) / test_size
        unknown_acc = unknown_correct / unknown_words
        known_acc = correct / known_words
        print(f"Overall error: {1 - accuracy}\n"
              f"Unknown error: {1 - unknown_acc}\n"
              f"Known error: {1 - known_acc}")
        return 1 - accuracy


def split_train_test():
    sents = brown.tagged_sents(categories="news")
    train,test = train_test_split(sents, test_size=0.1, random_state=False)

    # temp = set()
    # for sent in test:
    #     for tup in sent:
    #         temp.add(tuple([tup[0], tup[1]]))
    # test = list(temp)
    return train, test



if __name__ == '__main__':
    train, test = split_train_test()
    p = posTagger(train, test)

    p.mle_error()