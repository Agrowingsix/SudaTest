import numpy as np
import random
from collections import defaultdict
from Config import config
import datetime

class dataset(object):
    def __init__(self, dataset_dir):
        self.sentences = []
        self.tags = []
        sentence = []
        tag = []
        word_num = 0
        with open(dataset_dir, encoding='utf-8') as f:
            for line in f:
                if line != '\n':
                    sentence.append(line.split()[1])
                    tag.append(line.split()[3])
                    word_num += 1
                else:
                    self.sentences.append(sentence)
                    self.tags.append(tag)
                    sentence = []
                    tag = []
        self.sentences_num = len(self.sentences)
        self.word_num = word_num
        print(f"{dataset_dir}:共{self.sentences_num}个句子,共{self.word_num}个词。")

    def split(self):
        return [(self.sentences[i], self.tags[i]) for i in range(len(self.sentences))]


class globalliner(object):
    def __init__(self, train_data_file=None, dev_data_file=None, test_data_file=None):
        self.train_data = dataset(train_data_file) if train_data_file else None
        self.dev_data = dataset(dev_data_file) if dev_data_file else None
        self.test_data = dataset(test_data_file) if test_data_file else None
        self.features = {}
        self.tag_dict = {}
        self.tag_list = []
        self.weights_matrix = None

    def create_feature_template(self, sentence, pre_tag, position):
        template = []
        cur_word = sentence[position]
        cur_word_first_char = cur_word[0]
        cur_word_last_char = cur_word[-1]

        last_word = sentence[position - 1] if position > 0 else '##'
        last_word_last_char = last_word[-1] if position > 0 else '#'
        next_word = sentence[position + 1] if position < len(sentence) - 1 else '$$'
        next_word_first_char = next_word[0] if position < len(sentence) - 1 else '$'

        template.append('01:' + pre_tag)
        template.extend([
            '02:' + cur_word,
            '03:' + last_word,
            '04:' + next_word,
            '05:' + cur_word + '*' + last_word_last_char,
            '06:' + cur_word + '*' + next_word_first_char,
            '07:' + cur_word_first_char,
            '08:' + cur_word_last_char
        ])

        for i in range(1, len(cur_word) - 1):
            template.append('09:' + cur_word[i])
            template.append('10:' + cur_word[0] + '*' + cur_word[i])
            template.append('11:' + cur_word[-1] + '*' + cur_word[i])

        if len(cur_word) == 1:
            template.append('12:' + cur_word + '*' + last_word_last_char + '*' + next_word_first_char)

        for i in range(len(cur_word) - 1):
            if cur_word[i] == cur_word[i + 1]:
                template.append('13:' + cur_word[i] + '*' + 'consecutive')

        for i in range(4):
            if i < len(cur_word):
                template.append('14:' + cur_word[:i + 1])
                template.append('15:' + cur_word[-(i + 1):])

        return template

    def create_feature_space(self):
        for sentence, tags in zip(self.train_data.sentences, self.train_data.tags):
            for j, tag in enumerate(tags):
                pre_tag = '<BOS>' if j == 0 else tags[j - 1]
                template = self.create_feature_template(sentence, pre_tag, j)
                for f in template:
                    if f not in self.features:
                        self.features[f] = len(self.features)
                if tag not in self.tag_list:
                    self.tag_list.append(tag)

        self.tag_list.sort()
        self.tag_dict = {t: i for i, t in enumerate(self.tag_list)}
        self.weights_matrix = np.zeros((len(self.features), len(self.tag_dict)), dtype=np.float64)
        print(f"特征空间: {len(self.tag_dict)} 标签 × {len(self.features)} 特征 = {len(self.features) * len(self.tag_dict)} 参数")

    def score(self, features):
        return np.sum([self.weights_matrix[self.features[f]] for f in features if f in self.features], axis=0)

    def predict(self, sentence):
        length = len(sentence)
        num_tags = len(self.tag_list)
        dp = np.zeros((length, num_tags))
        path = np.zeros((length, num_tags), dtype=int)

        first_features = self.create_feature_template(sentence, '<BOS>', 0)
        dp[0] = self.score(first_features)

        for i in range(1, length):
            for curr in range(num_tags):
                best_score = float('-inf')
                best_prev = -1
                for prev in range(num_tags):
                    pre_tag = self.tag_list[prev]
                    features = self.create_feature_template(sentence, pre_tag, i)
                    score = dp[i - 1][prev] + self.score(features)[curr]
                    if score > best_score:
                        best_score = score
                        best_prev = prev
                dp[i][curr] = best_score
                path[i][curr] = best_prev

        best_last = np.argmax(dp[length - 1])
        best_seq = [best_last]
        for i in range(length - 1, 0, -1):
            best_last = path[i][best_last]
            best_seq.append(best_last)

        return [self.tag_list[i] for i in reversed(best_seq)]

    def evaluate(self, data):
        correct = total = 0
        for sentence, tags in zip(data.sentences, data.tags):
            pred = self.predict(sentence)
            for p, t in zip(pred, tags):
                if p == t:
                    correct += 1
                total += 1
        return correct, total, correct / total

    def train(self, epochs=20):
        best_acc = 0.0
        best_weights = np.copy(self.weights_matrix)
        for epoch in range(epochs):
            print(f"迭代 {epoch+1}/{epochs}")
            updates = 0
            for sentence, gold_tags in zip(self.train_data.sentences, self.train_data.tags):
                pred_tags = self.predict(sentence)
                if pred_tags != gold_tags:
                    updates += 1
                    for i in range(len(sentence)):
                        gold_pre = '<BOS>' if i == 0 else gold_tags[i - 1]
                        pred_pre = '<BOS>' if i == 0 else pred_tags[i - 1]
                        gold_features = self.create_feature_template(sentence, gold_pre, i)
                        pred_features = self.create_feature_template(sentence, pred_pre, i)
                        for f in gold_features:
                            if f in self.features:
                                self.weights_matrix[self.features[f], self.tag_dict[gold_tags[i]]] += 1
                        for f in pred_features:
                            if f in self.features:
                                self.weights_matrix[self.features[f], self.tag_dict[pred_tags[i]]] -= 1

            _, _, dev_acc = self.evaluate(self.dev_data)
            print(f"\tDev 准确率: {dev_acc:.4f}")
            if dev_acc > best_acc:
                best_acc = dev_acc
                best_weights = np.copy(self.weights_matrix)

        self.weights_matrix = best_weights

if __name__ == '__main__':
    train_data_file = config['train_data_file']
    dev_data_file = config['dev_data_file']
    test_data_file = config['test_data_file']

    model = globalliner(train_data_file, dev_data_file, test_data_file)
    model.create_feature_space()
    model.train()
    _, _, test_acc = model.evaluate(model.test_data)
    print(f"测试集准确率: {test_acc:.4f}")
