import numpy as np
import random
from collections import defaultdict
from Config import config

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

class globalliner(object):
    def __init__(self, train_data_file=None, dev_data_file=None, test_data_file=None):
        self.train_data = dataset(train_data_file) if train_data_file else None
        self.dev_data = dataset(dev_data_file) if dev_data_file else None
        self.test_data = dataset(test_data_file) if test_data_file else None
        self.features = {}
        self.tag_dict = {}
        self.tag_list = []
        self.weights = None
        self.BOS = '<BOS>'

    def create_feature_template(self, sentence, pre_tag, position):
        template = []
        cur_word = sentence[position]
        cur_word_first_char = cur_word[0]
        cur_word_last_char = cur_word[-1]

        last_word = sentence[position - 1] if position > 0 else '##'
        last_word_last_char = last_word[-1] if position > 0 else '#'
        next_word = sentence[position + 1] if position < len(sentence) - 1 else '$$'
        next_word_first_char = next_word[0] if position < len(sentence) - 1 else '$'

        template.extend([
            '01:' + pre_tag,
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
            for j in range(len(sentence)):
                pre_tag = tags[j - 1] if j > 0 else self.BOS
                template = self.create_feature_template(sentence, pre_tag, j)
                for f in template:
                    if f not in self.features:
                        self.features[f] = len(self.features)
                if tags[j] not in self.tag_list:
                    self.tag_list.append(tags[j])

        self.tag_list.sort()
        self.tag_dict = {t: i for i, t in enumerate(self.tag_list)}
        self.weights = np.zeros((len(self.features), len(self.tag_dict)), dtype=np.float64)
        print(f"特征空间: {len(self.tag_dict)} 标签 × {len(self.features)} 特征 = {len(self.features) * len(self.tag_dict)} 参数")

    def score(self, feature_ids):
        return np.sum(self.weights[feature_ids], axis=0)

    def predict(self, sentence):
        n = len(sentence)
        m = len(self.tag_list)
        dp = np.full((n, m), -np.inf)
        path = np.zeros((n, m), dtype=int)

        # 初始化第一列
        for t in range(m):
            tag = self.tag_list[t]
            template = self.create_feature_template(sentence, self.BOS, 0)
            feature_ids = [self.features[f] for f in template if f in self.features]
            dp[0][t] = self.score(feature_ids)[t]

        for i in range(1, n):
            for curr in range(m):
                curr_tag = self.tag_list[curr]
                for prev in range(m):
                    prev_tag = self.tag_list[prev]
                    template = self.create_feature_template(sentence, prev_tag, i)
                    feature_ids = [self.features[f] for f in template if f in self.features]
                    score = dp[i-1][prev] + self.score(feature_ids)[curr]
                    if score > dp[i][curr]:
                        dp[i][curr] = score
                        path[i][curr] = prev

        # 回溯
        pred = []
        last = np.argmax(dp[n-1])
        for i in range(n-1, -1, -1):
            pred.append(self.tag_list[last])
            last = path[i][last]
        pred.reverse()
        return pred

    def evaluate(self, data):
        total = correct = 0
        for sentence, tags in zip(data.sentences, data.tags):
            pred_tags = self.predict(sentence)
            for pt, tt in zip(pred_tags, tags):
                if pt == tt:
                    correct += 1
                total += 1
        return correct, total, correct / total

    def update_weights(self, gold_fids, gold_tag_id, pred_fids, pred_tag_id):
        for fid in gold_fids:
            self.weights[fid][gold_tag_id] += 1
        for fid in pred_fids:
            self.weights[fid][pred_tag_id] -= 1

    def online_train(self, epochs=20, shuffle=False):
        best_dev_acc = 0.0
        best_weights = np.copy(self.weights)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            indices = list(range(len(self.train_data.sentences)))
            if shuffle:
                random.shuffle(indices)
            for idx in indices:
                sentence = self.train_data.sentences[idx]
                gold_tags = self.train_data.tags[idx]
                pred_tags = self.predict(sentence)
                if pred_tags != gold_tags:
                    for j in range(len(sentence)):
                        pre_gold = gold_tags[j - 1] if j > 0 else self.BOS
                        pre_pred = pred_tags[j - 1] if j > 0 else self.BOS
                        gold_template = self.create_feature_template(sentence, pre_gold, j)
                        pred_template = self.create_feature_template(sentence, pre_pred, j)
                        gold_fids = [self.features[f] for f in gold_template if f in self.features]
                        pred_fids = [self.features[f] for f in pred_template if f in self.features]
                        self.update_weights(gold_fids, self.tag_dict[gold_tags[j]], pred_fids, self.tag_dict[pred_tags[j]])
            _, _, dev_acc = self.evaluate(self.dev_data)
            print(f"\tDev Acc: {dev_acc:.4f}")
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_weights = np.copy(self.weights)
        self.weights = best_weights

if __name__ == '__main__':
    train_data_file = config['train_data_file']
    dev_data_file = config['dev_data_file']
    test_data_file = config['test_data_file']

    model = globalliner(train_data_file, dev_data_file, test_data_file)
    model.create_feature_space()
    model.online_train()
    _, _, test_acc = model.evaluate(model.test_data)
    print(f"测试集准确率: {test_acc:.4f}")

