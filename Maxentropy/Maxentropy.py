from random import random
from collections import defaultdict
from Config import config
import numpy as np
import random


class dataset(object):
    def __init__(self, dataset_dir):
        self.sentences = []
        self.tags = []
        sentence = []
        tag = []
        word_num = 0
        # _  戴相龙	_	NR	_	_	2	VMOD	_	_
        with open(dataset_dir, encoding='utf-8') as f:
            for line in f:
                if line !='\n':
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

        print('%s:共%d个句子,共%d个词。' % (dataset_dir, self.sentences_num, self.word_num))

    def split(self):
        data = []
        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):
                data.append((self.sentences[i], j, self.tags[i][j]))
        return data

class Maxentropy(object):
    def __init__(self, train_data_file=None, dev_data_file=None, test_data_file=None):
        self.train_data = dataset(train_data_file ) if train_data_file is not None else None
        self.dev_data = dataset(dev_data_file ) if dev_data_file is not None else None
        self.test_data = dataset(test_data_file ) if test_data_file is not None else None
        self.features = {}
        self.tag_dict = {}
        self.tag_list = []

        # 矩阵形式的权重存储
        self.weight_matrix = None  # 标签×特征 权重矩阵

    def create_feature_template(self, sentence, position):
        template = []
        cur_word = sentence[position]
        cur_word_first_char = cur_word[0]
        cur_word_last_char = cur_word[-1]
        if position == 0:
            last_word = '##'
            last_word_last_char = '#'
        else:
            last_word = sentence[position - 1]
            last_word_last_char = sentence[position - 1][-1]

        if position == len(sentence) - 1:
            next_word = '$$'
            next_word_first_char = '$'
        else:
            next_word = sentence[position + 1]
            next_word_first_char = sentence[position + 1][0]

        template.append('02:' + cur_word)
        template.append('03:' + last_word)
        template.append('04:' + next_word)
        template.append('05:' + cur_word + '*' + last_word_last_char)
        template.append('06:' + cur_word + '*' + next_word_first_char)
        template.append('07:' + cur_word_first_char)
        template.append('08:' + cur_word_last_char)

        for i in range(1, len(sentence[position]) - 1):
            template.append('09:' + sentence[position][i])
            template.append('10:' + sentence[position][0] + '*' + sentence[position][i])
            template.append('11:' + sentence[position][-1] + '*' + sentence[position][i])

        if len(sentence[position]) == 1:
            template.append('12:' + cur_word + '*' + last_word_last_char + '*' + next_word_first_char)

        for i in range(0, len(sentence[position]) - 1):
            if sentence[position][i] == sentence[position][i + 1]:
                template.append('13:' + sentence[position][i] + '*' + 'consecutive')

        for i in range(0, 4):
            if i > len(sentence[position]) - 1:
                break
            template.append('14:' + sentence[position][0:i + 1])
            template.append('15:' + sentence[position][-(i + 1)::])

        return template

    def create_feature_space(self):
        # 创建特征空间
        for i in range(len(self.train_data.sentences)):
            sentence = self.train_data.sentences[i]
            tags = self.train_data.tags[i]
            for j in range(len(sentence)):
                template = self.create_feature_template(sentence, j)
                for f in template:
                    if f not in self.features:
                        self.features[f] = len(self.features)
                for tag in tags:
                    if tag not in self.tag_list:
                        self.tag_list.append(tag)

        self.tag_list = sorted(self.tag_list)
        self.tag_dict = {t: i for i, t in enumerate(self.tag_list)}

        # 初始化矩阵：行=标签数，列=特征数
        num_tags = len(self.tag_dict)
        num_features = len(self.features)

        self.weight_matrix = np.zeros((num_features, num_tags), dtype=np.float64)

        print(f"特征空间: {num_tags} 标签 × {num_features} 特征 = {num_tags * num_features} 参数")

    def predict(self, sentence,position,averaged=False):
        """预测指定位置标签(寻找概率最高的标签)"""
        template = self.create_feature_template(sentence, position)
        feature_ids = [self.features[f] for f in template if f in self.features]
        scores = np.sum(self.weight_matrix[feature_ids],axis=0)
        return self.tag_list[np.argmax(scores)]

    def evaluate(self,data):
        total = correct = 0
        for sentence, tags in zip(data.sentences, data.tags):
            for j, true_tag in enumerate(tags):
                pred = self.predict(sentence, j)
                if pred == true_tag:
                    correct += 1
                total += 1
        return correct, total,correct/total
    def train(self,max_epochs=20,learning_rate=0.1,shuffle=True,l2_reg=0.01,batch_size=32):
        """最大熵模型（梯度下降）"""
        data = self.train_data.split()
        best_dev_acc = 0.0
        no_improve_count = 0
        self.best_weights = np.copy(self.weight_matrix)

        for epoch in range(max_epochs):
            print(f'Epoch {epoch + 1}/{max_epochs}')

            if shuffle:
                random.shuffle(data)
                print('\tShuffled training data')

            gradient = defaultdict(lambda: np.zeros(len(self.tag_list)))
            batch_count = 0

            #梯度下降
            for idx, (sentence, position, gold_tag) in enumerate(data):
                # 获取真实标签ID
                gold_tag_id = self.tag_dict[gold_tag]

                #生成特征
                template = self.create_feature_template(sentence, position)
                feature_ids = [self.features[f] for f in template if f in self.features]
                scores = np.sum(self.weight_matrix[feature_ids],axis=0)
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / np.sum(exp_scores)

                for fid in feature_ids:
                    gradient[fid][gold_tag_id] += 1.0
                    gradient[fid] -= probs

                batch_count +=1

                if batch_count == batch_size or idx == len(data) - 1:
                    #添加L2正则化项
                    for fid, grad in gradient.items():
                        self.weight_matrix[fid] += learning_rate * (grad - l2_reg * self.weight_matrix[fid])
                    gradient.clear()
                    batch_count = 0

            _, _, dev_acc = self.evaluate(self.dev_data)
            print(f"\tDev Acc: {dev_acc:.4f}")
            # 早停机制
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                no_improve_count = 0
                # 保存最佳模型
                self.best_weights = np.copy(self.weight_matrix)
            else:
                no_improve_count += 1
                if no_improve_count >= 3:
                    print("\tEarly stopping")
                # 学习率衰减
                learning_rate *= 0.95
        self.weight_matrix = self.best_weights

if __name__ == '__main__':
    train_data_file = config['train_data_file']
    dev_data_file = config['dev_data_file']
    test_data_file = config['test_data_file']
    model = Maxentropy(train_data_file, dev_data_file, test_data_file)
    model.create_feature_space()
    model.train()
    # 评估测试集
    #_, _, test_acc = model.evaluate(model.test_data)
    #print(f"测试集准确率: {test_acc:.4f}")

    sentence = ["我", "喜欢", "我的", "手机"]
    print("位置3 ('手机') 的标签概率:")
    probs = model.predict_probability(sentence, 3)
    for tag, prob in probs.items():
        print(f"{tag}: {prob:.4f}")





