from random import random

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
        f = open(dataset_dir,encoding='utf-8')
        # _  戴相龙	_	NR	_	_	2	VMOD	_	_
        while True:
            line = f.readline()
            if not line:
                break
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

        print('%s:共%d个句子,共%d个词。' % (dataset_dir, self.sentences_num, self.word_num))
        f.close()

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
        self.v_matrix = None  # 标签×特征 平均权重矩阵
        self.update_times_matrix = None  # 标签×特征 更新时间矩阵

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

        self.weight_matrix = np.zeros((num_tags, num_features), dtype=np.float64)
        self.v_matrix = np.zeros((num_tags, num_features), dtype=np.float64)
        self.update_times_matrix = np.zeros((num_tags, num_features), dtype=np.int32)

        print(f"特征空间: {num_tags} 标签 × {num_features} 特征 = {num_tags * num_features} 参数")


    def conditional_probability(self, feature_ids, tag_id, averaged=False):
        """计算给定特征的条件概率 P(tag|features)"""
        scores = []

        for tid in range(len(self.tag_list)):
            if averaged:
                score = np.sum(self.v_matrix[tid, feature_ids])
            else:
                score = np.sum(self.weight_matrix[tid, feature_ids])
            scores.append(score)

        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)

        prob = exp_scores[tag_id] / np.sum(exp_scores)
        return prob
    def predict(self, sentence,position,averaged=False):
        """预测指定位置标签(寻找概率最高的标签)"""
        template = self.create_feature_template(sentence, position)
        feature_ids = [self.features[f] for f in template if f in self.features]

        #计算每个标签的概率
        probs = []
        for tag in self.tag_list:
            tag_id = self.tag_dict[tag]
            prob = self.conditional_probability(feature_ids, tag_id, averaged)
            probs.append(prob)

        #选择概率最高的标签
        tag_id = np.argmax(probs)

        return self.tag_list[tag_id]

    def predict_probability(self, sentence,position,averaged=False):
        """返回所有标签的概率分布"""
        template = self.create_feature_template(sentence, position)
        feature_ids = [self.features[f] for f in template if f in self.features]

        #计算每个标签的该率
        probs = {}
        for tag in self.tag_list:
            tag_id = self.tag_dict[tag]
            prob = self.conditional_probability(feature_ids, tag_id, averaged)
            probs[tag] = prob

        return probs

    def train(self,max_epochs=20,learning_rate=0.1,shuffle=True,l2_reg=0.01):
        """最大熵模型（梯度下降）"""
        data = self.train_data.split()
        best_dev_acc = 0.0
        no_improve_count = 0

        for epoch in range(max_epochs):
            print(f'Epoch {epoch + 1}/{max_epochs}')
            total_loss = 0.0

            if shuffle:
                random.shuffle(data)
                print('\tShuffled training data')

            #梯度下降
            for sentence,position,gold_tag in data:
                # 获取真实标签ID
                gold_tag_id = self.tag_dict[gold_tag]

                #生成特征
                template = self.create_feature_template(sentence, position)
                feature_ids = [self.features[f] for f in template if f in self.features]

                #计算梯度
                gradient = np.zeros_like(self.weight_matrix,dtype=np.float64)

                for fid in feature_ids:
                    gradient[gold_tag_id,fid] += 1.0

                for tag_id in range(len(self.tag_list)):
                    prob = self.conditional_probability(feature_ids, tag_id)
                    for fid in feature_ids:
                        gradient[tag_id,fid] -= prob
                #添加L2正则化项
                gradient += l2_reg * self.weight_matrix

                #更新权重
                self.weight_matrix -=learning_rate * gradient

                #计算损失
                ture_prob = self.conditional_probability(feature_ids, gold_tag_id)
                log_loss = -np.log(ture_prob + 1e-10)
                reg_loss = 0.5 * l2_reg * np.sum(self.weight_matrix**2)
                total_loss += log_loss + reg_loss

            # 评估
            train_correct, train_total, train_acc = self.evaluate(self.train_data)
            dev_correct, dev_total, dev_acc = self.evaluate(self.dev_data)

            avg_loss = total_loss / len(data)
            print(f'\tLoss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Dev Acc: {dev_acc:.4f}')

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
                    # 恢复最佳权重
                    self.weight_matrix = self.best_weights
                    break

            # 学习率衰减
            learning_rate *= 0.95

    def evaluate(self, data, averaged=False):
        """评估模型性能"""
        total_num = 0
        correct_num = 0
        for i in range(len(data.sentences)):
            sentence = data.sentences[i]
            tags = data.tags[i]
            total_num += len(tags)
            for j in range(len(sentence)):
                predict_tag = self.predict(sentence, j, averaged)
                if predict_tag == tags[j]:
                    correct_num += 1
        accuracy = correct_num / total_num if total_num > 0 else 0
        return correct_num, total_num, accuracy

if __name__ == '__main__':
    train_data_file = config['train_data_file']
    dev_data_file = config['dev_data_file']
    test_data_file = config['test_data_file']
    model = Maxentropy(train_data_file, dev_data_file, test_data_file)
    model.create_feature_space()
    model.train()
    # 评估测试集
    _, _, test_acc = model.evaluate(model.test_data, averaged=True)
    print(f"测试集准确率: {test_acc:.4f}")

    # 示例预测
    sentence = ["我", "喜欢", "我的", "手机"]
    print("位置3 ('手机') 的标签概率:")
    probs = model.predict_probability(sentence, 3)
    for tag, prob in probs.items():
        print(f"{tag}: {prob:.4f}")





