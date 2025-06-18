import datetime
import numpy as np
import random
from Config import config

class dataset(object):
    def __init__(self, filename):
        self.filename = filename
        self.sentences = []
        self.tags = []
        sentence = []
        tag = []
        word_num = 0
        f = open(filename, encoding='utf-8')
        while True:
            line = f.readline()
            if not line:
                break
            if line == '\n':
                self.sentences.append(sentence)
                self.tags.append(tag)
                sentence = []
                tag = []
            else:
                sentence.append(line.split()[1])
                tag.append(line.split()[3])
                word_num += 1
        self.sentences_num = len(self.sentences)
        self.word_num = word_num

        print('%s:共%d个句子,共%d个词。' % (filename, self.sentences_num, self.word_num))
        f.close()

    def split(self):
        data = []
        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):
                data.append((self.sentences[i], j, self.tags[i][j]))
        return data


class liner_model(object):
    def __init__(self, train_data_file=None, dev_data_file=None, test_data_file=None):
        self.train_data = dataset(train_data_file) if train_data_file is not None else None
        self.dev_data = dataset(dev_data_file) if dev_data_file is not None else None
        self.test_data = dataset(test_data_file) if test_data_file is not None else None
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

        self.weight_matrix = np.zeros((num_tags, num_features), dtype=np.int32)
        self.v_matrix = np.zeros((num_tags, num_features), dtype=np.int32)
        self.update_times_matrix = np.zeros((num_tags, num_features), dtype=np.int32)

        print(f"特征空间: {num_tags} 标签 × {num_features} 特征 = {num_tags * num_features} 参数")

    def dot(self, feature_ids, tag, averaged=False):
        """计算特征向量与权重向量的点积"""
        tag_id = self.tag_dict[tag]
        if averaged:
            return np.sum(self.v_matrix[tag_id, feature_ids])
        else:
            return np.sum(self.weight_matrix[tag_id, feature_ids])

    def predict(self, sentence, position, averaged=False):
        """预测指定位置的标签"""
        # 生成特征模板并转换为ID
        template = self.create_feature_template(sentence, position)
        feature_ids = [self.features[f] for f in template if f in self.features]

        # 计算每个标签的得分
        scores = []
        for tag in self.tag_list:
            scores.append(self.dot(feature_ids, tag, averaged))

        # 选择最高分标签
        tag_id = np.argmax(scores)
        return self.tag_list[tag_id]

    def save(self, path):
        """保存模型权重"""
        with open(path, 'w', encoding='utf-8') as f:
            for tag, tag_id in self.tag_dict.items():
                for feature, feature_id in self.features.items():
                    weight = self.weight_matrix[tag_id, feature_id]
                    if weight != 0:
                        # 格式: 特征模板+权重值
                        f.write(f"{tag}*{feature}\t{weight}\n")

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

    def online_train(self, iterator=20, averaged=False, shuffle=True, exitor=20):
        """在线训练模型"""
        max_dev_precision = 0.0
        max_iterator = -1
        update_counter = 0  # 全局更新计数器

        # 准备训练数据
        data = self.train_data.split()

        if averaged:
            print('使用平均权重(v_matrix)进行验证')

        for iter in range(iterator):
            start_time = datetime.datetime.now()
            print(f'迭代: {iter}')

            if shuffle:
                random.shuffle(data)
                print('\t打乱训练数据顺序')

            for i in range(len(data)):
                sentence, position, gold_tag = data[i]
                predict_tag = self.predict(sentence, position, False)

                if predict_tag != gold_tag:
                    update_counter += 1
                    gold_tag_id = self.tag_dict[gold_tag]
                    predict_tag_id = self.tag_dict[predict_tag]

                    # 生成特征ID
                    features = self.create_feature_template(sentence, position)
                    feature_ids = [self.features[f] for f in features if f in self.features]

                    # 更新权重矩阵
                    for fid in feature_ids:
                        # 记录旧权重值
                        last_predict_weight = self.weight_matrix[predict_tag_id, fid]
                        last_gold_weight = self.weight_matrix[gold_tag_id, fid]

                        # 更新错误标签权重
                        self.weight_matrix[predict_tag_id, fid] -= 1

                        # 更新正确标签权重
                        self.weight_matrix[gold_tag_id, fid] += 1

                        # 更新平均权重矩阵
                        self.update_v_matrix(
                            predict_tag_id, fid,
                            last_predict_weight,
                            update_counter
                        )
                        self.update_v_matrix(
                            gold_tag_id, fid,
                            last_gold_weight,
                            update_counter
                        )

            # 更新未修改权重的平均权重
            self.update_unmodified_weights(update_counter)

            # 评估性能
            train_correct, train_total, train_acc = self.evaluate(self.train_data, False)
            dev_correct, dev_total, dev_acc = self.evaluate(self.dev_data, averaged)

            print(f'\t训练准确率: {train_correct}/{train_total} = {train_acc:.4f}')
            print(f'\t验证准确率: {dev_correct}/{dev_total} = {dev_acc:.4f}')

            if self.test_data is not None:
                test_correct, test_total, test_acc = self.evaluate(self.test_data, averaged)
                print(f'\t测试准确率: {test_correct}/{test_total} = {test_acc:.4f}')

            # 早停机制
            if dev_acc > max_dev_precision:
                max_dev_precision = dev_acc
                max_iterator = iter
                counter = 0
            else:
                counter += 1

            # 保存检查点
            if counter == 0:
                self.save(f'./model_iter_{iter}.txt')

            # 计算迭代时间
            iter_time = datetime.datetime.now() - start_time
            print(f"\t迭代耗时: {iter_time}")

            # 终止条件
            if train_correct == train_total:
                print("训练集100%准确，提前终止")
                break
            if counter >= exitor:
                print(f"验证精度连续{exitor}次未提升，提前终止")
                break

        print(f'最佳迭代: {max_iterator}, 最高验证精度: {max_dev_precision:.4f}')
        return max_dev_precision

    def update_v_matrix(self, tag_id, feature_id, last_weight, update_time):
        """更新平均权重矩阵的单个元素"""
        last_update = self.update_times_matrix[tag_id, feature_id]
        time_diff = update_time - last_update - 1

        # 计算平均权重的增量更新
        self.v_matrix[tag_id, feature_id] += time_diff * last_weight + self.weight_matrix[tag_id, feature_id]
        self.update_times_matrix[tag_id, feature_id] = update_time

    def update_unmodified_weights(self, current_update_time):
        """更新未修改权重的平均权重"""
        # 找出未更新的权重位置
        unmodified_mask = self.update_times_matrix != current_update_time

        # 计算需要累积的步数
        time_diffs = current_update_time - self.update_times_matrix[unmodified_mask]

        # 批量更新平均权重
        self.v_matrix[unmodified_mask] += time_diffs * self.weight_matrix[unmodified_mask]

        # 更新这些权重的时间戳
        self.update_times_matrix[unmodified_mask] = current_update_time


if __name__ == '__main__':
    train_data_file = config['train_data_file']
    dev_data_file = config['dev_data_file']
    test_data_file = config['test_data_file']
    averaged = config['averaged']
    iterator = config['iterator']
    shuffle = config['shuffle']
    exitor = config['exitor']

    lm = liner_model(train_data_file, dev_data_file, test_data_file)
    lm.create_feature_space()
    lm.online_train(iterator, averaged, shuffle, exitor)
