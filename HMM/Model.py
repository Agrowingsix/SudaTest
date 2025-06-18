# 一阶隐马尔科夫模型

import pickle
import numpy as np
from Config import *
from dataload import DataLoader

class HMM:
    def __init__(self, train_set_dir):
        self.dataset = DataLoader(train_set_dir)  # 训练集对象
        self.word_num = len(self.dataset.words) # 词数
        self.tag_num = len(self.dataset.tags) # 词性数
        #self.transition_matrix = np.zeros([self.tag_num, self.tag_num]) # 词性转移矩阵
        #self.emit_matrix = np.zeros([self.tag_num-2, self.word_num]) # 词性-词发射矩阵
        self.train() # 开始训练

    def train(self):
        """
        训练HMM模型，统计转移矩阵和发射矩阵参数
        使用向量化操作优化性能，添加详细注释
        """
        # 重置矩阵确保每次训练都是从头开始
        self.transition_matrix = np.zeros((self.tag_num, self.tag_num))
        self.emit_matrix = np.zeros((self.tag_num - 2, self.word_num))

        # 获取特殊标记的ID
        bos_id = self.dataset.tag2id['<BOS>']
        eos_id = self.dataset.tag2id['<EOS>']

        for sentence in self.dataset.sentences:
            # 提取词性序列和词语序列
            tags = [tag for _, tag in sentence]
            words = [word for word, _ in sentence]

            # 获取词性ID序列和词语ID序列
            tag_ids = [self.dataset.tag2id[tag] for tag in tags]
            word_ids = [self.dataset.word2id.get(word, self.dataset.word2id['<UNK>']) for word in words]

            # 处理句首转移：<BOS> -> 第一个词性
            self.transition_matrix[bos_id, tag_ids[0]] += 1

            # 处理句内转移：tag_i -> tag_{i+1}
            for i in range(len(tag_ids) - 1):
                prev_tag, next_tag = tag_ids[i], tag_ids[i + 1]
                self.transition_matrix[prev_tag, next_tag] += 1

            # 处理句尾转移：最后一个词性 -> <EOS>
            self.transition_matrix[tag_ids[-1], eos_id] += 1

            # 处理发射矩阵：词性 -> 词语
            for tag_id, word_id in zip(tag_ids, word_ids):
                # 只更新有效词性（排除<BOS>和<EOS>）
                if tag_id < self.tag_num - 2:  # 确保是有效词性
                    self.emit_matrix[tag_id, word_id] += 1

        # 平滑处理并转换为对数概率
        self.transition_matrix = self.apply_smoothing(self.transition_matrix)
        self.emit_matrix = self.apply_smoothing(self.emit_matrix)

    def apply_smoothing(self, matrix, alpha=0.5):
        """
        应用加性平滑并转换为对数概率
        :param matrix: 原始计数矩阵
        :param alpha: 平滑因子
        :return: 平滑后的对数概率矩阵
        """
        # 加alpha平滑
        smoothed = matrix + alpha

        # 计算每行的行和（考虑平滑后的分母）
        row_sums = smoothed.sum(axis=1, keepdims=True) + alpha * matrix.shape[1]

        # 归一化为概率
        prob_matrix = smoothed / row_sums

        # 转换为对数概率（避免零值问题）
        return np.log(prob_matrix + 1e-10)  # 添加极小值防止log(0)

    def viterbi_predict(self, sentence):
        # 将单词列表转换为对应的 id 列表
        word_ids = [self.dataset.word2id.get(word, self.dataset.word2id['<UNK>']) for word in sentence]
        num_words = len(sentence)  # 单词数量
        num_tags = self.tag_num - 2  # 词性数量（不含 BOS 和 EOS）

        # 初始化 DP 矩阵和回溯矩阵
        dp_table = np.zeros((num_words, num_tags))
        backtrack = np.zeros((num_words, num_tags), dtype='int')

        # 初始化第一个词的 DP 值和回溯路径
        bos_idx = self.dataset.tag2id['<BOS>']
        dp_table[0] = self.transition_matrix[bos_idx, :-2] + self.emit_matrix[:, word_ids[0]]
        backtrack[0] = np.full(num_tags, -1)

        # 前向传播计算各词各词性的最大概率
        for i in range(1, num_words):
            # 计算从上一个词各词性转移到当前词各词性的概率
            scores = dp_table[i - 1].reshape(num_tags, 1) + self.transition_matrix[:-2, :-2]
            backtrack[i] = np.argmax(scores, axis=0)  # 记录最大概率对应的上一个词的词性索引
            dp_table[i] = np.max(scores, axis=0) + self.emit_matrix[:, word_ids[i]]  # 更新当前词各词性的最大概率

        # 计算最后一个词转移到 EOS 的概率，并确定最后一个词的词性索引
        eos_idx = self.dataset.tag2id['<EOS>']
        last_probs = dp_table[-1] + self.transition_matrix[:-2, eos_idx]
        best_last_tag = np.argmax(last_probs)

        # 回溯得到整个序列的词性索引列表
        best_tags = [best_last_tag]
        for i in range(num_words - 1, 0, -1):
            best_tags.append(backtrack[i][best_tags[-1]])
        best_tags.reverse()

        # 将词性索引转换为词性标签
        return [self.dataset.tags[tag_idx] for tag_idx in best_tags]

    def evaluate(self, test_set):
        correct_words = 0  # 预测正确的词数
        total_words = 0  # 总词数

        for i in range(len(test_set.states)):
            sentence = test_set.states[i]
            golden_tags = test_set.golden_tag[i]

            # 预测词性序列
            pred_tags = self.viterbi_predict(sentence)

            # 统计正确的词数
            total_words += len(pred_tags)
            correct_words += sum(1 for j in range(len(pred_tags)) if pred_tags[j] == golden_tags[j])

        # 输出评估结果
        print("总句子数：{:d}，总词数：{:d}，预测正确的词数：{:d}，预测正确率：{:f}。".
              format(len(test_set.states), total_words, correct_words, correct_words / total_words))

    def save(self, save_dir):
        """
        保存模型
        :param save_dir:
        :return:
        """
        with open(save_dir, 'wb') as f:
            pickle.dump(self, f)
