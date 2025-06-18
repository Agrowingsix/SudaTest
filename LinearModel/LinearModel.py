from Config import config
import pickle
import numpy as np


class Dataset(object):
    def __init__(self,dataset_dir):
        self.sentences = []
        self.tags = []
        sentence = []
        tag = []
        word_num = 0
        f = open(dataset_dir,encoding='utf-8')
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

        print('%s:共%d个句子,共%d个词。' % (dataset_dir, self.sentences_num, self.word_num))
        f.close()
    def split(self):
        data = []
        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):
                data.append((self.sentences[i],j,self.tags[i][j])) #f(S,i,t) 句子S中wi标注为t时所对应的特征向量
        return data

class liner_model:
    def __init__(self,train_data_file,dev_data_file,test_data_file):
        self.train_data = Dataset(train_data_file)
        self.dev_data = Dataset(dev_data_file)
        self.test_data = None
        self.features = {}
        self.weights = []
        self.tag_list = []
        self.v = []
        self.tag_dict = {}
    def create_features_template(self,sentence,tag,position):
        template = []
        cur_word = sentence[position]
        cur_tag = tag
        cur_word_first_char = cur_word[0]
        cur_word_last_char = cur_word[-1]
        if position == 0:
            last_word = '##'
            last_word_last_char = '#'
        else:
            last_word = sentence[position-1]
            last_word_last_char = sentence[position-1][-1]
        if position == len(sentence)-1:
            next_word = '$$'
            next_word_first_char = '$'
        else:
            next_word = sentence[position+1]
            next_word_first_char = sentence[position+1][0]

        template.append('02:' + cur_tag + '*' + cur_word) # t + wi
        template.append('03:' + cur_tag + '*' + last_word) # t + wi-1
        template.append('04:' + cur_tag + '*' + next_word ) # t + wi+1
        template.append('05:' + cur_tag + '*' + cur_word + '*' + last_word_last_char) # t + wi + ci-1
        template.append('06:' + cur_tag + '*' + cur_word+ '*' + next_word_first_char) # t + wi + ci+1
        template.append('07' + cur_tag + '*' + cur_word_first_char) # t + ci,0
        template.append('08:' + cur_tag + '*' + cur_word_last_char)

        for i in range(1,len(sentence[position]) - 1):
            template.append('09:' + cur_tag + '*' + sentence[position][i])
            template.append('10:' + cur_tag + '*' + sentence[position][0] + '*' + sentence[position][i])
            template.append('11:' + cur_tag + '*' + sentence[position][-1] + '*' + sentence[position][i])
        if len(sentence[position]) == 1:
            template.append('12:' + cur_tag + '*' + cur_word + '*' + last_word_last_char + '*' + next_word_first_char)

        if len(sentence[position]) > 1 and sentence[position][0] == sentence[position][1]:
            template.append('13:' + cur_tag + '*' + sentence[position][0] + '*' + 'consecutive')

        for i in range(0, 4):
            if i > len(sentence[position]) - 1:
                break
            template.append('14:' + cur_tag + '*' + sentence[position][0:i + 1])
            template.append('15:' + cur_tag + '*' + sentence[position][-(i + 1)::])

        return template
    #创建特征空间
    def create_feature_space(self):
        for i in range(len(self.train_data.sentences)):
            sentence = self.train_data.sentences[i]
            tags = self.train_data.tags[i]
            for j in range(len(sentence)):
                template = self.create_features_template(sentence,tags[j],j)
                for k in template:
                    if k not in self.features.keys():
                        self.features[k] = len(self.features)
                for tag in tags:
                    if tag not in self.tag_list:
                        self.tag_list.append(tag)
        self.weights=np.zeros(len(self.features),dtype='int32')
        self.v = np.zeros(len(self.features),dtype='int32')
        self.tag_list = sorted(self.tag_list)
        self.tag_dict = {t: i for i,t in enumerate(self.tag_list)}
        print("the total number of features is %d" % (len(self.features)))
    def dot(self,feature,averaged=False):
        score = 0
        for f in feature:
            if f in self.features:
                if averaged == False:
                    score += self.weights[self.features[f]]
                else:
                    score += self.v[self.features[f]]
        return score
    def predict(self,sentence,position,averaged=False):
        #max_score = -float("inf")
        #best_tag = None
        #for tag in self.tag_list:
        #    cur_socre = self.dot(self.create_features_template(sentence,tag,position))
        #    if cur_score > max_score:
        #        max_score = cur_score
        #        best_tag = tag
        #        return tag
        tagid = np.argmax([self.dot(self.create_features_template(sentence,tag,position),averaged=averaged) for tag in self.tag_list])
        return self.tag_list[tagid]

    def save(self,path):
        f = open(path, 'w', encoding='utf-8')
        for key in self.features:
            f.write(key + '\t' + str(self.weights[self.features[key]]) + '\n')
        f.close()

    def evaluate(self,data,averaged=False):
        total = 0
        correct = 0
        for i in range(len(data.sentences)):
            sentence = data.sentences[i]
            tags = data.tags[i]
            total +=len(tags)
            for j in range(len(sentence)):
                predict_tag = self.predict(sentence,j,averaged)
                if predict_tag == tags[j]:
                    correct +=1
        return (correct,total,correct/total)

    def online_train(self,iterator = 20,averaged=False,shuffle=False,exitor=20):
        max_dev_precision = 0
        max_iterator = -1
        counter = 0
        data = self.train_data.split()
        for iter in range(iterator):
            for i in range(len(data)):
                sentence = data[i][0]
                j = data[i][1]
                gold_tag = data[i][2]
                predict_tag = self.predict(sentence,j,averaged)
                if predict_tag != gold_tag:
                    feature_max = self.create_features_template(sentence, predict_tag, j)
                    feature_gold = self.create_features_template(sentence, gold_tag, j)
                    for f in feature_max:
                        if f in self.features.keys():
                            self.weights[self.features[f]] -= 1
                    for f in feature_gold:
                        if f in self.features.keys():
                            self.weights[self.features[f]] += 1
                    self.v += self.weights

            train_correct_num, total_num, train_precision = self.evaluate(self.train_data, False)
            print('\t' + 'train准确率：%d / %d = %f' % (train_correct_num, total_num, train_precision))
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev_data, averaged)
            print('\t' + 'dev准确率：%d / %d = %f' % (dev_correct_num, dev_num, dev_precision))

            if self.test_data != None:
                test_correct_num, test_num, test_precision = self.evaluate(self.test_data, averaged)
                print('\t' + 'test准确率：%d / %d = %f' % (test_correct_num, test_num, test_precision))

            if dev_precision > max_dev_precision:
                max_dev_precision = dev_precision
                max_iterator = iter
                counter = 0
            else:
                counter += 1
                # self.save('./result.txt')

            if train_correct_num == total_num:
                break
            if counter >= exitor:
                break
        print('iterator = %d , max_dev_precision = %f' % (max_iterator, max_dev_precision))

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









