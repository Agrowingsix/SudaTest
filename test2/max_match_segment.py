import os


def load_dictionary(file_path):
    dictionary = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        # 跳过第一行
        next(f)
        for line in f:
            word = line.strip()
            if word:
                dictionary.add(word)
    return dictionary


def max_match_segment(text, dictionary, max_len=10):
    """前向最大匹配算法"""
    result = []
    index = 0
    text_length = len(text)

    while index < text_length:
        matched = False
        for length in range(min(max_len, text_length - index), 0, -1):
            word = text[index:index + length]
            if word in dictionary or length == 1:
                result.append(word)
                index += length
                matched = True
                break

        if not matched:
            result.append(text[index])
            index += 1

    return result


def reverse_max_match_segment(text, dictionary, max_len=10):
    """后向最大匹配算法"""
    result = []
    index = len(text)  # 从文本末尾开始

    while index > 0:
        matched = False
        # 确定当前窗口的起始位置
        start = max(0, index - max_len)

        # 尝试从最长可能词长开始匹配
        for length in range(min(max_len, index - start), 0, -1):
            # 从当前窗口末尾向前取词
            word_start = index - length
            word = text[word_start:index]

            # 如果词在词典中或是单个字符
            if word in dictionary or length == 1:
                result.append(word)
                index = word_start  # 移动到当前词之前的位置
                matched = True
                break

        # 如果没有匹配到任何词
        if not matched:
            # 切分单个字符（从当前位置向前一个字符）
            result.append(text[index - 1:index])
            index -= 1

    # 因为是从后向前切分的，需要反转结果
    return result[::-1]


def get_string(word_list):
    """将词列表合并为字符串"""
    return ''.join(word_list)


def evaluate_results(segmented_file, answer_file):
    """评估分词结果"""
    # 读取分词结果
    with open(segmented_file, 'r', encoding='utf-8-sig') as f:
        segmented_text = f.read()
        segmented_words = segmented_text.split()

    # 读取标准答案
    with open(answer_file, 'r', encoding='utf-8-sig') as f:
        answer_text = f.read()
        answer_words = answer_text.split()

    seg_index = 0
    ans_index = 0
    correct_count = 0

    while seg_index < len(segmented_words) and ans_index < len(answer_words):
        if segmented_words[seg_index] == answer_words[ans_index]:
            correct_count += 1
            seg_index += 1
            ans_index += 1
        else:
            offset_i = 1
            offset_j = 1
            found = False

            while seg_index + offset_i <= len(segmented_words):
                offset_j = 1
                while ans_index + offset_j <= len(answer_words):
                    seg_str = get_string(segmented_words[seg_index:seg_index + offset_i])
                    ans_str = get_string(answer_words[ans_index:ans_index + offset_j])
                    if seg_str == ans_str:
                        found = True
                        break
                    offset_j += 1

                if found:
                    break
                offset_i += 1
            if found:
                seg_index += offset_i
                ans_index += offset_j
            else:
                seg_index += 1
                ans_index += 1

    total_segmented = len(segmented_words)
    total_answer = len(answer_words)

    precision = correct_count / total_segmented if total_segmented > 0 else 0
    recall = correct_count / total_answer if total_answer > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return correct_count, total_segmented, total_answer, precision, recall, f1_score


def main():
    # 加载词典
    dict_path = "../data/Dict.txt"
    dictionary = load_dictionary(dict_path)

    # 读取待分词文本
    sentence_path = "../data/Sentence.txt"
    with open(sentence_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 确保输出目录存在
    os.makedirs("results", exist_ok=True)

    # ============== 前向匹配分词 ==============
    segmented_forward = max_match_segment(content, dictionary)
    result_forward_str = ' '.join(segmented_forward)
    output_forward_path = "results/result_forward.txt"
    with open(output_forward_path, 'w', encoding='utf-8') as f:
        f.write(result_forward_str)
    print(f"前向分词结果已保存到: {os.path.abspath(output_forward_path)}")

    # ============== 后向匹配分词 ==============
    segmented_backward = reverse_max_match_segment(content, dictionary)
    result_backward_str = ' '.join(segmented_backward)
    output_backward_path = "results/result_backward.txt"
    with open(output_backward_path, 'w', encoding='utf-8') as f:
        f.write(result_backward_str)
    print(f"后向分词结果已保存到: {os.path.abspath(output_backward_path)}")

    # ============== 评估结果 ==============
    answer_path = "../data/Answer.txt"
    if os.path.exists(answer_path):
        # 评估前向分词
        print("\n前向分词结果评估:")
        fwd_correct, fwd_total_seg, fwd_total_ans, fwd_precision, fwd_recall, fwd_f1 = evaluate_results(
            output_forward_path, answer_path
        )
        print(f"**正确识别的词数：{fwd_correct}")
        print(f"**识别出的总体个数：{fwd_total_seg}")
        print(f"**测试集中的总体个数：{fwd_total_ans}")
        print(f"**正确率：{fwd_precision:.5f}")
        print(f"**召回率：{fwd_recall:.5f}")
        print(f"**F值：{fwd_f1:.5f}")

        # 评估后向分词
        print("\n后向分词结果评估:")
        bwd_correct, bwd_total_seg, bwd_total_ans, bwd_precision, bwd_recall, bwd_f1 = evaluate_results(
            output_backward_path, answer_path
        )
        print(f"**正确识别的词数：{bwd_correct}")
        print(f"**识别出的总体个数：{bwd_total_seg}")
        print(f"**测试集中的总体个数：{bwd_total_ans}")
        print(f"**正确率：{bwd_precision:.5f}")
        print(f"**召回率：{bwd_recall:.5f}")
        print(f"**F值：{bwd_f1:.5f}")

        # 比较两种方法
        print("\n两种方法比较:")
        print(f"前向 F值: {fwd_f1:.5f}, 后向 F值: {bwd_f1:.5f}")
        if fwd_f1 > bwd_f1:
            print("前向最大匹配效果更好")
        elif bwd_f1 > fwd_f1:
            print("后向最大匹配效果更好")
        else:
            print("两种方法效果相同")
    else:
        print("\n警告：未找到Answer.txt文件，无法进行评估")


if __name__ == "__main__":
    main()