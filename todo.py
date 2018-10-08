import torch
from config import config

_config = config()


def evaluate(golden_list, predict_list):
    precision, recall = get_precision_recall(golden_list, predict_list)
    return f1_score(precision, recall)


def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    pass;


def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    pass;


def get_precision_recall(golden_list, predict_list) -> tuple:
    golden_tags = get_tags(golden_list)
    gt = get_dict_len(golden_tags)
    predict_tags = get_tags(predict_list)
    predict_len = get_dict_len(predict_tags)
    fp = get_fp(golden_tags, golden_list, predict_list)
    return fp / gt, fp / predict_len


def get_dict_len(tags: dict):
    length = 0
    for key in tags:
        length += len(tags[key])
    return length


def f1_score(precision, recall) -> float:
    return (2.0 * precision * recall) / (precision + recall)


def get_tags(tag_list: list):
    tar_find = False
    hyp_find = False
    current_tag = []
    tags_result = {}
    for num_of_sent, sent in enumerate(tag_list):
        for num_of_word, word in enumerate(sent):
            if tar_find:
                if word != "I-TAR":
                    if num_of_sent not in tags_result:
                        tags_result[num_of_sent] = [current_tag]
                    else:
                        tags_result[num_of_sent].append(current_tag)
                    current_tag = list()
                    tar_find = False
                else:
                    current_tag.append(num_of_word)
                    finalised_list(num_of_word, len(sent), num_of_sent, tags_result, current_tag, tar_find)
            elif hyp_find:
                if word != "I-HYP":
                    if num_of_sent not in tags_result:
                        tags_result[num_of_sent] = [current_tag]
                    else:
                        tags_result[num_of_sent].append(current_tag)
                    current_tag = list()
                    hyp_find = False
                else:
                    current_tag.append(num_of_word)
                    finalised_list(num_of_word, len(sent), num_of_sent, tags_result, current_tag, hyp_find)
            elif word != "O":
                current_tag = list()
                current_tag.append(num_of_word)
            if word == "B-TAR":
                tar_find = True
                current_tag = list()
                current_tag.append(num_of_word)
                finalised_list(num_of_word, len(sent), num_of_sent, tags_result, current_tag, tar_find)
            elif word == "B-HYP":
                hyp_find = True
                current_tag = list()
                current_tag.append(num_of_word)
                finalised_list(num_of_word, len(sent), num_of_sent, tags_result, current_tag, hyp_find)
        tar_find = False
        hyp_find = False
        current_tag = list()
    return tags_result


def finalised_list(num_of_word, len_of_sent, num_of_sent, tags_result, current_tag, match):
    if num_of_word == len_of_sent - 1:
        if num_of_sent not in tags_result:
            tags_result[num_of_sent] = [current_tag]
        else:
            tags_result[num_of_sent].append(current_tag)
        current_tag = list()
        match = False


def get_fp(golden_tags, golden_list, predict_list):
    match = 0
    for key in golden_tags:
        sent_num = key
        golden_tag = golden_tags[sent_num]
        for tags in golden_tag:
            if match_list(sent_num, tags, golden_list, predict_list):
                match += 1
    return match


def match_list(send_num: int, taget: list, golden_list, predict_list) -> bool:
    for index in taget:
        if golden_list[send_num][index] != predict_list[send_num][index]:
            return False
    return True
