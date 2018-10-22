import torch
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend
from config import config

_config = config()


def evaluate(golden_list, predict_list):
    try:
        precision, recall = get_precision_recall(golden_list, predict_list)
        return f1_score(precision, recall)
    except ZeroDivisionError:
        return 0


def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    # TODO:new_LSTMCell
    pass;


def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    batch_size = batch_char_index_matrices.size()
    mini_batch = batch_char_index_matrices.view(batch_size[0] * batch_size[1], batch_size[2])
    mini_batch_len_list = batch_word_len_lists.view(batch_size[0] * batch_size[1])

    input_char_embeds = model.char_embeds(mini_batch)
    perm_idx, sorted_batch_word_len_list = model.sort_input(mini_batch_len_list)
    _, desorted_indices = torch.sort(perm_idx, descending=False)
    sorted_input_embeds = input_char_embeds[perm_idx]

    output_sequence = rnn.pack_padded_sequence(sorted_input_embeds, lengths=sorted_batch_word_len_list.data.tolist(), batch_first=True)
    output_sequence, state = model.char_lstm(output_sequence)

    output_sequence = torch.transpose(state[0], 0, 1).contiguous()

    tmp_size = output_sequence.size()
    output_sequence = output_sequence.view(batch_size[0] * batch_size[1], tmp_size[1] * tmp_size[2])
    output_sequence = output_sequence[desorted_indices]
    output_sequence_size = output_sequence.size()
    output_sequence = output_sequence.view(batch_size[0], batch_size[1], output_sequence_size[1])

    return output_sequence


def get_char_word_seq(model, batch_char_index_lists, batch_char_len_lists):
    input_char_embeds = model.char_embeds(batch_char_index_lists)
    perm_idx, sorted_batch_word_len_list = model.sort_input(batch_char_len_lists)
    sorted_input_embeds = input_char_embeds[perm_idx]
    _, desorted_indices = torch.sort(perm_idx, descending=False)
    output_sequence = rnn.pack_padded_sequence(sorted_input_embeds,
                                               lengths=sorted_batch_word_len_list.data.tolist(), batch_first=True)
    output_sequence, state = model.char_lstm(output_sequence)
    output_sequence, _ = rnn.pad_packed_sequence(output_sequence, batch_first=True)
    output_sequence = output_sequence[desorted_indices]
    return output_sequence


def get_precision_recall(golden_list, predict_list) -> tuple:
    golden_tags = get_tags(golden_list)
    gt = get_dict_len(golden_tags)
    predict_tags = get_tags(predict_list)
    predict_len = get_dict_len(predict_tags)
    tp = get_tp(golden_tags, golden_list, predict_list)
    return tp / gt, tp / predict_len


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
            if (not tar_find and word == "I-TAR") or (not hyp_find and word == 'I-HYP'):
                continue
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


def get_tp(golden_tags, golden_list, predict_list):
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
