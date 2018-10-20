import torch
import torch.nn.utils.rnn as rnn
from config import config

_config = config()


def evaluate(golden_list, predict_list):
    try:
        precision, recall = get_precision_recall(golden_list, predict_list)
        return f1_score(precision, recall)
    except ZeroDivisionError:
        return 0

def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
	
    if input.is_cuda:
        igates = F.linear(input, w_ih)
        hgates = F.linear(hidden[0], w_hh)
        state = fusedBackend.LSTMFused()
        return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    #ingate = F.sigmoid(ingate) # 1-forgetGate
    forgetgate = F.sigmoid(forgetgate)
    ingate_coupled = torch.add(1, torch.neg(forgetgate))
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    #cy = (forgetgate * cx) + (ingate * cellgate)
    cy = (forgetgate * cx) + (ingate_coupled * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy

    pass;


def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    a = list()
    for batch_char_index_list, batch_char_len_lists in zip(batch_char_index_matrices, batch_word_len_lists):
        out_seq = get_char_word_seq(model, batch_char_index_list, batch_char_len_lists)
        out_seq = narrow_the_matrix(out_seq)
        a.append(out_seq)
    a = torch.stack(a)
    return a


def narrow_the_matrix(output_seq):
    result = [torch.cat([word[0][50:], word[len(word) - 1][0:50]]) for word in output_seq]
    return torch.stack(result)


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
