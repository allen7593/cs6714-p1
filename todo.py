import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from config import config

_config = config()

# At least up to 3 decimal places, as is the common research practice.
def evaluate(golden_list, predict_list):
    golden_tags = get_tags(golden_list)
    gt = get_dict_len(golden_tags)
    predict_tags = get_tags(predict_list)
    predict_len = get_dict_len(predict_tags)
    f1 = 0
    if (gt == 0) and (predict_len == 0):
        f1 = 1
    else:
        try:
            precision, recall = get_precision_recall(golden_tags, golden_list, predict_list, gt, predict_len)
            f1 = f1_score(precision, recall)
        except ZeroDivisionError:
            f1 = 0
    f1 = round(f1, 3)
    return f1

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
    ingate_coupled = torch.add(torch.neg(forgetgate),1)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    #cy = (forgetgate * cx) + (ingate * cellgate)
    cy = (forgetgate * cx) + (ingate_coupled * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy

    pass;

def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
        
    batch_size = batch_char_index_matrices.size() # [num_of_len, num_of_word, num_of_char]
    minibatch = batch_char_index_matrices.view(batch_size[0] * batch_size[1], batch_size[2]) # [total_num_of_word, num_of_char]
    minibatch_word_length_list = batch_word_len_lists.view(batch_size[0] * batch_size[1]) # [word_length]
    
    input_char_embeds = model.char_embeds(minibatch)    
    perm_idx, sorted_minibatch_word_len_list = model.sort_input(minibatch_word_length_list)
    sorted_input_embeds = input_char_embeds[perm_idx]
    _, desorted_indices = torch.sort(perm_idx, descending=False)
    output_sequence = rnn.pack_padded_sequence(sorted_input_embeds,
                                               lengths=sorted_minibatch_word_len_list.data.tolist(), batch_first=True)

    output_sequence, (hidden_state, cell_state) = model.char_lstm(output_sequence)
    
    # hidden_state: [num_of_layer*dir, batch_size, hidden_dim]
    hidden_state = torch.cat((hidden_state[0,:,:],hidden_state[1,:,:]), -1)
    hidden_state = hidden_state[desorted_indices]
    result = hidden_state.view(batch_size[0], batch_size[1], -1)
    return result


def get_precision_recall(golden_tags, golden_list, predict_list, gt, predict_len) -> tuple:
    tp = get_tp(golden_tags, golden_list, predict_list)
    return tp / gt, tp / predict_len


# The number of tags(results) in the tags_dictionary
# tags: {SentNum: [[wordNum, wordNum], [..]]}
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
    current_tag = [] # [wordNum, wordNum]
    tags_result = {} # {SentNum: [[wordNum, wordNum], ]}
    for num_of_sent, sent in enumerate(tag_list):
        for num_of_word, word in enumerate(sent):
            # Check if there are any tags are done
            # If tar_find
            if tar_find:
                if word != "I-TAR": # == "O" or "I-HYP" or "B-HYP" or "B-TAR"
                    if num_of_sent not in tags_result:
                        tags_result[num_of_sent] = [current_tag]
                    else:
                        tags_result[num_of_sent].append(current_tag)
                    # reset the current tag and tar_find
                    current_tag = list()
                    tar_find = False
                else:
                    current_tag.append(num_of_word)
            # If hy_find
            elif hyp_find:
                if word != "I-HYP": # == "O" or "I-TAR" or "B-TAR" or "B-HYP"
                    if num_of_sent not in tags_result:
                        tags_result[num_of_sent] = [current_tag]
                    else:
                        tags_result[num_of_sent].append(current_tag)
                    # reset the current tag and hyp_find
                    current_tag = list()
                    hyp_find = False
                else:
                    current_tag.append(num_of_word)

            # Deal with the current word
            # Valid I-TAR and I-HYP are dealt with in the tag found case above
            # Invalid I-* and O should be ignored(the invalid I-* shouldn't appear)
            # B-TAR and B-HYP should be dealt with in both cases      
            if word == "B-TAR":
                tar_find = True
                current_tag = list()
                current_tag.append(num_of_word)
            if word == "B-HYP":
                hyp_find = True
                current_tag = list()
                current_tag.append(num_of_word)
        # the last check on tar_find and hyp_find
        if tar_find or hyp_find:
            if num_of_sent not in tags_result:
                tags_result[num_of_sent] = [current_tag]
            else:
                tags_result[num_of_sent].append(current_tag)
        tar_find = False
        hyp_find = False
        current_tag = list()

    return tags_result

'''
def finalised_list(num_of_word, len_of_sent, num_of_sent, tags_result, current_tag, match):
    if num_of_word == len_of_sent - 1:
        if num_of_sent not in tags_result:
            tags_result[num_of_sent] = [current_tag]
        else:
            tags_result[num_of_sent].append(current_tag)
        current_tag = list()
        match = False
'''


# get the true positive value derived from golden_list and predict_list
# golden_tags: {SentNum: [[wordNum, wordNum], [..]]}
def get_tp(golden_tags, golden_list, predict_list):
    match = 0
    for key in golden_tags:
        sent_num = key # index of the sentence
        golden_tag = golden_tags[sent_num] # tags_list in this sentence
        for tags in golden_tag: # each tag in this sentence, like [wordNum, wordNum]
            if match_list(sent_num, tags, golden_list, predict_list):
                match += 1
    return match

# Check if elements match given sentNum and wordNum
def match_list(sent_num: int, taget: list, golden_list, predict_list) -> bool:
    for index in taget:
        if golden_list[sent_num][index] != predict_list[sent_num][index]:
            return False
    return True
