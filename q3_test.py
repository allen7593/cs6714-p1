import torch
import numpy as np
from config import config
from model import sequence_labeling
from randomness import apply_random_seed
from data_io import DataReader, gen_embedding_from_file, read_tag_vocab
import torch.nn.utils.rnn as rnn

_config = config()
apply_random_seed()

    
tag_dict = read_tag_vocab(config.output_tag_file)
reversed_tag_dict = {v: k for (k, v) in tag_dict.items()}
word_embedding, word_dict = gen_embedding_from_file(config.word_embedding_file, config.word_embedding_dim)
char_embedding, char_dict = gen_embedding_from_file(config.char_embedding_file, config.char_embedding_dim)

_config.nwords = len(word_dict)
_config.ntags = len(tag_dict)
_config.nchars = len(char_dict)
model = sequence_labeling(_config, word_embedding, char_embedding)


def get_word_ids(w):
    word = w.lower()
    if word in word_dict:
        return word_dict[word]
    else:
        return word_dict[UNKNOWN_WORD]

def get_char_ids(c):
    if c in char_dict:
        return char_dict[c]
    else:
        return char_dict[UNKNOWN_CHAR]

sen1 = ['Potion', 'Mastery', 'is', 'specialization', 'of', 'Alchemy', '.']
sen2 = ['A', 'Guild', 'is', 'association', 'of', 'craftsmen', '.']



sentence_list = [sen1] + [sen2]

word_index_lists = [[get_word_ids(word) for word in sentence] for sentence in sentence_list]
char_index_matrix = [[[get_char_ids(char) for char in word] for word in sentence] for sentence in sentence_list]
word_len_lists = [[len(word) for word in sentence] for sentence in char_index_matrix]
sentence_len_list = [len(x) for x in word_len_lists]


batch_char_index_matrices = np.zeros((len(word_index_lists), max(sentence_len_list), max(map(max, word_len_lists))), dtype=int)
for i, (char_index_matrix, word_len_list) in enumerate(zip(char_index_matrix, word_len_lists)):
    for j in range(len(word_len_list)):
        batch_char_index_matrices[i, j, :word_len_list[j]] = char_index_matrix[j]
        
        
batch_word_len_lists = np.ones((len(word_index_lists), max(sentence_len_list)), dtype=int) # cannot set default value to 0
for i, (word_len, sent_len) in enumerate(zip(word_len_lists, sentence_len_list)):
    batch_word_len_lists[i, :sent_len] = word_len
    
batch_word_len_lists = torch.from_numpy(np.array(batch_word_len_lists)).long()
batch_char_index_matrices = torch.from_numpy(batch_char_index_matrices).long()


'''
def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    a = list()
    for batch_char_index_list, batch_char_len_lists in zip(batch_char_index_matrices, batch_word_len_lists):
        out_seq = get_char_word_seq(model, batch_char_index_list, batch_char_len_lists)
        out_seq = narrow_the_matrix(out_seq)
        a.append(out_seq)
    return torch.stack(a)


def narrow_the_matrix(output_seq):
    # concat lastouput[fisrhalf] and firstoutput[latter half]
    # result = list()
    # for word in output_seq:
    #     result.append(word[len(word) - 1][0:config.char_lstm_output_dim]+word[0][config.char_lstm_output_dim:])
    # return result
    
    result = [torch.cat([word[0][config.char_lstm_output_dim:], word[len(word) - 1][0:config.char_lstm_output_dim]]) for word in output_seq]
    return torch.stack(result)

# For each sentence
def get_char_word_seq(model, batch_char_index_lists, batch_char_len_lists):
    input_char_embeds = model.char_embeds(batch_char_index_lists)
    # input_char_embeds = self.non_recurrent_dropout(input_char_embeds)
    # [max_word_length, sencond_max, ..]
    perm_idx, sorted_batch_word_len_list = model.sort_input(batch_char_len_lists)
    sorted_input_embeds = input_char_embeds[perm_idx]
    _, desorted_indices = torch.sort(perm_idx, descending=False)
    output_sequence = rnn.pack_padded_sequence(sorted_input_embeds,
                                               lengths=sorted_batch_word_len_list.data.tolist(), batch_first=True)
    output_sequence, state = model.char_lstm(output_sequence)
    output_sequence, _ = rnn.pad_packed_sequence(output_sequence, batch_first=True)
    output_sequence = output_sequence[desorted_indices]
    # output_sequence = self.non_recurrent_dropout(output_sequence)
    return output_sequence
'''


## 
def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    
    # Given an input of the size [2,7,14], we will convert it a minibatch of the shape [14,14] to 
    # represent 14 words(7 in each sentence), and 14 characters in each word.
    print(batch_char_index_matrices)
    sent_length = list()
    minibatch = list()
    minibatch_word_length_list = list()

    print(batch_word_len_lists)

    for sent in batch_char_index_matrices:
        sent_length.append(len(sent))
        for word in sent:
            minibatch.append(word) # a list of char-indices list
    for sent in batch_word_len_lists:
        for word_length in sent:
            minibatch_word_length_list.append(word_length)

    
    minibatch = torch.stack(minibatch) # convert to be a tensor of variable lengths
    minibatch_word_length_list = torch.stack(minibatch_word_length_list) # convert to be a tensor
    print(minibatch)
    print(minibatch_word_length_list)
    if len(minibatch) != len(minibatch_word_length_list):
        print("minibatch is wrong!")

    # words * chars_in_word
    input_char_embeds = model.char_embeds(minibatch)
    # input_char_embeds = model.non_recurrent_dropout(input_char_embeds)
    # input_char_embeds = self.non_recurrent_dropout(input_char_embeds)
    # [max_word_length, sencond_max, ..]
    perm_idx, sorted_minibatch_word_len_list = model.sort_input(minibatch_word_length_list)
    sorted_input_embeds = input_char_embeds[perm_idx]
    print("sorted")
    print(sorted_input_embeds)
    print(perm_idx)
    _, desorted_indices = torch.sort(perm_idx, descending=False)
    
    print(desorted_indices)
    output_sequence = rnn.pack_padded_sequence(sorted_input_embeds,
                                               lengths=sorted_minibatch_word_len_list.data.tolist(), batch_first=True)

    # input shape: batch_size(num_of_words) * seq_length(num_of_char_in_a_word) * char_embedding_dim
    output_sequence, (hidden_state, cell_state) = model.char_lstm(output_sequence)
    # hidden_state is of shape [num_of_layer, batch_size(num_of_words), hidden_size]
    print(len(hidden_state))
    normal_states = hidden_state[0]
    #print(normal_states[5])
    normal_states = normal_states[desorted_indices]
    #print(normal_states)
    reverse_states = hidden_state[1]
    print(reverse_states)
    reverse_states = reverse_states[desorted_indices]

    result = list()
    for i in range(0, len(minibatch)):
        normal_state = normal_states[i]
        reverse_state = reverse_states[i]#[len(minibatch)-1-i]
        word_state = torch.cat((normal_state, reverse_state))
        #print (len(word_state))
        result.append(word_state)

    output_sequence, _ = rnn.pad_packed_sequence(output_sequence, batch_first=True)
    output_sequence = output_sequence[desorted_indices]

    # print (len(result))
    # [totoal_num,100]
    # print (sent_length)
    # print (len(result))

    final_result = list()
    word_index = 0
    j =0
    for l in sent_length:
        sent = []
        #j=0
        for i in range(0,l):
            # print(j)
            # j += 1
            index = word_index+i
            #print("index")
            sent.append(result[index])
        # print(len(sent))
        final_result.append(torch.stack(sent))
        word_index = word_index + l

    return torch.stack(final_result)
    # [sent, word_in_Sent, 100]



    # output_sequence = self.non_recurrent_dropout(output_sequence)
    ## NOTE: Please DO NOT USE for Loops to iterate over the mini-batch.
    
    
    # Get corresponding char_Embeddings, we will have a Final Tensor of the shape [14, 14, 50]
    # Sort the mini-batch wrt word-lengths, to form a pack_padded sequence.
    # Feed the pack_padded sequence to the char_LSTM layer.
    
    
    # Get hidden state of the shape [2,14,50].
    # Recover the hidden_states corresponding to the sorted index.
    # Re-shape it to get a Tensor the shape [2,7,100].
    
    
    
    #return result


answer = get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists)
answer = answer.data.numpy()
result = np.load('./answer.npy')
# print(len(answer.tolist()))
# print(len(answer.tolist()[0]))
print(answer.tolist())


try:
    assert np.allclose(np.asarray(answer.tolist()), np.asarray(result.tolist()), atol=0.001)
    print('Your implementation is Correct')
except:
    print('Your implementation is not Correct')