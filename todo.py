import torch
from config import config

_config = config()


def evaluate(golden_list, predict_list):
    pass;

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
    pass;
