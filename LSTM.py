import torch
import torch.nn as nn
import torch.functional as F

from WordEmbeddingReader import WordEmbeddingReader


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_of_layer: int, batch_first: bool,
                 bidirectional: bool, hidden_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        # load embedding data
        word_emd_reader = WordEmbeddingReader("data/word_embeddings.txt")
        with word_emd_reader as reader:
            pre_trained_word_emd = [(word, emd) for word, emd in reader]

        self.word2idx = [word for word, emd in pre_trained_word_emd]
        emd_matrix = [emd for word, emd in pre_trained_word_emd]
        weight = torch.Tensor(emd_matrix)
        self.word_embeddings = nn.Embedding.from_pretrained(weight)
        # lstm
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_of_layer,
            batch_first=batch_first,
            bidirectional=bidirectional
        )

        # hidden to tag
        self.hidden2tag = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
