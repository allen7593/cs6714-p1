import torch
import torch.nn as nn

from WordEmbeddingReader import WordEmbeddingReader

BATCH_SIZE = 5
EPOCH = 100
LR = 0.1

if __name__ == '__main__':
    torch.manual_seed(1)

    lstm = nn.LSTM(input_size=3, hidden_size=4, bidirectional=True, batch_first=True)
    inputs = [torch.randn(1, 3) for _ in range(5)]
    hidden = (torch.randn(2, 1, 4),
              torch.randn(2, 1, 4))

    word_emd_reader = WordEmbeddingReader("data/word_embeddings.txt")
    with word_emd_reader as reader:
        pre_trained_word_emd = [(word, emd) for word, emd in reader]

    word2idx = [word for word, emd in pre_trained_word_emd]
    emd_matrix = [emd for word, emd in pre_trained_word_emd]
    weight = torch.Tensor(emd_matrix)
    embedding = nn.Embedding.from_pretrained(weight)
    print(embedding)