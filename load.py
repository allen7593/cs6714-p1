import torch
from main import LSTMTagger

from DataReader import DataReader

EMBEDDING_DIM = 3
HIDDEN_DIM = 4


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

word_to_ix = {}
data = DataReader("dev.txt")
with data as d:
    training_data = [(list_word, list_hyp) for list_word, list_hyp in d]
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

model = torch.load('mytraining.pt')

# See what the scores are after training
torch.save(model, 'mytraining.pt')
while True:
    input_data = input("Give me your query: ")
    if input_data == "quit":
        break
    with torch.no_grad():
        inputs = prepare_sequence(input_data.split(), word_to_ix)
        tag_scores = model(inputs)

        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
        print(tag_scores)
