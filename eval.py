from data_io import DataReader, gen_embedding_from_file, read_tag_vocab
from config import config
from model import sequence_labeling
from tqdm import tqdm
from todo import evaluate
import torch
from randomness import apply_random_seed
import time


if __name__ == "__main__":

	_config = config()
	apply_random_seed()

	tag_dict = read_tag_vocab(_config.output_tag_file)
	reversed_tag_dict = {v: k for (k, v) in tag_dict.items()}
	word_embedding, word_dict = gen_embedding_from_file(_config.word_embedding_file, _config.word_embedding_dim)
	char_embedding, char_dict = gen_embedding_from_file(_config.char_embedding_file, _config.char_embedding_dim)

	_config.nwords = len(word_dict)
	_config.ntags = len(tag_dict)
	_config.nchars = len(char_dict)

	test = DataReader(_config, _config.test_file, word_dict, char_dict, tag_dict, _config.batch_size)

	model = sequence_labeling(_config, word_embedding, char_embedding)
	model.load_state_dict(torch.load(_config.model_file))
	model.eval()

	loss = 0.0
	batch_size = 0
	correct = 0
	num = 0
	for batch_sentence_len_list, batch_word_index_lists, batch_word_mask, batch_char_index_matrices, batch_char_mask, batch_word_len_lists, batch_tag_index_list in test:
		batch_loss = model.forward(batch_word_index_lists, batch_sentence_len_list, batch_word_mask, batch_char_index_matrices, batch_word_len_lists, batch_char_mask, batch_tag_index_list)
		loss += batch_loss
		batch_size += 1

		# F1
		pred_dev_ins, golden_dev_ins = [], []
		pred_batch_tag = model.decode(batch_word_index_lists, batch_sentence_len_list,
											  batch_char_index_matrices, batch_word_len_lists, batch_char_mask)
		pre_ins = [[reversed_tag_dict[t] for t in tag[:l]] for tag, l in
								 zip(pred_batch_tag.data.tolist(), batch_sentence_len_list.data.tolist())]
		gold_ins = [[reversed_tag_dict[t] for t in tag[:l]] for tag, l in
								   zip(batch_tag_index_list.data.tolist(), batch_sentence_len_list.data.tolist())]
		for i in range(0, len(pre_ins)):
			num += 1
			if(pre_ins[i]==gold_ins[i]):
				correct += 1
		pred_dev_ins += pre_ins
		golden_dev_ins += gold_ins
	F1 = evaluate(golden_dev_ins, pred_dev_ins)
	accuracy = correct/num
	loss /= batch_size

	print (_config.model_file + ": F1 SCORE: %8f, Loss: %8f, Accuracy: %8f" % (F1, loss, accuracy))

