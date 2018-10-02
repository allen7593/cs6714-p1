from unittest import TestCase

from DataProcessor import process_padding, process_unknown_word
from DataReader import DataReader
from WordEmbeddingReader import WordEmbeddingReader


class TestTrainingData(TestCase):
    data_reader = None
    word_emd_reader = None

    @classmethod
    def setUpClass(cls):
        cls.data_reader = DataReader("../data/dev.txt", 5)
        cls.word_emd_reader = WordEmbeddingReader("../data/word_embeddings.txt")

    def test_training_data(self):
        with self.data_reader as d:
            each_list = [item for item in d]
        print(each_list)

    def test_process_padding(self):
        with self.data_reader as d:
            each_list = [item for item in d]

        for batch in each_list:
            max_sent_len = process_padding(batch)
            print(max_sent_len)
            print(batch)

    def test_word_embedding_reader(self):
        with self.word_emd_reader as d:
            each_list = [word for word, emd in d]
        print(each_list)

        with self.word_emd_reader as d:
            pretrained_emd = [emd for word, emd in d]
        print(pretrained_emd)

    def test_process_unknown_word(self):
        with self.word_emd_reader  as w:
            word_list = [word for word, emd in w]

        with self.data_reader as d:
            batch_list = [item[0] for item in d]

        for batch in batch_list:
            process_unknown_word(batch, word_list)
            print(batch)
