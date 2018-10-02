class DataReader:
    data_path: str
    batch_size: int
    EOF:bool
    reader = None

    def __init__(self, data_path: str, batch_size: int):
        self.data_path = data_path
        self.batch_size = batch_size
        self.EOF = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.EOF:
            raise StopIteration()
        if self.reader:
            sentence_count = 0
            sentence_list, sentence_hyp_list = list(), list()
            try:
                while sentence_count < self.batch_size:
                    sentence, hyp = self.__get_next_tuple()
                    sentence_list.append(sentence)
                    sentence_hyp_list.append(hyp)
                    sentence_count += 1
                return sentence_list, sentence_hyp_list
            except EOFError:
                self.EOF = True
                if sentence_list and sentence_hyp_list:
                    return sentence_list, sentence_hyp_list
                else:
                    raise StopIteration()

    def __enter__(self):
        self.reader = open(self.data_path, "r", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.reader:
            self.reader.close()

    def __get_next_tuple(self) -> tuple:
        list_of_word = list()
        list_of_hyp = list()
        line = self.reader.readline()
        if not line:
            raise EOFError()

        while line != "\n" and line != "\r" and line != "\r\n" and line:
            line = line.replace("\r\n", "").replace("\r", "").replace("\n", "").strip()
            items = line.split()
            if len(items) == 2:
                list_of_word.append(items[0])
                list_of_hyp.append(items[1])
            elif len(items) == 1:
                list_of_word.append("")
                list_of_hyp.append(items[0])
            line = self.reader.readline()
        return list_of_word, list_of_hyp
