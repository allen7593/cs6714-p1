class WordEmbeddingReader:
    data_path: str
    reader = None

    def __init__(self, data_path: str):
        self.data_path = data_path

    def __iter__(self):
        return self

    def __next__(self):
        if self.reader:
            try:
                return self.__get_next_tuple()
            except EOFError:
                raise StopIteration()

    def __enter__(self):
        self.reader = open(self.data_path, "r", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.reader:
            self.reader.close()

    def __get_next_tuple(self) -> tuple:
        line = self.reader.readline()
        if not line:
            raise EOFError()
        line = line.replace("\r\n", "").replace("\r", "").replace("\n", "").strip()
        items = line.split()
        word = items[0]
        list_of_emb = list(map(float, items[1:]))

        return word, list_of_emb
