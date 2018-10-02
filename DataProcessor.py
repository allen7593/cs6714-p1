def process_padding(sentence_batch: tuple) -> int:
    PAD = '<PAD>'
    HYP = 'O'
    # find max len in sentence batch
    max_sent_len = 0
    for sentence_list, hyp_list in zip(sentence_batch[0], sentence_batch[1]):
        if len(sentence_list) > max_sent_len:
            max_sent_len = len(sentence_list)

    # add padding to sentence
    for sentence_list, hyp_list in zip(sentence_batch[0], sentence_batch[1]):
        if len(sentence_list) < max_sent_len:
            off_set = (max_sent_len - len(sentence_list))
            sentence_list += [PAD] * off_set
            hyp_list += [HYP] * off_set
    return max_sent_len


def process_unknown_word(sentence_batch: list, word_embedding: list):
    UNK_WROD = '<UNK_WORD>'
    for sentence in sentence_batch:
        for index, word in enumerate(sentence):
            if word not in word_embedding:
                sentence[index] = UNK_WROD
