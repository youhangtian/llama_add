class Tokenizer():
    def __init__(self):
        self.vocab = '<PAD>,<BOS>,<EOS>,1,2,3,4,5,6,7,8,9,0,+,='.split(',')
        self.vocab_id = {word: i for i, word in enumerate(self.vocab)}
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2 
        self.vocab_len = len(self.vocab)

    def encode_batch(self, texts, add_special_tokens=True, enable_padding=True):
        texts_ids = []
        max_len = 0
        for text in texts:
            text_ids = [self.vocab_id[t] for t in text]
            if add_special_tokens: text_ids = [self.bos_id] + text_ids

            max_len = max(max_len, len(text_ids))
            texts_ids.append(text_ids)

        if enable_padding:
            for i in range(len(texts_ids)):
                ids_len = len(texts_ids[i])
                texts_ids[i] = [self.pad_id] * (max_len - ids_len) + texts_ids[i]

        return texts_ids
    
    def decode(self, ids, skip_special_tokens=True):
        if skip_special_tokens and self.eos_id in ids:
            index = ids.index(self.eos_id)
            ids = ids[:index]

        vocab_arr = [self.vocab[id] for id in ids]
        return ''.join(vocab_arr)
    