import sentencepiece as spm


class Llama2Tokenizer:
    def __init__(self, tokenizer_file):
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_file)
        self.tokenizer = sp

    def encode(self, text):
        return self.tokenizer.Encode(text)

    def decode(self, ids):
        return self.tokenizer.Decode(ids)

