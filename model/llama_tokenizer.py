import sentencepiece as spm


class LlamaTokenizer:
    def __init__(self, tokenizer_file):
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_file)
        self.tokenizer = sp

    def encode(self, text):
        a = self.tokenizer.encode_as_ids(text)
        return self.tokenizer.Encode(text)

    def decode(self, ids):
        return self.tokenizer.Decode(ids)


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    tokenizer_file = hf_hub_download(
        repo_id="NousResearch/Llama-2-7b-chat-hf",
        filename="tokenizer.model",
        local_dir="Llama-2-7b"
    )
    tokenizer = LlamaTokenizer(tokenizer_file)
    id = tokenizer.encode("dd")
    print(id)
    a = tokenizer.decode(id)
    print(a)
