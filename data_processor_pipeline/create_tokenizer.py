from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
#from transformers

class creator_tokenizer():
    def __init__(self, sentences):
        with open("corpus.txt", "w", encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence+"\n")
    def create_tokenizer(self):
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(word_size=5000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        files = ["corpus.txt"]
        tokenizer.train(files, trainer)
        tokenizer.save("tokenizer.json")

if __name__=="__main__":
    pass

'''
corpus = data['sentence']
# Ghi dữ liệu vào file
with open("corpus.txt", "w", encoding="utf-8") as f:
    for line in corpus:
        f.write(line + "\n")

        from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

tokenizer.pre_tokenizer = Whitespace()

trainer = WordPieceTrainer(vocab_size=5000,
                           special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

files = ["corpus.txt"]  

tokenizer.train(files, trainer)

tokenizer.save("custom_tokenizer.json")
'''