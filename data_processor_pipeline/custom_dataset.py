from data_processor_pipeline.vectorizer import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
import pickle
import numpy as np

def save_vocab(vocab, file_path):
	with open(file_path, 'wb') as f:
		pickle.dump(vocab, f)

def load_vocab_from_file(file_path):
	with open(file_path, 'rb') as f:
		vocab=pickle.load(f)
	return vocab

class Custom_Dataset(Dataset):
	def __init__(self, sentences, labels, file_path=None, is_save_vocab=False, max_length=None):
		self.sentences=sentences
		self.labels=labels
		#sentences=tokenize(self.sentences)
		if max_length is None:
			self.max_length=int(np.percentile([len(seq) for seq in sentences], 95))
		elif max_length is not None:
			self.max_length=max_length
		if file_path is None:
			self.vocab=build_vocab_from_iterator(yield_token(sentences), specials=["<unk>"])
			self.vocab.set_default_index(self.vocab["<unk>"])
		elif file_path is not None:
			self.vocab=load_vocab_from_file(file_path)
		if is_save_vocab:
			save_vocab(self.vocab, file_path='vocab.pkl')
		self.numericalized=numericalize(sentences, self.vocab)
		self.padded_numericalized=padded_sequences(self.numericalized, self.vocab, max_length=self.max_length)
	def __len__(self):
		return len(self.padded_numericalized)
	def __getitem__(self, idx):
		return {
			'text': self.sentences[idx],
			'numericalized_encode': self.padded_numericalized[idx],
			'labels': self.labels[idx]
		}
if __name__=="__main__":
	pass