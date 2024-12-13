from data_processor_pipeline.vectorizer import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

class Custom_Dataset(Dataset):
	def __init__(self, sentences, sentiment, topic, file_vocab=None, is_save_vocab=True):
		self.sentences=sentences
		self.sentiment=sentiment
		self.topic=topic
		self.tokenized=tokenize(self.sentences)
		self.vocab=build_vocab(sentences_token=self.tokenized, file_name=file_vocab, save_vocab=is_save_vocab)
		self.max_length=int(np.percentile([len(seq) for seq in self.tokenized],95))
		self.numericalized=numericalize(self.tokenized, self.vocab)
		self.padded_numericalized=padded_sequences(self.numericalized, self.vocab, max_length=self.max_length)
	def __len__(self):
		return len(self.padded_numericalized)
	def __getitem__(self, idx):
		return {
			'text': self.sentences[idx],
			'numericalized_encode': self.padded_numericalized[idx],
			'sentiment': self.sentiment[idx],
			'topic': self.topic[idx]
		}
if __name__=="__main__":
	pass