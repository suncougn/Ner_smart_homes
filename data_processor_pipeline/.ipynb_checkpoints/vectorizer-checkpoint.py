import torch
from torch.nn.utils.rnn import pad_sequence
import re

def tokenize(sentences):
    tokenized = []
    for sentence in sentences:
        tokens = re.findall(r'\w+', sentence.lower())
        tokenized.append(tokens)
    return tokenized

def build_vocab(sentences_token=None, file_name=None, save_vocab=False):
	vocab={}
	if file_name is None:
		vocab["<unk>"]=0
		for sentence_token in sentences_token:
			for token in sentence_token:
				vocab[token]=len(vocab)
	elif file_name is not None:
		with open(file_name,'r',encoding='utf-8-sig') as f:
			for line in f.readlines():
				word, idx = line.strip().split(',')
				vocab[word]=int(idx)
	if save_vocab:
		savevocab(vocab, file_name='vocab.txt')
	return vocab

def savevocab(vocab, file_name='vocab.txt'):
	with open(file_name,'w',encoding='utf-8-sig') as f:
		for i in vocab:
			f.write(f"{i},{vocab.get(i)}\n")

def numericalize(sentences_token, vocab):
	numericalized=[]
	for sentence_token in sentences_token:
		numericalized.append([vocab.get(token,vocab["<unk>"]) for token in sentence_token])
	return numericalized

def padded_sequences(numericalized, vocab, max_length=None):
	padded_sequences = pad_sequence([torch.tensor(i) for i in numericalized], 
									batch_first=True,
									padding_value=vocab["<unk>"])
	if max_length is not None:
		if padded_sequences.size(1) > max_length:
			padded_sequences=padded_sequences[:,:max_length]
		if padded_sequences.size(1) < max_length:
			padding=torch.full((padded_sequences.size(0), padded_sequences(1)-max_length), vocab["<unk>"])
			padded_sequences=torch.cat([padded_sequences,padding], dim=1)
	return padded_sequences

if __name__=="__main__":
	pass