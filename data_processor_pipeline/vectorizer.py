import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import re

tokenizer=get_tokenizer('basic_english')

def tokenize(sentences):
    return [tokenizer(sentence) for sentence in sentences]

def yield_token(sentences):
	for sentence in sentences:
		yield sentence

def numericalize(sentences_token, vocab):
	return [[vocab[token] for token in sentence] for sentence in sentences_token]

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


'''
from data_processor_pipeline.custom_dataset import *
csdt=Custom_Dataset(sentences, labels, is_save_vocab=True, max_length=25)
input_ids = pad_sequences(csdt.numericalized,
                          maxlen=25, dtype="long", value=0.0,
                          truncating="post", padding="post")
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=25, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")
'''