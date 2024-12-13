import re
import pandas as pd
class processor():
	def __init__(self):
		self.TAGS={
			'changing value': 'cv',
		    'command': 'cm',
		    'device': 'dv',
		    'duration': 'dr',
		    'location': 'loc',
		    'scene': 'sc',
		    'target number': 'tn',
		    'time at': 'ta'
		}
	def process_sen_word(self, i, sentence):
		token_words=[]
		token_sentences=[]
		tokens=sentence.split()
		for index, word in enumerate(tokens):
			token_sentences.append(f'{i}')
			token_words.append(word)
		return token_words, token_sentences
	def split_sentence_annotation(self, text):
		pattern= r'\s*\[([^]]+)\]\s*'
		result=re.split(pattern, text)
		result= [segment.strip() for segment in result if segment.strip()]
		return result
	def labeling_sentence_annotation(self, text):
		ano_texts = []
		if ':' in text:
			arr = text.split(':')
			tag = self.TAGS[arr[0].strip()]
			values = arr[1].split()
			if len(values) == 1:
				ano_texts.append(f'B-{tag}')
			elif len(values) == 2:
				ano_texts.append(f'B-{tag}')
				ano_texts.append(f'E-{tag}')
			else:
				for i,value in enumerate(values):
					if i >0 and i< len(values)-1:
						ano_texts.append(f'I-{tag}')
					else:
						if i == 0:
							ano_texts.append(f'B-{tag}')
						else:
							ano_texts.append(f'E-{tag}')

		else:
			for val in text.split():
				ano_texts.append('O')
		return ano_texts
	def lebeling_splited_annotation(self, splited_annotation):
		labeled=[]
		for annotation in splited_annotation:
			labels = self.labeling_sentence_annotation(annotation)
			for label in labels:
				labeled.append(label)
		return labeled
	def create_csv(self, dataset):
		sentences=[]
		words=[]
		tags=[]
		pos=[]
		for i, element in enumerate(dataset):
			sentence_annotation=element['sentence_annotation']
			sentence=element['sentence']
			token_word, token_sentence = self.process_sen_word(i, sentence)
			splited_annotation= self.split_sentence_annotation(sentence_annotation)
			labeled=self.lebeling_splited_annotation(splited_annotation)
			if len(token_word)==len(labeled):
				for word, sentence, tag in zip(token_word, token_sentence, labeled):
					pos.append('O')
					sentences.append(sentence)
					words.append(word)
					tags.append(tag)
		return pd.DataFrame({
		    "Sentence #": sentences,
		    "Word": words,
		    "POS": pos,
		    "Tag": tags
		})
if __name__=="__main__":
	pass