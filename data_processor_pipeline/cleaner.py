import regex as re
import string
import nltk
from underthesea import word_tokenize, text_normalize

class data_clean():
	def __init__(self):
		self.emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"
                u"\U0001F300-\U0001F5FF" 
                u"\U0001F680-\U0001F6FF"  
                u"\U0001F1E0-\U0001F1FF"  
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
        "]+", flags=re.UNICODE)
		self.vietnamese_stopword=[]
		with open('data_processor_pipeline/vietnamese-stopwords.txt','r',encoding='utf-8') as f:
			for line in f.readlines():
				self.vietnamese_stopword.append(line.strip())
	def clean_text(self, text):
		text=re.sub(r"<[^>]>", "", str(text))
		emoticons=re.findall(r"(?:[:;=])(?:-)?(?:[)(PD])", text)
		text=re.sub(r"(?:[:;=])(?:-)?(?:[)(PD])", "", text)
		text=re.sub(self.emoji_pattern,"",text)
		text=text.translate(str.maketrans('','',string.punctuation))
		text=re.sub(r"([a-zA-Z]+?)\1+","\1",text)
		text=text.strip()
		text=re.sub('\s+',' ', text)
		text=text.lower()
		text=text_normalize(text)
		text=word_tokenize(text, format='text')
		# tmp=[]
		# for token in text.split():
		# 	if token not in self.vietnamese_stopword:
		# 		tmp.append(token)
		text=text+ " " + " ".join(emoticons).replace('-','')
		return text
	def text_process(self, train_df, test_df):
		train_df['sentence']=train_df['sentence'].map(lambda x:self.clean_text(x))
		test_df['sentence']=test_df['sentence'].map(lambda x:self.clean_text(x))
		return train_df, test_df

if __name__=="__main__":
	pass
