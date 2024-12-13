from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.encodings = self.tokenizer.encode_batch(
            [f"[CLS] {sentence} [SEP]" for sentence in self.sentences]
        )

        self.padding_token_ids = [
            encoding.ids + [0]*(max_length-len(encoding.ids))
            if len(encoding.ids) < self.max_length
            else encoding.ids[:self.max_length]
            for encoding in self.encodings
        ]

        self.attention_masks = [
            [1 if token != 0 else 0 for token in tokens] for tokens in self.padding_token_ids 
        ]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'sentence' : self.sentences[idx],
            'input_ids': torch.tensor(self.padding_token_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

if __name__=="__main__":
    pass

'''
class CustomDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length, cls_token="[CLS]", sep_token="[SEP]"):
        """
        Dataset tuỳ chỉnh.
        Args:
            sentences (list): Danh sách các câu.
            labels (list): Danh sách nhãn.
            tokenizer: Tokenizer với hàm encode_batch.
            max_length (int): Độ dài tối đa của mỗi câu.
            cls_token (str): Token bắt đầu câu.
            sep_token (str): Token kết thúc câu.
        """
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cls_token = cls_token
        self.sep_token = sep_token

        # Thêm token đặc biệt và mã hóa các câu
        self.encodings = self.tokenizer.encode_batch(
            [f"{self.cls_token} {sentence} {self.sep_token}" for sentence in self.sentences]
        )

        # Thêm padding và attention mask
        self.padded_token_ids = [
            encoding.ids + [0] * (self.max_length - len(encoding.ids))
            if len(encoding.ids) < self.max_length
            else encoding.ids[:self.max_length]
            for encoding in self.encodings
        ]
        self.attention_masks = [
            [1 if token != 0 else 0 for token in tokens] for tokens in self.padded_token_ids
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Trả về một mẫu tại chỉ mục idx.
        Args:
            idx (int): Chỉ mục của mẫu.
        Returns:
            dict: Gồm sentences, input_ids, attention_mask, và label.
        """
        return {
            'sentence': self.sentences[idx],
            'input_ids': torch.tensor(self.padded_token_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

'''