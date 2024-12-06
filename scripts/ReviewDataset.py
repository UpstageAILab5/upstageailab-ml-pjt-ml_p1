import torch

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df['Review_Text'] 
        self.labels = df['Label']
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):  # 이 메소드 추가
        return len(self.texts)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            try:
                idx = int(idx)
            except ValueError:
                raise TypeError("Index must be convertible to an integer")
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': torch.tensor(label, dtype=torch.long)
        }
