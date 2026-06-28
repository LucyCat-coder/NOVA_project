import torch
from torch.utils.data import Dataset
import tiktoken

class TextDataset(Dataset):
    def __init__(self, file_path, block_size):
        self.block_size = block_size
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab  # 50257

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokens = self.tokenizer.encode(text, disallowed_special=())
        print(f"Загружено {len(self.tokens)} токенов из {file_path}")

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y