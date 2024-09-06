import random 
import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader 

def get_data(min_length=10, max_length=20):
    words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    p = np.array([7, 5, 5, 7, 6, 5, 7, 6, 5, 7])
    p = p / p.sum()

    n1 = random.randint(min_length, max_length)
    s1 = np.random.choice(words, size=n1, replace=True, p=p)
    i1 = int(''.join(s1.tolist()))

    n2 = random.randint(min_length, max_length)
    s2 = np.random.choice(words, size=n2, replace=True, p=p)
    i2 = int(''.join(s2.tolist()))

    x = f'{i1}+{i2}='
    y = f'{i1+i2}'

    return x, y

class TwoSumDataset(Dataset):
    def __init__(self, size=100000):
        super(Dataset, self).__init__()
        self.size = size 

    def __len__(self):
        return self.size 

    def __getitem__(self, i):
        x, y = get_data()
        return {'xy': x+y, 'y': y}
    
def get_dataloader(batch_size=64, tokenizer=None):        
    train_ds = TwoSumDataset()

    def data_collator(items, tokenizer=tokenizer):
        xys = [item['xy'] for item in items]
        ys = [item['y'] for item in items]

        xys_ids = tokenizer.encode_batch(xys, add_special_tokens=True, enable_padding=True)
        ys_ids = tokenizer.encode_batch(ys, add_special_tokens=False, enable_padding=False)

        input_ids = []
        labels = []
        attention_mask = []

        for i in range(len(xys_ids)):
            xy_ids = xys_ids[i] + [tokenizer.eos_id]
            y_ids = ys_ids[i] + [tokenizer.eos_id]
            y_ids = [-100] * (len(xy_ids) - len(y_ids)) + y_ids
            mask = [0 if i == tokenizer.pad_id else 1 for i in xy_ids]

            input_ids.append(torch.LongTensor(xy_ids))
            labels.append(torch.LongTensor(y_ids))
            attention_mask.append(torch.LongTensor(mask))

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.stack(attention_mask)

        batch = {'input_ids': input_ids.cuda(),
                'labels': labels.cuda(),
                'attention_mask': attention_mask.cuda()}
        
        return batch

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=data_collator
    )

    return train_dl
