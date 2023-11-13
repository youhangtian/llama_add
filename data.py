import random 
import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader 

words = '<PAD>,<BOS>,<EOS>,1,2,3,4,5,6,7,8,9,0,+,='
vocab = {word: i for i, word in enumerate(words.split(','))}
vocab_r = [k for k, v in vocab.items()]

def get_data(min_length=10, max_length=20):
    words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    p = np.array([7, 5, 5, 7, 6, 5, 7, 6, 5, 7])
    p = p / p.sum()

    n1 = random.randint(min_length, max_length)
    s1 = np.random.choice(words, size=n1, replace=True, p=p)
    s1 = s1.tolist()

    n2 = random.randint(min_length, max_length)
    s2 = np.random.choice(words, size=n2, replace=True, p=p)
    s2 = s2.tolist()

    x = ['<BOS>'] + s1 + ['+'] + s2 + ['=']

    y = int(''.join(s1)) + int(''.join(s2))
    y = list(str(y)) + ['<EOS>']

    return x, y

class TwoSumDataset(Dataset):
    def __init__(self, size=100000, min_length=10, max_length=20):
        super(Dataset, self).__init__()
        self.size = size 
        self.min_length = min_length 
        self.max_length = max_length 

    def __len__(self):
        return self.size 

    def __getitem__(self, i):
        x, y = get_data(self.min_length, self.max_length)

        context_ids = [vocab[i] for i in x]
        target_ids = [vocab[i] for i in y]

        input_ids = context_ids + target_ids 

        labels = [-100] * len(context_ids) + target_ids 
        masks = [0 if t==vocab['<PAD>'] else 1 for t in input_ids]

        example = {'input_ids': input_ids,
                   'labels': labels,
                   'attention_mask': masks}
        
        return example
    
def data_collator(examples):
    len_ids = [len(example['input_ids']) for example in examples]
    longest = max(len_ids)

    input_ids = []
    labels_list = []
    masks_list = []

    for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
        ids = example['input_ids']
        labs = example['labels']
        masks = example['attention_mask']

        ids = [vocab['<PAD>']] * (longest - length) + ids 
        labs = [-100] * (longest - length) + labs 
        masks = [0] * (longest - length) + masks 

        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labs))
        masks_list.append(torch.LongTensor(masks))

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    attention_mask = torch.stack(masks_list)

    return {
        'input_ids': input_ids.cuda(),
        'labels': labels.cuda(),
        'attention_mask': attention_mask.cuda()
    }


def get_dataloader(batch_size=64):
    train_ds = TwoSumDataset(size=100000)

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=data_collator 
    )

    return train_dl
