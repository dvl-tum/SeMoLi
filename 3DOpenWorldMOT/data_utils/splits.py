import os
import numpy as np

def get_seq_list(path, mode='train', percentage=1.0):
    seqs = os.listdir(path)
    if mode == 'train':
        seqs = seqs[:int(len(seqs)*percentage)]
    else:
        seqs = seqs[int(len(seqs)*percentage):]
    return seqs



if __name__ == "__main__":
    data_path = '/workspace/Waymo_Converted'
    dataset = data_path.split('/')[-1]
    save_dir = f'data_utils/seq_splits_{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    for per in np.arange(0.0, 1.1, 0.1):
        per = np.round(per,decimals=1)
        for split in ['train_gnn', 'train_detector', 'val_gnn', 'val_detector']:
            p = os.path.join(data_path + '_train' if 'train' in split else data_path + '_val', dataset, 'train' if 'train' in split else 'val')
            seqs = get_seq_list(p, 'train' if 'gnn' in split else 'val', per)
            with open(f'{save_dir}/{per}_{split}.txt', 'w') as f:
                f.write('\n'.join(seqs))
            # with open(f'{save_dir}/{per}_{split}.txt', 'r') as f:
            #     data = f.read()
            #     print(data)
