import os


def get_seq_list(path, mode='train', percentage=1.0):
    seqs = os.listdir(path)
    if mode == 'train':
        seqs = seqs[:int(len(seqs)*percentage)]
    else:
        seqs = seqs[int(len(seqs)*percentage):]
    return seqs
