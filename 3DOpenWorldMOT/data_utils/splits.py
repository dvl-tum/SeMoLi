import os
import numpy as np

def get_seq_list(path, detection_set='train_gnn',percentage=1.0):
    # start by listing all sequences
    seqs = os.listdir(path)

    # for evaluation take all validation sequences
    if detection_set == 'val_evaluation' or detection_set == 'val_test':
        return seqs

    # choose x and or x-1 to get detector or gnn sequences
    if 'gnn' in detection_set:
        seqs = seqs[:int(len(seqs)*percentage)]
    elif 'detector' in detection_set:
        seqs = seqs[int(len(seqs)*percentage):]

    # training always 80% and validation always 20%
    if 'train' in detection_set:
        seqs = seqs[:int(len(seqs)*0.8)]
    else:
        seqs = seqs[int(len(seqs)*0.8):]

    return seqs

def get_seq_list_fixed_val(path, detection_set='train_gnn',percentage=1.0):
    if not 'Argo' in path:
        save_path = f'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/data_utils/new_seq_splits_Waymo_Converted_fixed_val/{percentage}_{detection_set}.txt'
    else:
        save_path = f'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/data_utils/new_seq_splits_AV2_fixed_val/{percentage}_{detection_set}.txt'
    if os.path.isfile(save_path):
        with open(save_path, 'r') as f:
            seqs = f.read()
            seqs = seqs.split('\n')
        return seqs

    # start by listing all sequences
    seqs = os.listdir(path)
    
    # for evaluation take all validation sequences
    if detection_set == 'val_evaluation' or detection_set == 'val_test':
        return seqs
    
    if 'val' in detection_set:
        seqs = seqs[:int(len(seqs)*0.2*0.1*2)]
        if 'gnn' in detection_set:
            seqs = seqs[:int(len(seqs)*0.5)]
        else:
            seqs = seqs[int(len(seqs)*0.5):]
        return seqs
    else:
        seqs = seqs[int(len(seqs)*0.2*0.1*2):]

    # choose x and or x-1 to get detector or gnn sequences
    if 'gnn' in detection_set:
        seqs = seqs[:int(len(seqs)*percentage)]
    elif 'detector' in detection_set:
        seqs = seqs[int(len(seqs)*percentage):]
    
    return seqs
 


if __name__ == "__main__":
    data_path = '/dvlresearch/jenny/Waymo_Converted_GT'
    data_path = '/workspace/Argoverse2/'
    fixed_val = True
    if fixed_val:
        save_dir = f'new_seq_splits_Waymo_Converted_fixed_val'
    else:
        save_dir = f'new_seq_splits_Waymo_Converted'
     
    save_dir = f'new_seq_splits_AV2_fixed_val'
    os.makedirs(save_dir, exist_ok=True)
    for per in np.arange(0.0, 1.1, 0.1):
        per = np.round(per,decimals=1)
        for detection_set in ['train_gnn', 'train_detector', 'val_gnn', 'val_detector', 'val_evaluation']:
            
            # get split directory
            if detection_set == 'val_evaluation':
                if per != 1.0:
                    continue
                split = 'val'
            elif detection_set == 'val_test':
                split = 'test'
            else:
                split = 'train'

            # get sequences
            p = os.path.join(data_path, split)
            if not fixed_val:
                seqs = get_seq_list(p, detection_set, per)
            else:
                seqs = get_seq_list_fixed_val(p, detection_set, per)
            print(per, detection_set, len(seqs))
            # save sequences
            with open(f'{save_dir}/{per}_{detection_set}.txt', 'w') as f:
                f.write('\n'.join(seqs))
