from torch.utils.data import Dataset
import pickle

class MTDataset(Dataset):
    def __init__(self, data_path: str=None):
#         assert data_path not None, "Please supply a path to a pickle file for the dataset"
        
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f) # self.data is a list of sequences. Each sequence is a list of metadata at idx 0, states at [1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        batch = self.data[index]
        meta_data = batch[0]
        seqs = batch[1:]
        return meta_data, seqs