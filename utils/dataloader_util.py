from torch.utils.data import Dataset
import torch
from collections import namedtuple
Data = namedtuple('Data',('masked','original','loss_weights'))
class BertMaskDataset(Dataset):
    """
    Inputs 
        data: the sequences for training, where seq lengths are equal
        n_embed: the number of embeddings (i.e. n_embed=5 will have a sequences with integers 0-4)
    Yields
        masked, original, loss_weights
        where masked follows:
          each item in sequence has a 15% chance of being masked
          an item that is masked has a 
            80% chance of being a mask token
            10% chance of being unchanged
            10% chance of being a random other token
    The mask token will be n = n_embed (1 higher than the max in data)
    """
    def __init__(self,data,n_embed):
        self.mask_token = n_embed
        self.n_embed = n_embed
        self.data = torch.tensor(data,dtype=torch.int64)
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        for idx in range(self.__len__()):
            yield self[idx]
    def __getitem__(self,idx):
        seq = self.data[idx] 
        masked_inds = torch.rand(len(seq),dtype=torch.float32) < 0.15
        loss_weights = masked_inds.float()
        masked = seq.clone()
        masked[masked_inds] = self._mask(masked[masked_inds])
        return Data(masked,seq,loss_weights)._asdict()#'masked','original','loss_weights'
    def _mask(self,seq):
        other_seq = torch.randint(0,self.n_embed,(len(seq),))
        chance = torch.rand(len(seq),dtype=torch.float32)
        # Mask Token
        ind = chance < 0.8
        seq[ind] = self.mask_token
        # Other Replace
        ind = (chance >= 0.8) * (chance < 0.9)
        seq[ind] = other_seq[ind]
        # Else remain unchanged
        return seq
