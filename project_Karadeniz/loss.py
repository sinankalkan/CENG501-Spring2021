import torch 
import numpy as np 


class Triplet_with_DR(torch.nn.Module):
    def __init__(self,margin=1,reg_factor=0.3):
        super(Triplet_with_DR, self).__init__()
        #stated in the paper
        self.margin = 2.5
        self.reg_factor = reg_factor

    def lossF(self,a,n,p):
        return torch.nn.functional.relu((a-p).pow(2).sum()-(a-n).pow(2).sum()+self.margin- self.reg_factor*torch.nn.functional.cosine_similarity(n-a,p-a))
    
    