import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

def assign_cluster(q):
    r"""
    Assign cells to clusters based on softmax.
    
    Args:
        matrix q where q_{ij} measures the probability that embedded point z_i
        belongs to centroid j. (q.shape == [num_cells, num_clusters])
    Returns:
        assigned clusters of cells
    """
    return torch.max(q, 1)[1]

def pairwise_dist(q1, q2, p=2):
    """
    pairwise distance in the z space[[based on q of the clusters]]
    """
    # NOTE cos = nn.CosineSimilarity()  #cosine distance
    return torch.cdist(q1, q2, p=p)

def target_distribution(q):
    r"""
    Computes and returns the target distribution P based on Q.
    
    Args:
        q: similarity between embedded point z_i and cluster center j 
            measured by Student's t-distribution
    Returns:
        a tensor (matrix) where the (i,j) element is p_{ij}
    """
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def calc_entropy(q, y): 
    r"""
    Ags:
        q: 
        y: 
    """
    assigns = assign_cluster(q)
    centroids = torch.unique(assigns) # assigned centroids
    ent = 0
    for centroid in centroids:
        counts = torch.unique(y[assigns==centroid], return_counts=True)[1]
        p = counts / torch.sum(counts)
        ent += torch.sum(-p*torch.log(p))
    return ent / centroids.shape[0]

# def gaussian_kernel_dist(dist):
#     """
#     multi-RBF kernel
    
#     NOTE define the distance func.;define diff. kernels, e.g., gaussian kernel; 
#     on original space or reconstructed space??
#     """
#     sigmas = torch.FloatTensor([1e-2,1e-1,1,10])
#     beta = 1. / (2. * sigmas)
#     s = torch.matmul(torch.reshape(beta,(len(sigmas),1)),torch.reshape(dist,(1,-1)))
#     return torch.reshape(torch.sum(torch.exp(-s),dim=0),dist.size())/len(sigmas)

def calc_q(z, cluster_centroids, alpha=1):
    r"""
    Compute Q (q_{ij} matrix) matrix from embedding data.
    
    Args:
        z: mapping of input x into the hidden layer
        cluster_centroids_: cluster centroids in latent space
    
    Returns: 
        a matrix where element (i,j) is the value of q_{ij}, measuring
        the similarity between embedded point z_i and centroid u_j
        soft assignment (probability) of assigning point (cell) i to cluster j
    
    NOTE
        order preserving
    """
    q = 1.0 / \
        (1.0 + \
            ( (torch.sum(torch.pow(z.unsqueeze(1) - cluster_centroids, 2), 2)) \
                / alpha ) )
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q

def add_noise(d): # for denoising AE
    r"""
    TODO self adding rand noise ? ref: Vincent et al. 2010 in DEC.
    """
    noise = 0.0*torch.randn(d.size())    # TODO 0.0?
    nd = d + noise
    return nd

def get_id_pheno(lab, pheno_name):
    """
    Get subjectID/sampleID - phenotype label indexed dataframe.
    """
    group = lab.groupby(['subjectID', pheno_name]).size()
    group = group[group>0] # filtering
    id_pheno = group.index.to_frame()
    return id_pheno

class subDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        """
        Returns the number of samples in the dataset. 
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index idx.
        """
        return self.x[idx], \
               self.y[idx], \
               torch.tensor(idx).to(torch.int64)
    