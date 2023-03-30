"""
version: Nov 17
version: Nov 22

"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

#%%
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

def assign_cluster(q):
    r"""
    Assign cluster of cells based on softmax.
    
    Args:
        matrix q where q_{ij} measures the probability that embedded point z_i
        belongs to centroid j
    Shape:
        num_cells * num_clusters
    Returns:
        max probabilities of assignments of embedded points
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

def calc_entropy(assigns, y): 
    r"""Mean entropy of clusters' Y label.

    For all points assigned to a cluster centroid in the embedded space,
    compute the entropy of their phenotype (e.g. AD or CTL), and
    returns the mean entropy.

    TODO    not the best option, 
            change utils.assign_cluster output as matrix will be better
    
    Ags:
        assigns: assignments of input data points to cluster centroids
        y: labels of input data
    """
    ent = []
    for centroid in torch.unique(assigns):
        cts = torch.unique(y[assigns==centroid], return_counts=True)[1]
        p = cts / torch.sum(cts)
        t = -1 * torch.sum(p*torch.log(p))
        ent.append(t.detach().cpu().numpy().item())
    return np.mean(ent)

def gaussian_kernel_dist(dist):
    """
    multi-RBF kernel
    
    NOTE define the distance func.;define diff. kernels, e.g., gaussian kernel; 
    on original space or reconstructed space??
    """
    sigmas = torch.FloatTensor([1e-2,1e-1,1,10]).to(device)
    beta = 1. / (2. * sigmas).to(device)
    s = torch.matmul(torch.reshape(beta,(len(sigmas),1)),torch.reshape(dist,(1,-1)))
    return torch.reshape(torch.sum(torch.exp(-s),dim=0),dist.size())/len(sigmas)

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
    noise = 0.0*torch.randn(d.size()).to(device)    # TODO 0.0?
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
        return self.x[idx].to(device), \
               self.y[idx].to(device), \
               torch.tensor(idx).to(torch.int64).to(device)
    