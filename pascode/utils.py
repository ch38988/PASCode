"""
version: Nov 17
version: Nov 22

"""
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

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

#
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

# ###################################################

# class trnDatasetNew(Dataset):
#     def __init__(self):
#         ann01 = sc.read_h5ad('/home/che82/project/PED2/rosmap.h5ad')
#         sc.pp.highly_variable_genes(ann01)
#         meta = ann01.obs
#         hvg=ann01[:,ann01.var.highly_variable]
#         #hvg = ann01
#         scores = meta['braaksc']
        
#         sid = scores.isin([1,6])
#         sx = hvg[sid].X.toarray()
#         scaler = preprocessing.StandardScaler().fit(sx)
#         tscores = scores[sid]#;tscores[tscores==6]=1

#         self.x = torch.from_numpy(scaler.transform(sx)).float()  #should transform together...
#         self.y = torch.from_numpy(np.array(tscores))
#         self.dns = hvg[sid].obs['Subject']
#         self.ctp = hvg[sid].obs['ceradsc']
#         self.braaksc = hvg[sid].obs['braaksc']
#     def __len__(self):
#          return self.x.shape[0]
#     def __getitem__(self, idx):
#         return torch.from_numpy(np.array(self.x[idx])).to(device), torch.from_numpy(np.array(self.y[idx])).to(device), torch.from_numpy(np.array(idx)).long().to(device)
        
# class tstDatasetNew(Dataset):
#     def __init__(self):
#         ann01 = sc.read_h5ad('/home/che82/project/PED2/rosmap.h5ad')
#         sc.pp.highly_variable_genes(ann01)
#         meta = ann01.obs
#         hvg=ann01[:,ann01.var.highly_variable]
#         #hvg = ann01
#         scores = meta['braaksc']
        
#         sid = scores.isin([1,2,3,4,5,6])
#         sx = hvg[sid].X.toarray()
#         scaler = preprocessing.StandardScaler().fit(sx)
#         tscores = scores[sid]#;tscores[tscores==6]=1

#         self.x = torch.from_numpy(scaler.transform(sx)).float()  #should transform together...
#         self.y = torch.from_numpy(np.array(tscores))
#         self.dns = hvg[sid].obs['Subject']
#         self.ctp = hvg[sid].obs['ceradsc']
#         self.braaksc = hvg[sid].obs['braaksc']
#     def __len__(self):
#          return self.x.shape[0]
#     def __getitem__(self, idx):
#         return torch.from_numpy(np.array(self.x[idx])).to(device), torch.from_numpy(np.array(self.y[idx])).to(device), torch.from_numpy(np.array(idx)).long().to(device)
        

# class tstDatasetNewTest(Dataset):
#     def __init__(self):
#         ann02 = sc.read_h5ad('/home/che82/project/PED2/rosmap.h5ad')
#         #sc.pp.highly_variable_genes(ann02)
#         meta = ann02.obs
#         hvg=ann02[:,ann02.var.highly_variable]

#         scores = meta['braaksc']
        
#         sid = scores.isin([1,2,3,4,5,6])
#         sx = hvg[sid].X.toarray()
#         scaler = preprocessing.StandardScaler().fit(sx)
#         tscores = scores[sid]#;tscores[tscores==6]=1

#         self.x = torch.from_numpy(scaler.transform(sx)).float()  #should transform together...
#         self.y = torch.from_numpy(np.array(tscores))
#         self.dns = hvg[sid].obs['Subject']
#         self.ctp = hvg[sid].obs['ceradsc']
#         self.braaksc = hvg[sid].obs['braaksc']
#     def __len__(self):
#          return self.x.shape[0]
#     def __getitem__(self, idx):
#         return torch.from_numpy(np.array(self.x[idx])).to(device), torch.from_numpy(np.array(self.y[idx])).to(device), torch.from_numpy(np.array(idx)).long().to(device)
#
# def read_prep(data_path, op):
#     """
#     Read raw data.
    
#     Args:
#         op: 0 - gene expression matrix
#             1 - MIT
#             2 - 
#     """
#     print('Reading data...')

#     if op == 0:
#         X = pyreadr.read_r(data_path[0])
#         print('Preprocessing data...')
#         lab = pd.read_csv(data_path[1])
#         X =  X[None].T
#         X = StandardScaler().fit_transform(X)
#         X = pd.DataFrame(X)
#     if op == 1:
#         rdata = pyreadr.read_r("/home/athan/scACC/data/mit.self.org.RData")
#         print('Preprocessing data...')
#         gxp = pd.concat([rdata['gexpr_AD'], rdata['gexpr_CTL']])
#         lab = pd.concat([rdata['AD_cells'], rdata['CTL_cells']])
#         # phenotype is AD(1)/CTL(0)
#         lab['AD'] = [1]*rdata['gexpr_AD'].shape[0] + [0]*rdata['gexpr_CTL'].shape[0]
#         lab.index = lab.TAG
#         # select Highly Variable Genes from gene expression data
#         print("Selecting HVGs...")
#         agxp = anndata.AnnData(gxp)
#         sc.pp.highly_variable_genes(agxp, min_mean=.0125, max_mean=3, min_disp=.25)
#         gxp = gxp.loc[:, agxp.var.highly_variable]
#         # scaling 
#         print("Scaling...")
#         scaler = preprocessing.StandardScaler().fit(gxp)
#         scaled_features = scaler.transform(gxp)
#         X = pd.DataFrame(scaled_features, columns=gxp.columns, index=gxp.index)
#     if op == 2:
#         pass

#     print('Data preprocessing complete.\n')
#     return X, lab

# def get_sp_pheno(lab, op):
#     """
#     Set 1-1 correspondence between samples and the chosen phenotype.

#     Args:
#         pheno: the chosen phenotype name (string)
#         lab: lab data.
#         op: 0 - gene expression matrix
#         op: 1 - MIT
#     Returns:
#         sample id list and phenotype label list. 
#     """
#     if op == 0:
#         if pheno == 'Diagnosis':
#             lab['Diagnosis'] = lab['Diagnosis'].map({'Control':0, 'AD':1})
#         if pheno == 'pathology.group':
#             lab['pathology.group'] = lab['pathology.group'].map({
#                 'no-pathology':0,
#                 'early-pathology':1,
#                 'late-pathology':2})
#         if pheno == 'braaksc':
#             pass  # important: for braaksc TODO
#     if op == 1:
#         sample = 'subjectID'
#         pheno = 'AD'
#    
#     group = group[group>0] # filtering
#     sample = list(group.index.to_frame()[sample])
#     pheno = list(group.index.to_frame()[pheno])

#     return sample, pheno