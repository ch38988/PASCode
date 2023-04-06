# %%
# for debug
%reload_ext autoreload
%autoreload 2

# %%
from pascode import PASCode
import scanpy as sc
import torch
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import random
import anndata
import numpy as np
import pandas as pd
import pyreadr

torch.manual_seed(2023)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = torch.device('cuda') # NOTE!

# %%
############################################################################### 
########################## Data reading (Custom) ##############################
###############################################################################
print("Reading data...")
import pickle as pkl
with open('/home/che82/project/scACC/TBdata/dTB2.batch.corrected.pkl','rb') as f: 
    [gxp01,gxp02,meta] = pkl.load(f)
print("Reading complete.")

############################################################################### 
########################## Data preprocessing (Custom) ########################
###############################################################################
#%%
gxp = pd.concat([gxp01, gxp02])
gxp = gxp.T
meta.index=meta['cell_id']
#normalize & scale # NOTE default is axis=0; should scaling by feature axis
scaler = preprocessing.StandardScaler().fit(gxp)
#scaler = preprocessing.MinMaxScaler().fit(gxp)
gxp = scaler.transform(gxp)
X = gxp
lab = meta
pheno_name='TB_status'
sampleID_name = 'donor'
lab[pheno_name].replace({"CASE":1, "CONTROL":0}, inplace=True)

#%%
############################################################################### 
################################### Data preprocessing ########################
###############################################################################
# get sample IDs and phenotypes (list)
group = lab.groupby([sampleID_name, pheno_name]).size()
group = group[group>0] # filtering
id_pheno = group.index.to_frame()
sample_num = id_pheno.shape[0]
# validation set: 64 donors with donor and phenotype labels
val_size = 0.2 # NOTE
val_num_half = int(sample_num*val_size / 2)
val_set = pd.DataFrame({
    sampleID_name:random.sample(list(id_pheno[id_pheno[pheno_name]==0][sampleID_name]), val_num_half) 
           + random.sample(list(id_pheno[id_pheno[pheno_name]==1][sampleID_name]), val_num_half),
    'phenotype':[0]*val_num_half + [1]*val_num_half})
# train/test set
train_test_set = id_pheno[~(id_pheno[sampleID_name].isin(val_set[sampleID_name]))]
X = X[lab[sampleID_name].isin(train_test_set[sampleID_name]).values]
lab = lab[lab[sampleID_name].isin(train_test_set[sampleID_name])]
X = pd.DataFrame(X)

# %%
############################################################################### 
############################### Cross validation ##############################
############################################################################### 
# prepare data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
cv_accuracy = []
# cv_aucroc = []
iter_num = 0

for train_index, test_index in skf.split(train_test_set['donor'], train_test_set[pheno_name]):
    iter_num += 1
    print('\n', '#'*39 + " Fold " + str(iter_num) + ' ' + '#'*39)
    print("Training index:", train_index, "\n Testing index:", test_index, '\n')

    train_sample = [train_test_set['donor'][i] for i in train_index]
    test_sample = [train_test_set['donor'][i] for i in test_index]
    train_filter = lab['donor'].isin(train_sample).values
    test_filter = lab['donor'].isin(test_sample).values

    X_train = torch.tensor(X.loc[train_filter].values).float().to(device)
    lab_train = lab[train_filter]
    id_train = lab_train['donor'].values
    y_train = torch.tensor(lab_train[pheno_name].values)
    X_test = torch.tensor(X.loc[test_filter].values).float().to(device)
    lab_test = lab[test_filter]
    id_test = lab_test['donor'].values
    y_test = torch.tensor(lab_test[pheno_name].values)
    
    # use model
    psc = PASCode(            
            latent_dim=16,
            n_clusters=50, 
            lambda_cluster=2, 
            lambda_phenotype=3, 
            device='cuda:0', 
            alpha=1,
            dropout=.2)

    psc.train(
            X_train, 
            y_train,
            epoch_pretrain=30, # NOTE
            epoch_train=24,            
            batch_size=2**14,
            lr_pretrain=1e-4,
            lr_train=1e-5,
            require_pretrain_phase=True,
            require_train_phase=True, 
            evaluation=True,
            plot_evaluation=True, # print metrics per epoch
            id_train=id_train, X_test=X_test, y_test=y_test, id_test=id_test
            )

    cv_accuracy.append(psc.accuracy_test) # get val accuracy
    # torch.cuda.empty_cache() # free up mem

np.mean(cv_accuracy)
# %% [markdown]
# # Grid search on CV

# %%
# latent_dim=[2,3,2**2,2**3,2**4,2**5,2**6,2**7],
# n_clusters=[30,40,50], 
# lambda_cluster=[.3,0.1,0.5,0.8,0.95], 
# lambda_phenotype=[0.5,0.8,1], 
# epoch_pretrain=[10,15,20],
# epoch_train=[13,17,24],  
# batch_size=[2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16]
# lr=[1e-3, 1e-4, 1e-5]



# %% [markdown]
# # Visualization

# %% [markdown]
# ## Heatmap on the donor-cluster-fraction matrix

# %% [markdown]
# ### Training

# %%
import seaborn as sns

X_new_train = psc.get_donor_cluster_fraction_matrix(X_train, id_train)

new_index = []
for donor in X_new_train.index:
    new_name = donor + ' (' + str(train_test_set.loc[donor].AD.values[0]) + ')'
    new_index.append(new_name)
X_new_train.index = new_index

sns.heatmap(X_new_train)

# %% [markdown]
# ### Testing

# %%
import seaborn as sns

X_new_test = psc.get_donor_cluster_fraction_matrix(X_test, id_test)

new_index = []
for donor in X_new_test.index:
    new_name = donor + ' (' + str(train_test_set.loc[donor].AD.values[0]) + ')'
    new_index.append(new_name)
X_new_test.index = new_index

sns.heatmap(X_new_test)

# %% [markdown]
# ## tsne in the lab original data 

# %%
psc.plot_embedding(X=np.vstack([lab.tsne1.values, lab.tsne2.values]).T, y=lab['braaksc.1'].values, label='braak', title='original braak')

# %% [markdown]
# ## UMAP on both training and testing data

# %%
embd_train = psc.get_embedding(X_train, reducer='umap')
embd_test = psc.get_embedding(X_test, reducer='umap')

# %%
# On training data
lab_train['assign'] = psc.get_assigns(X_train)
psc.plot_embedding(X=embd_train, y=lab_train['assign'].values, label='assign', title='assign (train)', require_distinguishable_colors=True)

# %%
psc.plot_embedding(X=embd_train, y=lab_train['TB_status'].values, label='TB_status', title='TB_status (train)', require_distinguishable_colors=False)
# %%
psc.plot_embedding(X=embd_train, y=lab_train['cluster_ids'].values, label='cluster_ids', title='cluster_ids (train)', require_distinguishable_colors=False)
# %%
psc.plot_embedding(X=embd_train, y=lab_train['cluster_name'].values, label='cluster_name', title='cluster_name (train)', require_distinguishable_colors=False)

# %%
# on test data
lab_test['assign'] = psc.get_assigns(X_test)
psc.plot_embedding(X=embd_test, y=lab_test['assign'].values, label='assign', title='assign (test)', require_distinguishable_colors=True)

# %%
psc.plot_embedding(X=embd_test, y=lab_test['braaksc'].values, label='braak', title='braak (test)', require_distinguishable_colors=False)

# %%
psc.plot_embedding(X=embd_test, y=lab_test['AD'].values, label='AD', title='AD (test)', require_distinguishable_colors=False)

# %%
psc.plot_embedding(X=embd_test, y=lab_test['broad.cell.type'].values, label='broad celltype', title='broad celltype (test)', require_distinguishable_colors=False)

# %%
psc.plot_embedding(X=embd_test, y=lab_test['Subcluster'].values, label='subcelltype', title='sub celltype (test)', require_distinguishable_colors=False)

# %% [markdown]
# # Visualization

# %% [markdown]
# ## Heatmap on the donor-cluster-fraction matrix

# %% [markdown]
# ### Training

# %%
import seaborn as sns

X_new_train = psc.get_donor_cluster_fraction_matrix(X_train, id_train)

new_index = []
for donor in X_new_train.index:
    new_name = donor + ' (' + str(train_test_set.loc[donor].AD.values[0]) + ')'
    new_index.append(new_name)
X_new_train.index = new_index

sns.heatmap(X_new_train)

# %% [markdown]
# ### Testing

# %%
import seaborn as sns

X_new_test = psc.get_donor_cluster_fraction_matrix(X_test, id_test)

new_index = []
for donor in X_new_test.index:
    new_name = donor + ' (' + str(train_test_set.loc[donor].AD.values[0]) + ')'
    new_index.append(new_name)
X_new_test.index = new_index

sns.heatmap(X_new_test)

# %% [markdown]
# ## tsne in the lab original data 

# %%


# %%
psc.plot_embedding(X=np.vstack([lab.tsne1.values, lab.tsne2.values]).T, y=lab['braaksc'].values, label='braak', title='original braak')

# %% [markdown]
# ## UMAP on both training and testing data

# %%
embd_train = psc.get_embedding(X_train, reducer='umap')
embd_test = psc.get_embedding(X_test, reducer='umap')

# %%
# On training data
lab_train['assign'] = psc.get_assigns(X_train)
psc.plot_embedding(X=embd_train, y=lab_train['assign'].values, label='assign', title='assign (train)', require_distinguishable_colors=True)

# %%
psc.plot_embedding(X=embd_train, y=lab_train['braaksc'].values, label='braak', title='braak (train)', require_distinguishable_colors=False)

# %%
psc.plot_embedding(X=embd_train, y=lab_train['AD'].values, label='AD', title='AD (train)', require_distinguishable_colors=False)

# %%
psc.plot_embedding(X=embd_train, y=lab_train['broad.cell.type'].values, label='broad celltype', title='broad celltype (train)', require_distinguishable_colors=False)

# %%
psc.plot_embedding(X=embd_train, y=lab_train['Subcluster'].values, label='subcelltype', title='sub celltype (train)', require_distinguishable_colors=False)

# %%
# on test data
lab_test['assign'] = psc.get_assigns(X_test)
psc.plot_embedding(X=embd_test, y=lab_test['assign'].values, label='assign', title='assign (test)', require_distinguishable_colors=True)

# %%
psc.plot_embedding(X=embd_test, y=lab_test['braaksc'].values, label='braak', title='braak (test)', require_distinguishable_colors=False)

# %%
psc.plot_embedding(X=embd_test, y=lab_test['AD'].values, label='AD', title='AD (test)', require_distinguishable_colors=False)

# %%
psc.plot_embedding(X=embd_test, y=lab_test['broad.cell.type'].values, label='broad celltype', title='broad celltype (test)', require_distinguishable_colors=False)

# %%
psc.plot_embedding(X=embd_test, y=lab_test['Subcluster'].values, label='subcelltype', title='sub celltype (test)', require_distinguishable_colors=False)

# %%


# %%


