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

torch.manual_seed(2022)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
device = torch.device('cuda')

# %%
############################################################################### 
################################### Data reading ##############################
###############################################################################
data_path = '/home/athan/PASCode/data/mit.self.org.RData'
print("Reading data...")
rdata = pyreadr.read_r(data_path)

# %%
############################################################################### 
################################### Data preprocessing ########################
###############################################################################
print('Preprocessing data...')
gxp = pd.concat([rdata['gexpr_AD'], rdata['gexpr_CTL']])
lab = pd.concat([rdata['AD_cells'], rdata['CTL_cells']])
lab.index=lab.TAG
# phenotype is AD(1)/CTL(0)
lab['AD'] = [1]*rdata['gexpr_AD'].shape[0] + [0]*rdata['gexpr_CTL'].shape[0]
# select Highly Variable Genes from gene expression data
print("Selecting HVGs...")
agxp = anndata.AnnData(gxp)
sc.pp.highly_variable_genes(agxp, min_mean=.0125, max_mean=3, min_disp=.25)
gxp = gxp.loc[:, agxp.var.highly_variable]
# scaling
print("Scaling...")
scaler = preprocessing.StandardScaler().fit(gxp)
scaled_features = scaler.transform(gxp)
X = pd.DataFrame(scaled_features, columns=gxp.columns, index=gxp.index)
print("Data preprocessing completed.")

# %%
############################################################################### 
######################### train, test, validation sets ########################
###############################################################################
pheno_name = 'AD'
sampleID_name = 'subjectID'
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
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
cv_acc = np.array([])
iter_num = 0

for train_index, test_index in skf.split(train_test_set['subjectID'], train_test_set['AD']):
    iter_num += 1
    print('#'*39 + " Fold " + str(iter_num) + ' ' + '#'*39)
    print("Training index:", train_index, "\ntraining index:", test_index)

    train_sample = [train_test_set['subjectID'][i] for i in train_index]
    test_sample = [train_test_set['subjectID'][i] for i in test_index]
    train_filter = lab['subjectID'].isin(train_sample).values
    test_filter = lab['subjectID'].isin(test_sample).values

    X_train = torch.tensor(X.loc[train_filter].values).float().to(device)
    lab_train = lab[train_filter]
    id_train = lab_train['subjectID'].values
    y_train = torch.tensor(lab_train['AD'].values)
    X_test = torch.tensor(X.loc[test_filter].values).float().to(device)
    lab_test = lab[test_filter]
    id_test = lab_test['subjectID'].values
    y_test = torch.tensor(lab_test['AD'].values)
    
    # use model
    psc = PASCode()
    psc.train(X_train, y_train)

    # perforamce
    psc.evaluate(X_train, y_train, id_train, X_test, y_test, id_test)

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
lab_train['assign'] = psc.get_assigns(X_train.to(device))
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



