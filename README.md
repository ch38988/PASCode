# Introduction
Phenotype Associated Single Cell Clustering with Autoencoder (PASCode).

# Requirements
Python=3.9.0

# Using the model
## Create an instance

```python
import PASCode
pascode = PASCode.PASCode()
```
Or,
```python
from PASCode import PAScode
pascode = PASCode()
```

To set hyperparameters:
```python
pascode = PASCode(latent_dim=3, 
              n_clusters=30, 
              lambda_cluster=.3, 
              lambda_phenotype=.7, 
              device='cpu', 
              alpha=1,
              dropout=.2)
```


## Train
```python
pascode.train(X_train, y_train)
```
To set hyperparameters:
```python
pascode.train(X_train,
          y_train,
          epoch_pretrain=7,
          epoch_train=7,                
          batch_size=1024,
          lr=1e-4,
          require_pretrain_phase=True,
          require_train_phase=True)
```     

## Get donor-cluster-fraction matrix
To use the trained model, one can call the the method _get_donor_clustering_matrix_ as follows. 

```python
X_new = pascode.get_donor_clustering_matrix(X, id)
```

$X$ is a gene-by-cell matrix. The rows of this matrix are single cell identification strings (e.g., barcodes), the columns are genes, and the entries are gene expression levels.

$id$ is a list/array containing donor IDs ordered by one-to-one correspondence to the rows of $X$.

Returned is a matrix of which the rows are donor IDs, the columns are cluster indices, and the entries are fraction values of the number of cells a donor has in a cluster. 

## Visualization

```python
embd_train = pascode.get_embedding(X_train, reducer='umap')
embd_test = pascode.get_embedding(X_test, reducer='umap')
```

```python
# On training data
pascode.plot_embedding(X=embd_train, y=lab_train['AD'].values, label='AD', title='AD (train)', require_distinguishable_colors=False)
```     
