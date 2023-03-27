
#%%
from .utils import *

import torch
import sklearn.cluster
import sklearn.ensemble
import sklearn.preprocessing
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

import umap
import seaborn
from torch import nn
import torch.nn.functional as F

#%%
###############################################################################
################################ The PASCode Model  ###########################
###############################################################################

class PASCode():
    r"""
    The PASCode model.

    Args:
        latent_dim: latent dimension
        n_clusters: number of clusters
        lambda_cluster: for kl-div loss
        lambda_phenotype: for entropy loss
        dropout: for AE
        device: ...
        alpha: ...
    """
    def __init__(self, 
            latent_dim=3, 
            n_clusters=30, 
            lambda_cluster=.3, 
            lambda_phenotype=.7, 
            device='cpu', 
            alpha=1,
            dropout=.2
        ):
        
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.lambda_cluster = lambda_cluster
        self.lambda_phenotype = lambda_phenotype
        self.dropout = dropout
        self.alpha = alpha
        self.device = device

    def __call__(self, X):
        r"""
        Returns:
            x_bar: reconstructed 
            q: clustering Q matrix 
            z: embedding
        """
        z, X_bar = self.ae(X)
        q = calc_q(z, self.clusters, self.alpha)
        return X_bar, q, z
    
    def init_ae(self, X_train):
        r"""
        A helper function for creating the ae attribute in order for pretrained
        data loading.
        """
        self.ae = self._AE(
            input_dim=X_train.shape[1],
            latent_dim=self.latent_dim,
            dropout=self.dropout)

    def train(self,
            X_train, 
            y_train, 
            epoch_pretrain=7,
            epoch_train=7,                
            batch_size=1024,
            lr=1e-4,
            require_pretrain_phase=True,
            require_train_phase=True, 
        ):
        r"""
        Train PASCode model, including pretraining phase and training phases

        Args:
            X_train: cell-by-genes data matrix. rows are cells, columns are genes,
                and entries are gene expression levels
            y_train: 
        """
        self.init_ae(X_train)
        if require_pretrain_phase:
            print("Pretraining...")
            self._pretrain(X_train, lr, epoch_pretrain, batch_size)
            print("Pretraining complete.\n")
        if require_train_phase:
            print('Training...')
            self._train(X_train, y_train, lr, epoch_train, batch_size)
            print("Training complete.\n")

    def _pretrain(self, X_train, lr, epoch_pretrain, batch_size, optimizer='adam'):
        r"""
        Pretraining phase.
        Train the AE module in PASCode and initialize cluster self.. 
        """
        train_loader = torch.utils.data.DataLoader(
            X_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr)
        elif optimizer == '?': # TODO
            NotImplemented
        rec_loss = []
        for epoch in range(epoch_pretrain):
            total_loss = 0
            for _, x in enumerate(train_loader):
                x = x.to(device)
                # x_noise = add_noise(x) # for denoising AE
                z, x_hat = self.ae(x)
                optimizer.zero_grad()
                loss = F.mse_loss(x_hat, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print("epoch {}\t loss={:.4f}".format(
                epoch, total_loss/len(train_loader)))
            rec_loss.append(loss.item())

        print("Initializing cluster centroids...")
        self.clusters = torch.nn.Parameter(torch.Tensor(self.n_clusters, self.latent_dim))
        torch.nn.init.kaiming_normal_(self.clusters.data) # NOTE
        with torch.no_grad():
            z, x_hat = self.ae(X_train)
        km = sklearn.cluster.KMeans(n_clusters=self.n_clusters, n_init=20)  # TODO may change to leiden?
        km.fit_predict(z.data.cpu().numpy())
        self.clusters.data = torch.tensor(km.cluster_centers_).to(self.device)

    def _train(self, X_train, y_train, lr, epoch_train, batch_size):
        r"""
        Training phase.
        """
        # prepare training data loader
        train_loader = torch.utils.data.DataLoader(
            subDataset(X_train, y_train), # from utils
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True)

        # for temp eval
        loss_c = []
        loss_r =  []
        loss_p = []
        loss_total = []

        # train
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr)
        print("----- \t ------------ \t ------------- \t ------------- \t ------------")
        print("epoch \t (total) loss \t  cluster loss \t reconstr loss \t entropy loss")
        print("----- \t ------------ \t ------------- \t ------------  \t ------------")
        for epoch in range(epoch_train):
            with torch.no_grad():
                _, q, _ = self(X_train)
            p = target_distribution(q.data)
            # minibatch gradient descent to train AE
            self.ae.train()
            for _, (x, y, idx) in enumerate(train_loader):
                x = x.to(self.device)
                y = torch.from_numpy(pd.get_dummies(y).to_numpy()).to(self.device)
                # x2 = add_noise(x)  # for denoising AE
                x_bar, q, z = self(x)
                rec_ls = F.mse_loss(x_bar, x) # reconstruction loss
                kl_ls = F.kl_div(q.log(), p[idx]) # intracluster loss/KL

                # phenotype entropy loss
                wpheno = torch.matmul(torch.transpose(q,0,1), y.float()) + 1e-9 # weighted phenotype
                scls = torch.sum(wpheno, 1) # axis=1, row sum
                pcls = torch.div(wpheno, scls[:,None]) # probability of clusters
                lpcls = pcls.log()
                ent_ls = torch.mean(-1*torch.sum(pcls*lpcls, 1)) # -p*log(p)
                
                # total joint loss
                loss = self.lambda_cluster*kl_ls + self.lambda_phenotype*ent_ls + rec_ls

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            # evaluation on current training epoch
            self.ae.eval() # NOTE
            with torch.no_grad():
                X_bar, Q, z = self(X_train)
                rec_ls = F.mse_loss(X_bar, X_train)
                assigns = assign_cluster(Q)
                ent_ls = calc_entropy(assigns, y_train)
                kl_ls = F.kl_div(Q.log(), p)
                print("{:5} \t {:7.5f} \t {:7.5f} \t {:8.5f} \t  \t {:7.5f} "
                    .format(epoch, loss.item(), kl_ls.item(), rec_ls.item(), ent_ls.item()))
                loss_c.append(kl_ls.item())
                loss_r.append(rec_ls.item())
                loss_p.append(ent_ls.item())
                loss_total.append(kl_ls.item() + ent_ls.item() + rec_ls.item())

    def get_embedding(self, X, reducer='umap'):
        r"""
        Get the embedding of input data. 

        Args:
            X: input data

        Returns:
            embedding as a numpy array
        """
        z, _ = self.ae(X)
        z = z.detach().numpy()
        if self.latent_dim > 2 and reducer=='umap': 
            z = umap.UMAP(n_components=2).fit_transform(z)
        if self.latent_dim > 2 and reducer=='tsne': 
            z = TSNE(n_components=2, learning_rate='auto',
                       init='random', perplexity=35,random_state=2022).fit_transform(z)
        return z
    
    def plot_embedding(self, X, y, label=None, title=None, require_distinguishable_colors=False):
        # Generate 30 distinguishable colors using the 'viridis' colormap
        custom_colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FF8000", "#8000FF",
            "#FF007F", "#007FFF", "#7FFF00", "#FF7F00", "#00FF7F", "#7F00FF", "#C0C0C0", "#808080",
            "#400080", "#800040", "#804000", "#008040", "#408000", "#800080", "#408080", "#008080",
            "#804040", "#804080", "#408040", "#800000", "#008000", "#000080"
        ]
        n_colors = 0
        if type(y) == type(torch.Tensor()):
            n_colors = len(np.unique(y))
        elif type(y) == type(pd.DataFrame()):
            n_colors = len(np.unique(y.values))
        elif type(y) == type(pd.Series()):
            n_colors = len(np.unique(y.values))
        elif type(y) == type(np.array([])):
            n_colors = len(np.unique(y))

        info = pd.DataFrame({
            'z1':X[:, 0],
            'z2':X[:, 1],
            label:y,})
        if require_distinguishable_colors is True:
            seaborn.scatterplot(
                data=info, x="z1", y="z2", 
                hue=label, size=label, sizes=(15,15), palette=seaborn.color_palette(custom_colors, n_colors)).set(title=title)
        else:
            seaborn.scatterplot(
                data=info, x="z1", y="z2", 
                hue=label, size=label, sizes=(15,15)).set(title=title)
            
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    def get_clusters(self):
        r"""
        A helper function to get clusters in the embedding
        """
        return self.clusters.data

    def get_assigns(self, X):        
        r"""
        A helper function to get point assignments to clusters
        """
        with torch.no_grad():
            _, q, __ = self(X) # get Q matrix
        return assign_cluster(q).detach().cpu().numpy()

    def show_embedding(self, X, y, embd=None, title='', distinguishable_colors=False):
        r"""
        Args:
            X: cell-by-gene matrix 
            y: label
            embedding: 2-dim embedding
            title: title of plot
            distinguishable_colors: require distinguishable_colors or not 
        """
        embd = self.get_embedding(X)
        # NOTE specifically for snATACmeta from UCI
        color_dict = dict({
            'ODC':'brown',
            'EX':'red',
            'MG': 'green',
            'ASC': 'purple',
            'INH': 'blue',
            'OPC':'orange',
            'Unknown':'white',
            'PER.END':'grey',
            'AD':'orange',
            'CTL':'blue'})
        
        # Generate 30 distinguishable colors using the 'viridis' colormap
        custom_colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FF8000", "#8000FF",
            "#FF007F", "#007FFF", "#7FFF00", "#FF7F00", "#00FF7F", "#7F00FF", "#C0C0C0", "#808080",
            "#400080", "#800040", "#804000", "#008040", "#408000", "#800080", "#408080", "#008080",
            "#804040", "#804080", "#408040", "#800000", "#008000", "#000080"
        ]
        n_colors = 0
        if type(y) == type(torch.Tensor()):
            n_colors = len(np.unique(y))
        elif type(y) == type(pd.DataFrame()):
            n_colors = len(np.unique(y.values))
        elif type(y) == type(pd.Series()):
            n_colors = len(np.unique(y.values))
        info = pd.DataFrame({
            'z1':embd[:, 0],
            'z2':embd[:, 1],
            'hue':y,})
        if distinguishable_colors is True:
            seaborn.scatterplot(
                data=info, x="z1", y="z2", 
                hue='hue', size='hue', sizes=(3,3), palette=seaborn.color_palette(custom_colors, n_colors)).set(title=title)
        else:
            seaborn.scatterplot(
                data=info, x="z1", y="z2", 
                hue='hue', size='hue', sizes=(3,3), palette=seaborn.color_palette(n_colors=n_colors)).set(title=title)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    def get_donor_cluster_fraction_matrix(self, X, id):
        r"""
        Get donor cluster fraction matrix.
        
        Args:
            X: gene expression. rows are cells
            id: donor IDs with 1-1 correspondence to rows of X 
            
        Returns:
            the donor cluster fraction matrix
        """
        with torch.no_grad():
            _, q, __ = self(X) # get Q matrix
        assigns = assign_cluster(q).detach().cpu().numpy()

        info = pd.DataFrame({
                'id':id,
                'cluster':assigns,
            })

        temp = info.groupby(['id', 'cluster']).size().unstack()
        dcf_mat = pd.DataFrame(0, index=temp.index, 
                                columns=list(range(self.n_clusters)))
        dcf_mat.loc[:, temp.columns] = temp
        dcf_mat[np.isnan(dcf_mat)] = 0
        dcf_mat = dcf_mat.div(dcf_mat.sum(axis=1), axis=0)
        return dcf_mat

    class _AE(nn.Module):
            r"""
            AutoEncoder module of PASCode.
            """
            def __init__(self, latent_dim, input_dim=None, dropout=.2):
                super().__init__()

                self.encoder = torch.nn.Sequential(
                    nn.Linear(input_dim, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, latent_dim),
                )
                self.decoder = torch.nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, input_dim)
                )
            
            def forward(self, x):
                z = self.encoder(x)
                x_hat = self.decoder(z)
                return z, x_hat