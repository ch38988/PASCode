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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from IPython.display import clear_output

import umap
import seaborn
from torch import nn
import torch.nn.functional as F

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
        self.device = torch.device(device)

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
        
    def init_ae(self, X_train):
        r"""
        A helper function for creating the ae attribute in order for pretrained
        data loading.
        """
        self.ae = self._AE(
            input_dim=X_train.shape[1],
            latent_dim=self.latent_dim,
            dropout=self.dropout).to(self.device)

    def train(self,
            X_train, 
            y_train, 
            epoch_pretrain=7,
            epoch_train=7,                
            batch_size=1024,
            lr_pretrain=1e-3,
            lr_train=1e-5,
            require_pretrain_phase=True,
            require_train_phase=True, 
            evaluation=True,
            plot_evaluation=True, # plot metics per epoch
            id_train=None, X_test=None, y_test=None, id_test=None # for printing out metrics per epoch
        ):
        r"""
        Train PASCode model, including pretraining phase and training phases

        Args:
            X_train: cell-by-genes data matrix. rows are cells, columns are genes,
                and entries are gene expression levels
            y_train: 
        """
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        self.init_ae(X_train)
        self.evaluation = evaluation
        self.plot_evaluation = plot_evaluation
        # if  not (id_train==None and X_test==None and y_test==None and id_test==None).all():
        self.id_train = id_train
        self.X_test = X_test
        self.y_test = y_test
        self.id_test = id_test
            
        if require_pretrain_phase:
            print("Pretraining...")
            self._pretrain(X_train, lr=lr_pretrain, epoch_pretrain=epoch_pretrain, batch_size=batch_size)
            print("Pretraining complete.\n")
        if require_train_phase:
            print('Training...')
            self._train(X_train, y_train, lr=lr_train, epoch_train=epoch_train, batch_size=batch_size)
            print("Training complete.\n")

    def _pretrain(self, X_train, lr, epoch_pretrain, batch_size, optimizer='adam'):
        r"""
        Pretraining phase.
        Train the AE module in PASCode and initialize cluster self.. 
        """
        X_train = X_train.to(self.device)

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
            for x in train_loader:
                x = x.to(self.device)
                z, x_hat = self.ae(x)
                optimizer.zero_grad()
                loss = F.mse_loss(x_hat, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print("epoch {}\t\t loss={:.4f}".format(epoch, total_loss/len(train_loader)))
            rec_loss.append(loss.item())

        print("Initializing cluster centroids...")
        self.clusters = torch.nn.Parameter(torch.Tensor(self.n_clusters, self.latent_dim))
        torch.nn.init.kaiming_normal_(self.clusters.data) # NOTE
        with torch.no_grad():
            z, x_hat = self.ae(X_train)
        km = sklearn.cluster.KMeans(n_clusters=self.n_clusters, n_init=20)
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

        # for eval
        self.precision_train = []
        self.recall_train = []
        self.f1_train = []
        self.accuracy_train = []
        self.roc_auc_train = []
        self.precision_test = []
        self.recall_test = []
        self.f1_test = []
        self.accuracy_test = []
        self.roc_auc_test = []
        self.loss_c = []
        self.loss_r = []
        self.loss_p = []
        self.loss_total = []
        self.epochs = []

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        # train
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr)
        for epoch in range(epoch_train):
            with torch.no_grad():
                _, q, _ = self(X_train)
            p = target_distribution(q.data)
            self.P = p # NOTE temp
            # minibatch gradient descent to train AE
            self.ae.train()
            for x, y, idx in train_loader:
                x = x.to(self.device)
                # x2 = add_noise(x)  # for denoising AE
                x_bar, q, _ = self(x)
                rec_loss = F.mse_loss(x_bar, x) # AE reconstruction loss
                kl_loss = F.kl_div(q.log(), p[idx], reduction='batchmean') # KL-div loss # NOTE "reduction = 'mean' doesn’t return the true kl divergence value, please use reduction = 'batchmean' which aligns with KL math definition. In the next major release, 'mean' will be changed to be the same as ‘batchmean’."
                # phenotype entropy loss
                ent_loss = calc_entropy(q, y)
                
                # total joint loss
                loss = self.lambda_cluster*kl_loss + self.lambda_phenotype*ent_loss + rec_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # print("----- \t ------------ \t ------------- \t ------------- \t ------------")
                # print("epoch \t (total) loss \t  cluster loss \t reconstr loss \t entropy loss")
                # print("----- \t ------------ \t ------------- \t ------------  \t ------------")
                # print("{:5} \t {:7.5f} \t\t {:7.5f} \t {:8.5f} \t \t {:7.5f} "
                #     .format(epoch, loss.item(), kl_loss.item(), rec_loss.item(), ent_loss.item()))
                            
            if self.evaluation:
                self.evaluate(X_train, y_train, epoch)


    def get_embedding(self, X, reducer='umap'):
        r"""
        Get the embedding of input data. 

        Args:
            X: input data

        Returns:
            embedding as a numpy array
        """
        X = X.to(self.device)
        z, _ = self.ae(X)
        z = z.detach().cpu().numpy()
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
            n_colors = len(np.unique(y.values.tolist()))
        elif type(y) == type(np.array([])):
            n_colors = len(np.unique(y.tolist()))

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

    def get_donor_cluster_fraction_matrix(self, X, id):
        r"""
        Get donor cluster fraction matrix.
        
        Args:
            X: gene expression. rows are cells, columns are genes
            id: a list/array. donor IDs with 1-1 correspondence to rows of X 
            
        Returns:
            the donor cluster fraction matrix
        """

        X = torch.tensor(X).to(self.device)
        with torch.no_grad():
            _, q, __ = self(X)
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

    def evaluate(self, X_train, y_train, epoch):
        # get donor cluster fraction matrix from traininig data
        X_train = X_train.cpu().numpy()
        y_train = y_train.cpu().numpy()
        X_new_train = self.get_donor_cluster_fraction_matrix(X_train, self.id_train)
        # get donor labels
        info_train = pd.DataFrame({
            'id':self.id_train,
            'label':y_train
        })
        y_true_train = info_train.groupby(['id', 'label']).size().index.to_frame()['label'].to_list()

        # get donor cluster fraction matrix from testing data
        X_new_test = self.get_donor_cluster_fraction_matrix(self.X_test, self.id_test)
        # get donor labels 
        info_test = pd.DataFrame({
            'id':self.id_test,
            'label':self.y_test
        })
        y_true_test = info_test.groupby(['id', 'label']).size().index.to_frame()['label'].to_list()

        clf = LogisticRegression().fit(X_new_train, y_true_train)
        y_pred_train = clf.predict(X_new_train)
        y_pred_test = clf.predict(X_new_test)

        # training data
        self.precision_train.append(precision_score(y_true_train, y_pred_train))
        self.recall_train.append(recall_score(y_true_train, y_pred_train))
        self.f1_train.append(f1_score(y_true_train, y_pred_train))
        self.accuracy_train.append(accuracy_score(y_true_train, y_pred_train))
        self.roc_auc_train.append(roc_auc_score(y_true_train, clf.predict_proba(X_new_train)[:, 1]))

        # testing data
        self.precision_test.append(precision_score(y_true_test, y_pred_test))
        self.recall_test.append(recall_score(y_true_test, y_pred_test))
        self.f1_test.append(f1_score(y_true_test, y_pred_test))
        self.accuracy_test.append(accuracy_score(y_true_test, y_pred_test))
        self.roc_auc_test.append(roc_auc_score(y_true_test, clf.predict_proba(X_new_test)[:, 1]))

        # evaluation on current training epoch
        X_train = torch.tensor(X_train).to(self.device)
        y_train = torch.tensor(y_train).to(self.device)

        self.ae.eval()
        with torch.no_grad():
            X_bar, q, _ = self(X_train)
            rec_loss = F.mse_loss(X_bar, X_train)
            assigns = assign_cluster(q)
            ent_loss = calc_entropy(assigns, torch.tensor(y_train).to(self.device))
            kl_loss = F.kl_div(q.log(), self.P)

            self.loss_c.append(kl_loss.item())
            self.loss_r.append(rec_loss.item())
            self.loss_p.append(ent_loss)
            self.loss_total.append(kl_loss.item() + ent_loss + rec_loss.item())

        # plot
        if self.plot_evaluation:
            epochs = np.arange(1, epoch+2)
            clear_output(wait=True)
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
            axes[0,0].plot(epochs, self.loss_total, label='total loss')
            axes[0,0].legend(loc='lower left')
            axes[0,1].plot(epochs, self.loss_p, label="ent loss")
            axes[0,1].legend(loc='lower left')
            axes[0,2].plot(epochs, self.loss_c, label="cluster loss")
            axes[0,2].legend(loc='lower left')
            axes[1,0].plot(epochs, self.loss_r, label="recstr loss")
            axes[1,0].legend(loc='lower left')
            axes[1,1].plot(epochs, self.accuracy_train, label='accuracy')
            axes[1,1].plot(epochs, self.accuracy_test, label='val accuracy')
            axes[1,1].legend(loc='lower left')
            axes[1,2].plot(epochs, self.roc_auc_train, label='roc-auc')
            axes[1,2].plot(epochs, self.roc_auc_test, label='val roc-auc')
            axes[1,2].legend(loc='lower left')
            axes[2,0].plot(epochs, self.precision_train, label='precision')
            axes[2,0].plot(epochs, self.precision_test, label='val precision')
            axes[2,0].legend(loc='lower left')
            axes[2,1].plot(epochs, self.recall_train, label='recall')
            axes[2,1].plot(epochs, self.recall_test, label='val recall')
            axes[2,1].legend(loc='lower left')
            axes[2,2].plot(epochs, self.f1_train, label='f1-score')
            axes[2,2].plot(epochs, self.f1_test, label='val f1-score')
            axes[2,2].legend(loc='lower left')
            plt.tight_layout()
            plt.show()