# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
We use a semi-supervised deep generative model of transcriptomics data to propagate labels
from a small set of labeled cells to a larger set of unlabeled cells. In particular we
use a dataset of peripheral blood mononuclear cells (PBMC) from 10x Genomics and
(approximately) reproduce Figure 6 in reference [1].

Note that for simplicity we do not reproduce every aspect of the scANVI pipeline. For
example, we do not use dropout in our neural network encoders/decoders, nor do we include
batch annotations in our model.

References:
[1] "Harmonization and Annotation of Single-cell Transcriptomics data with Deep Generative Models,"
    Chenling Xu, Romain Lopez, Edouard Mehlman, Jeffrey Regier, Michael I. Jordan, Nir Yosef.
[2] https://github.com/YosefLab/scvi-tutorials/blob/50dd3269abfe0c375ec47114f2c20725a016736f/seed_labeling.ipynb
"""

import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.nn.functional import softplus, softmax

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import broadcast_shape
from pyro.optim import Adam
from pyro.infer import SVI, config_enumerate, TraceEnum_ELBO

import scanpy as sc

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from data import get_data


# helper for making fully-connected neural networks
def make_fc(dims):
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers[:-1])  # exclude final ReLU non-linearity


# splits a tensor in half along the final dimension
def split_in_half(t):
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)


# helper for broadcasting inputs to neural net
def broadcast_inputs(input_args):
    shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
    input_args = [s.expand(shape) for s in input_args]
    return input_args


# used in parameterizing p(z2 | z1, y)
class Z2Decoder(nn.Module):
    def __init__(self, z1_dim, y_dim, z2_dim, hidden_dims):
        super().__init__()
        dims = [z1_dim + y_dim] + hidden_dims + [2 * z2_dim]
        self.fc = make_fc(dims)

    def forward(self, z1, y):
        z1_y = torch.cat([z1, y], dim=-1)
        # we reshape the input to be two-dimensional so that nn.BatchNorm1d behaves correctly
        _z1_y = z1_y.reshape(-1, z1_y.size(-1))
        hidden = self.fc(_z1_y)
        # if the input was three-dimensional we now restore the original shape
        hidden = hidden.reshape(z1_y.shape[:-1] + hidden.shape[-1:])
        loc, scale = split_in_half(hidden)
        # here and elsewhere softplus ensures that scale is positive
        scale = softplus(scale)
        return loc, scale


# used in parameterizing p(x | z2)
class XDecoder(nn.Module):
    def __init__(self, num_genes, z2_dim, hidden_dims):
        super().__init__()
        dims = [z2_dim] + hidden_dims + [2 * num_genes]
        self.fc = make_fc(dims)

    def forward(self, z2):
        gate_logits, mu = split_in_half(self.fc(z2))
        mu = softmax(mu, dim=-1)
        return gate_logits, mu


# used in parameterizing q(z2 | x) and q(l | x)
class Z2LEncoder(nn.Module):
    def __init__(self, num_genes, z2_dim, hidden_dims):
        super().__init__()
        dims = [num_genes] + hidden_dims + [2 * z2_dim + 2]
        self.fc = make_fc(dims)

    def forward(self, x):
        h1, h2 = split_in_half(self.fc(x))
        z2_loc, z2_scale = h1[..., :-1], softplus(h2[..., :-1])
        l_loc, l_scale = h1[..., -1:], softplus(h2[..., -1:])
        return z2_loc, z2_scale, l_loc, l_scale


# used in parameterizing q(z1 | z2, y)
class Z1Encoder(nn.Module):
    def __init__(self, num_labels, z1_dim, z2_dim, hidden_dims):
        super().__init__()
        dims = [num_labels + z2_dim] + hidden_dims + [2 * z1_dim]
        self.fc = make_fc(dims)

    def forward(self, z2, y):
        # this broadcasting is necessary since Pyro expands y during enumeration (but not z2)
        z2_y = broadcast_inputs([z2, y])
        z2_y = torch.cat(z2_y, dim=-1)
        # we reshape the input to be two-dimensional so that nn.BatchNorm1d behaves correctly
        _z2_y = z2_y.reshape(-1, z2_y.size(-1))
        hidden = self.fc(_z2_y)
        # if the input was three-dimensional we now restore the original shape
        hidden = hidden.reshape(z2_y.shape[:-1] + hidden.shape[-1:])
        loc, scale = split_in_half(hidden)
        scale = softplus(scale)
        return loc, scale


# used in parameterizing q(y | z2)
class Classifier(nn.Module):
    def __init__(self, z2_dim, hidden_dims, num_labels):
        super().__init__()
        dims = [z2_dim] + hidden_dims + [num_labels]
        self.fc = make_fc(dims)

    def forward(self, x):
        logits = self.fc(x)
        return logits


# encompasses the scANVI model and guide as a PyTorch nn.Module
class SCANVI(nn.Module):
    def __init__(self, num_genes, num_labels, l_loc, l_scale, latent_dim=10, alpha=0.1, scale_factor=1.0):
        assert isinstance(num_genes, int)
        self.num_genes = num_genes

        assert isinstance(num_labels, int) and num_labels > 1
        self.num_labels = num_labels

        # this will be the dimension of both z1 and z2
        assert isinstance(latent_dim, int) and latent_dim > 0
        self.latent_dim = latent_dim

        # the next two hyperparameters determine the prior over the log_count latent variable `l`
        assert isinstance(l_loc, float)
        self.l_loc = l_loc
        assert isinstance(l_scale, float) and l_scale > 0
        self.l_scale = l_scale

        # this hyperparameter controls the strength of the auxiliary classification loss
        assert isinstance(alpha, float) and alpha > 0
        self.alpha = alpha

        assert isinstance(scale_factor, float) and scale_factor > 0
        self.scale_factor = scale_factor

        super().__init__()

        # setup the various neural networks used in the model and guide
        self.z2_decoder = Z2Decoder(z1_dim=self.latent_dim, y_dim=self.num_labels,
                                    z2_dim=self.latent_dim, hidden_dims=[50])
        self.x_decoder = XDecoder(num_genes=num_genes, hidden_dims=[100], z2_dim=self.latent_dim)
        self.z2l_encoder = Z2LEncoder(num_genes=num_genes, z2_dim=self.latent_dim, hidden_dims=[100])
        self.classifier = Classifier(z2_dim=self.latent_dim, hidden_dims=[50], num_labels=num_labels)
        self.z1_encoder = Z1Encoder(num_labels=num_labels, z1_dim=self.latent_dim,
                                    z2_dim=self.latent_dim, hidden_dims=[50])

        self.epsilon = 1.0e-6

    def model(self, x, y=None):
        # register various nn.Modules with Pyro
        pyro.module("scanvi", self)

        # this gene-level parameter modulates the variance of the observation distribution
        theta = pyro.param("inverse_dispersion", x.new_ones(self.num_genes),
                           constraint=constraints.positive)

        # we scale all sample statements by scale_factor so that the ELBO is normalized
        # wrt the number of datapoints and genes
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            z1 = pyro.sample("z1", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1))
            y = pyro.sample("y", dist.OneHotCategorical(logits=x.new_zeros(self.num_labels)))

            z2_loc, z2_scale = self.z2_decoder(z1, y)
            z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

            l_scale = self.l_scale * x.new_ones(1)
            l = pyro.sample("l", dist.LogNormal(self.l_loc, l_scale).to_event(1))

            # note that by construction mu is normalized (i.e. mu.sum(-1) == 1) and the
            # total scale of counts for each cell is determined by `l`
            gate_logits, mu = self.x_decoder(z2)
            # TODO revisit this parameterization when https://github.com/pytorch/pytorch/issues/42449 is resolved
            nb_logits = (theta + self.epsilon).log() - (l * mu + self.epsilon).log()
            x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta,
                                                       logits=nb_logits)
            # observe the datapoint x using the observation distribution x_dist
            pyro.sample("x", x_dist.to_event(1), obs=x)

    # the guide specifies the variational distribution
    def guide(self, x, y=None):
        pyro.module("scanvi", self)
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            z2_loc, z2_scale, l_loc, l_scale = self.z2l_encoder(x)
            pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))
            z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

            y_logits = self.classifier(z2)
            y_dist = dist.OneHotCategorical(logits=y_logits)
            if y is None:
                # x is unlabeled so sample y using q(y|z2)
                y = pyro.sample("y", y_dist)
            else:
                # x is labeled so add a classification loss term
                # (this way q(y|z2) learns from both labeled and unlabeled data)
                classification_loss = y_dist.log_prob(y)
                # note that the negative sign appears because we're adding this term in the guide
                # and the guide log_prob appears in the ELBO as -log q
                pyro.factor("classification_loss", -self.alpha * classification_loss)

            z1_loc, z1_scale = self.z1_encoder(z2, y)
            pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))


def main(args):
    # clear param store
    pyro.clear_param_store()
    # fix random number seed
    pyro.util.set_rng_seed(args.seed)
    # enable optional validation warnings
    pyro.enable_validation(True)

    # load and pre-process data
    dataloader, num_genes, l_mean, l_scale, anndata = get_data(dataset=args.dataset, batch_size=args.batch_size,
                                                               cuda=args.cuda)

    # instantiate instance of model/guide and various neural networks
    scanvi = SCANVI(num_genes=num_genes, num_labels=4, l_loc=l_mean, l_scale=l_scale,
                    scale_factor=1.0 / (args.batch_size * num_genes))

    if args.cuda:
        scanvi.cuda()

    # setup an optimizer
    optim = Adam({"lr": args.learning_rate})

    # tell Pyro to enumerate out y when y is unobserved
    guide = config_enumerate(scanvi.guide, "parallel", expand=True)

    # setup a variational objective for gradient-based learning
    svi = SVI(scanvi.model, guide, optim, TraceEnum_ELBO())

    # training loop
    for epoch in range(args.num_epochs):
        losses = []

        for x, y in dataloader:
            loss = svi.step(x, y)
            losses.append(loss)

        print("[Epoch %04d]  Loss: %.4f" % (epoch, np.mean(losses)))

    # now that we're done training we'll inspect the latent representations we've learned
    if args.plot and args.dataset == 'pbmc':
        # compute latent representation (z2_loc) for each cell in the dataset
        latent_rep = scanvi.z2l_encoder(dataloader.data_x)[0]

        # compute inferred cell type probabilities for each cell
        y_logits = scanvi.classifier(latent_rep)
        y_probs = softmax(y_logits, dim=-1).data.cpu().numpy()

        # use scanpy to compute 2-dimensional UMAP coordinates using our
        # learned 10-dimensional latent representation z2
        anndata.obsm["X_scANVI"] = latent_rep.data.cpu().numpy()
        sc.pp.neighbors(anndata, use_rep="X_scANVI")
        sc.tl.umap(anndata)
        umap1, umap2 = anndata.obsm['X_umap'][:, 0], anndata.obsm['X_umap'][:, 1]

        # construct plots; all plots are scatterplots depicting the two-dimensional UMAP embedding
        # and only differ in how points are colored

        # the topmost plot depicts the 200 hand-curated seed labels in our dataset
        fig, axes = plt.subplots(3, 2)
        seed_marker_sizes = anndata.obs['seed_marker_sizes']
        axes[0, 0].scatter(umap1, umap2, s=seed_marker_sizes, c=anndata.obs['seed_colors'], marker='.', alpha=0.7)
        axes[0, 0].set_title('Hand-Curated Seed Labels')
        patch1 = Patch(color='lightcoral', label='CD8-Naive')
        patch2 = Patch(color='limegreen', label='CD4-Naive')
        patch3 = Patch(color='deepskyblue', label='CD4-Memory')
        patch4 = Patch(color='mediumorchid', label='CD4-Regulatory')
        axes[0, 1].legend(loc='center left', handles=[patch1, patch2, patch3, patch4])
        axes[0, 1].get_xaxis().set_visible(False)
        axes[0, 1].get_yaxis().set_visible(False)
        axes[0, 1].set_frame_on(False)

        # the remaining plots depict the inferred cell type probability for each of the four cell types
        s10 = axes[1, 0].scatter(umap1, umap2, s=1, c=y_probs[:, 0], marker='.', alpha=0.7)
        axes[1, 0].set_title('Inferred CD8-Naive probability')
        fig.colorbar(s10, ax=axes[1, 0])
        s11 = axes[1, 1].scatter(umap1, umap2, s=1, c=y_probs[:, 1], marker='.', alpha=0.7)
        axes[1, 1].set_title('Inferred CD4-Naive probability')
        fig.colorbar(s11, ax=axes[1, 1])
        s20 = axes[2, 0].scatter(umap1, umap2, s=1, c=y_probs[:, 2], marker='.', alpha=0.7)
        axes[2, 0].set_title('Inferred CD4-Memory probability')
        fig.colorbar(s20, ax=axes[2, 0])
        s21 = axes[2, 1].scatter(umap1, umap2, s=1, c=y_probs[:, 3], marker='.', alpha=0.7)
        axes[2, 1].set_title('Inferred CD4-Regulatory probability')
        fig.colorbar(s21, ax=axes[2, 1])

        fig.tight_layout()
        plt.savefig('scanvi.pdf')


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.4.0')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-s', '--seed', default=0, type=int, help='rng seed')
    parser.add_argument('-n', '--num-epochs', default=40, type=int, help='number of training epochs')
    parser.add_argument('-d', '--dataset', default='pbmc', type=str,
                        help='which dataset to use', choices=['pbmc', 'mock'])
    parser.add_argument('-bs', '--batch-size', default=100, type=int, help='mini-batch size')
    parser.add_argument('-lr', '--learning-rate', default=0.03, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('--plot', action='store_true', default=False, help='whether to make a plot')
    args = parser.parse_args()

    main(args)
