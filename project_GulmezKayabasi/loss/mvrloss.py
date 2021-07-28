import torch.nn as nn
import torch
import torch
import numpy as np


# CONSTRUCTOR => REGULARIZER CONSTAN
# INPUT => BATCH_of_Embeddings, TARGET
# GENERATE ALL TRIPLET IN BATCH


def get_all_triplets_indices(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
    return torch.where(triplets)


class MVR_Triplet(nn.Module):

    def __init__(self, margin, reg):
        super(MVR_Triplet, self).__init__()
        self.margin = margin
        self.reg = reg
        self.pairwise_euc = nn.PairwiseDistance(2)
        self.pairwise_cos = nn.CosineSimilarity(dim=1)
        self.relu = nn.ReLU()

    def forward(self, embeddings, labels):
        indices_tuple = get_all_triplets_indices(labels)
        anchor_idx, positive_idx, negative_idx = indices_tuple
        ap_vec = embeddings[positive_idx] - embeddings[anchor_idx]
        an_vec = embeddings[negative_idx] - embeddings[anchor_idx]
        cos_dist = self.pairwise_cos(ap_vec, an_vec)
        ap_dist = self.pairwise_euc(embeddings[anchor_idx], embeddings[positive_idx])
        an_dist = self.pairwise_euc(embeddings[anchor_idx], embeddings[negative_idx])
        penalties = self.relu(ap_dist - an_dist + self.margin - self.reg * cos_dist)
        return torch.mean(penalties)


class MVR_Proxy(nn.Module):

    def __init__(self, reg, no_class, embedding_dimension):
        super(MVR_Proxy, self).__init__()
        self.reg = reg
        self.no_class = no_class
        self.emb_dim = embedding_dimension
        with torch.no_grad():
            vec = np.random.randn(self.no_class, self.emb_dim)
            vec /= np.linalg.norm(vec, axis=1, keepdims=True)
            self.proxies = nn.Parameter((torch.from_numpy(vec)).float())

    def forward(self, embeddings, labels):
        anchor_embeddings = embeddings
        anchor_pr_dist = torch.cdist(anchor_embeddings, self.proxies)
        labels = labels.type(torch.long)
        ## Computation of euclidean distance term in not only nominator but also denominator
        euc_pproxy_anchor = anchor_pr_dist[range(len(labels)), labels]
        mask_euc = torch.ones(anchor_pr_dist.shape, dtype=torch.bool)
        mask_euc[range(len(labels)), labels] = torch.tensor([False])
        euc_nproxy_anchor = anchor_pr_dist[mask_euc]
        euc_nproxy_anchor = euc_nproxy_anchor.view(anchor_pr_dist.size()[0], -1)
        ## Computation of cosine term in denominator
        torch.utils.backcompat.broadcast_warning.enabled = True
        proxy_anc_dif = self.proxies.unsqueeze(0) - anchor_embeddings.unsqueeze(1)
        mask_cos = torch.ones(proxy_anc_dif.shape, dtype=torch.bool)
        mask_cos[range(len(labels)), labels, :] = torch.tensor([False])
        fn_fa = proxy_anc_dif[mask_cos].view(-1, self.no_class - 1, self.emb_dim)
        norm_fn_fa = torch.linalg.norm(fn_fa, dim=2, keepdims=True)
        normalized_fn_fa = torch.div(fn_fa, norm_fn_fa)
        normalized_fn_fa = normalized_fn_fa.transpose(2, 1)
        fp_fa = (self.proxies[labels] - anchor_embeddings)
        norm_fp_fa = torch.linalg.norm(fp_fa, dim=1, keepdims = True)
        normalized_fp_fa = torch.div(fp_fa, norm_fp_fa)
        normalized_fp_fa = normalized_fp_fa.unsqueeze(1)
        cosine_distances = torch.bmm(normalized_fp_fa, normalized_fn_fa)
        cosine_distances = cosine_distances.squeeze(1)
        ## General Cost Computation
        unexp_denominator = -1 * (euc_nproxy_anchor + self.reg * cosine_distances)  # TODO: stability of exp
        exp_denominator = torch.exp(unexp_denominator)
        tot_denominator = torch.sum(exp_denominator, dim=1)
        exp_nominator = torch.exp(-1*euc_pproxy_anchor)
        unlog_loss = torch.div(exp_nominator, tot_denominator)
        log_loss = -1 * torch.log(unlog_loss)

        return log_loss.mean()


class MVR_MS(nn.Module):
    # credits to Malong Technologies
    def __init__(self, scale_pos=2.0, scale_neg=50.0, thresh=0.5, margin=0.1):
        super(MVR_MS, self).__init__()
        self.thresh = thresh
        self.margin = margin

        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

    def forward(self, embeddings, labels):
        assert embeddings.size(0) == labels.size(0), \
            f"embeddings.size(0): {embeddings.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = embeddings.size(0)
        sim_mat = torch.matmul(embeddings, torch.t(embeddings))
    
        epsilon = 1e-5
        ms_loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # regularization part

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            ms_loss.append(pos_loss + neg_loss)

        if len(ms_loss) == 0:
            return torch.zeros([], requires_grad=True)

        ms_loss = sum(ms_loss) / batch_size
        return ms_loss


def mapMaskBinary(maskSmall, maskBig):
    # maskSmall is assumed to be created of True's of maskBig
    if maskSmall.shape == maskBig.shape:
        maskOut = maskSmall.clone()
    else:
        maskOut = maskBig.clone()
        iS = 0
        for iB in range(len(maskBig)):
            if maskBig[iB]:
                maskOut[iB] = maskBig[iB] and maskSmall[iS]
                iS += 1
    return maskOut


class MVR_MS_reg(nn.Module):
    def __init__(self, scale_pos=2.0, scale_neg=50.0, thresh=0.7, margin=0.1, gamma=0.3):
        super(MVR_MS_reg, self).__init__()
        self.thresh = thresh
        self.margin = margin

        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

        self.gamma = gamma
        self.cosSim = nn.CosineSimilarity(dim=1)

    def forward(self, embeddings, labels):
        assert embeddings.size(0) == labels.size(0), \
            f"embeddings.size(0): {embeddings.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = embeddings.size(0)
        sim_mat = torch.matmul(embeddings, torch.t(embeddings))

        epsilon = 1e-5
        ms_loss = list()

        for i in range(batch_size):
            pos_pair_labels_bool = labels == labels[i]
            pos_pair_ = sim_mat[i][pos_pair_labels_bool]
            pos_pair_epsilon_bool = pos_pair_ < 1 - epsilon
            pos_pair_ = pos_pair_[pos_pair_epsilon_bool]
            neg_pair_labels_bool = labels != labels[i]
            neg_pair_ = sim_mat[i][neg_pair_labels_bool]

            neg_pair_margin_bool = neg_pair_ + self.margin > min(pos_pair_) # boolean mask
            pos_pair_margin_bool = pos_pair_ - self.margin < max(neg_pair_)

            neg_pair = neg_pair_[neg_pair_margin_bool]
            pos_pair = pos_pair_[pos_pair_margin_bool]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            dummy = 0
            # regularization
            # -- f_p: map the index to previous binary masks
            pos_hardest_bool = torch.zeros(pos_pair.shape, dtype=torch.bool) #, device = torch.device('cuda'))
            pos_hardest_bool[torch.argmin(pos_pair)] = True
            pos_hardest_idx1 = pos_pair_margin_bool.nonzero()[pos_hardest_bool.nonzero()].flatten()
            pos_hardest_idx2 = pos_pair_epsilon_bool.nonzero()[pos_hardest_idx1].flatten()
            pos_hardest_idx3 = pos_pair_labels_bool.nonzero()[pos_hardest_idx2].flatten()
            f_p_vec = embeddings[pos_hardest_idx3,].flatten(0)
            f_p = f_p_vec.expand(neg_pair.shape[0], -1)
            # -- f_a
            f_a_vec = embeddings[i,]
            f_a = f_a_vec.expand(neg_pair.shape[0], -1)
            # -- f_n
            f_n_idx = neg_pair_labels_bool.nonzero()[neg_pair_margin_bool.nonzero()].flatten()
            f_n = embeddings[f_n_idx,]
            # --
            reg = self.gamma*nn.CosineSimilarity(dim=1)(f_n-f_a, f_p-f_a)
            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh - reg))))
            ms_loss.append(pos_loss + neg_loss)

        if len(ms_loss) == 0:
            return torch.zeros([], requires_grad=True)

        ms_loss = sum(ms_loss) / batch_size
        return ms_loss

if __name__ == '__main__':
    loss = MVR_Proxy(0.1, 8, 128)
    vec = np.random.randn(32, 128)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    embeddings = torch.from_numpy(vec).float()
    target = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    target = target.repeat(4)
    net_loss = loss(embeddings, target)
    print("{}".format(net_loss))
