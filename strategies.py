import faiss
import pdb
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import pairwise_distances


def core_set(unlabeled_emb, labeled_emb, unlabeled_indices, budget):
    candidates = np.ones(budget, dtype=np.int32) * -1

    unlabeled_emb = unlabeled_emb.numpy()

    labeled_emb = labeled_emb.numpy()

    n, dim = unlabeled_emb.shape

    # find nearest neighbor distances to each unlabeled data
    index = faiss.IndexFlatL2(dim)
    index.add(labeled_emb)
    min_distance, _ = index.search(unlabeled_emb, 1)
    min_distance = np.array(min_distance[:, 0])

    print("select candidates =>")
    for i in tqdm(range(budget)):
        candidate = min_distance.argmax()
        assert unlabeled_indices[candidate] not in candidates, "duplicate selection"
        candidates[i] = unlabeled_indices[candidate]

        # gpu_index = faiss.index_cpu_to_all_gpus(index, co)
        index.add(unlabeled_emb[candidate].reshape(1, -1))
        new_min_distance, _ = index.search(unlabeled_emb, 1)
        new_min_distance = np.array(new_min_distance[:, 0])

        for j in range(n):
            min_distance[j] = min(min_distance[j], new_min_distance[j])

    return candidates


def fast_kmeans(inference_feats, budget, num_images):
    clustermembers = []
    inference_feats = inference_feats.numpy()

    n, dim = inference_feats.shape
    max_points_per_centroid = int(2 * num_images // budget)

    kmeans = faiss.Kmeans(dim, budget, niter=20, nredo=5, verbose=True, max_points_per_centroid=max_points_per_centroid,
                          gpu=False)
    kmeans.train(inference_feats)
    centroids = kmeans.centroids

    index = faiss.IndexFlatL2(dim)
    # co = faiss.GpuMultipleClonerOptions()
    # co.useFloat16 = True
    # co.shard = True
    # gpu_index = faiss.index_cpu_to_all_gpus(index, co)
    index.add(centroids)
    _, assignments = index.search(inference_feats, 1)
    assignments = np.array(assignments[:, 0])

    clustermembers.extend(np.where(assignments == i)[0] for i in range(budget))

    # find k nearest neighbors to each centroid
    index = faiss.IndexFlatL2(dim)
    # co = faiss.GpuMultipleClonerOptions()
    # co.useFloat16 = True
    # co.shard = True
    # gpu_index = faiss.index_cpu_to_all_gpus(index, co)
    index.add(inference_feats)
    k = 200
    _, nn = index.search(centroids, k)

    empty_clusters = 0
    valid_nn = np.ones(budget, dtype=np.int32) * -1
    for i in tqdm(range(budget), desc="valid nn:"):
        valid_members = np.array(nn[i])[np.in1d(np.array(nn[i]), np.array(clustermembers[i]))]
        if len(valid_members):
            valid_nn[i] = valid_members[0]

        if not len(np.array(clustermembers[i])):
            empty_clusters += 1

    assert len(np.where(valid_nn == -1)[0]) == empty_clusters, \
        f'knn is small: {empty_clusters} empty clusters, {len(np.where(valid_nn == -1)[0])} clusters without nn'
    valid_nn = np.delete(valid_nn, np.where(valid_nn == -1)[0])

    return valid_nn


def uniform(inference_labels, splits, args):
    candidates = []
    for i in range(args.num_classes):
        candidates.append(np.random.choice(np.where(inference_labels == i)[0], len(np.where(inference_labels == i)[0]),
                                           replace=False))

    splits = [i // args.num_classes for i in splits]

    for split in splits:
        samples = np.zeros(split * args.num_classes, dtype=np.int32) * -1
        for i in range(args.num_classes):
            # this is for imagenet-lt that some classes have few examples
            ptr = min(split, len(np.where(inference_labels == i)[0]))
            samples[i * split:i * split + ptr] = candidates[i][:split]

        samples = np.delete(samples, np.where(samples == -1)[0])

        print('{} inidices are sampled in total, {} of them are unique'.format(len(samples), len(np.unique(samples))))
        print(
            '{}% of all categories is seen'.format(len(np.unique(inference_labels[samples])) / args.num_classes * 100))
        np.save(f'{args.inidices}/uniform_{args.dataset}_{split * args.num_classes}.npy', samples)


def random(all_indices, inference_labels, splits, args):
    shuffled_indices = np.random.choice(all_indices, len(all_indices), replace=False)
    for split in splits:
        samples = shuffled_indices[:split]
        print('{} inidices are sampled in total, {} of them are unique'.format(len(samples), len(np.unique(samples))))
        print(
            '{}% of all categories is seen'.format(len(np.unique(inference_labels[samples])) / args.num_classes * 100))
        np.save(f'{args.indices}/random_{args.dataset}_{split}.npy', samples)


def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0:
            pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll:
            ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


class UncertaintySampler:
    def __init__(self,
                 unlabeled_loader,
                 unlabeled_indices,
                 backbone,
                 device,
                 budget,
                 ftall=False,
                 num_images=0,
                 labeled_loader=None):
        self.unlabeled_loader = unlabeled_loader
        self.unlabeled_indices = unlabeled_indices
        self.labeled_loader = labeled_loader
        self.budget = budget
        self.ftall = ftall
        self.model = backbone
        self.device = device
        self.num_images = num_images
        self.pred = self.get_pred(backbone, device)

    def sample(self, strategy_name):
        return getattr(self, strategy_name)()

    # gradient embedding for badge (assumes cross-entropy loss)
    def _get_grad_embedding(self, unlabeled_indices):
        embDim = 512
        self.model.eval()
        nLab = len(np.unique(self.unlabeled_loader.dataset.dataset.targets))
        embedding = np.zeros([len(self.unlabeled_loader.dataset.dataset.targets), embDim * nLab])

        with torch.no_grad():
            for x, y, features, idxs in self.unlabeled_loader:
                x, y = x.to(self.device), y.to(self.device)
                batchProbs = self.pred[idxs].cpu().numpy()
                if self.ftall:
                    out = self.model.extract_features(x).cpu().numpy()
                else:
                    out = features.cpu().numpy()

                # out = out.data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for c in range(nLab):
                    for j in range(len(y)):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c: embDim * (c + 1)] = out[j] * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c: embDim * (c + 1)] = out[j] * (-1 * batchProbs[j][c])
            return embedding[unlabeled_indices]

    def get_pred(self, model, device):
        with torch.no_grad():
            model.eval()
            logits, ptr = None, 0
            for images, _, feature, indices in tqdm(self.unlabeled_loader, desc="unlabeled logit extraction"):
                if self.ftall:
                    images = images.to(device)
                    cur_logits = model(images).cpu()
                else:
                    feature = feature.to(device)
                    cur_logits = model(feature).cpu()
                B, D = cur_logits.shape
                inds = torch.arange(B) + ptr

                if not ptr:
                    logits = torch.zeros((len(self.unlabeled_loader.dataset), D)).float()

                logits.index_copy_(0, inds, cur_logits)
                ptr += B
            model.train()
        return F.softmax(logits.detach(), dim=1)

    def get_feats(self, model, device, loader=None):
        if loader is None:
            loader = self.unlabeled_loader
        with torch.no_grad():
            model.eval()
            feats, ptr = None, 0
            for images, _, feature, indices in tqdm(loader, desc="unlabeled features extraction"):
                if self.ftall:
                    images = images.to(device)
                    cur_feats = model.extract_features(images).cpu()
                else:
                    cur_feats = feature
                B, D = cur_feats.shape
                inds = torch.arange(B) + ptr

                if not ptr:
                    feats = torch.zeros((len(loader.dataset), D)).float()

                feats.index_copy_(0, inds, cur_feats)
                ptr += B
            model.train()
        return feats

    def entropy(self):
        entropies = torch.sum(self.pred * torch.log(self.pred), dim=1)
        return self.unlabeled_indices[entropies.sort()[1][:self.budget]]

    def smallest_margin(self):
        first_max_pred = self.pred.max(dim=1)[0]
        second_max_pred = self.pred.where(
            (self.pred.reshape(-1) != self.pred.max(dim=1)[0].repeat_interleave(self.pred.shape[1])).reshape(self.pred.shape),
            torch.tensor(-1.0)
        ).max(dim=1)[0]
        margin = first_max_pred - second_max_pred

        return self.unlabeled_indices[margin.sort()[1][:self.budget]]

    def largest_margin(self):
        max_pred = self.pred.max(dim=1)[0]
        min_pred = self.pred.min(dim=1)[0]
        margin = max_pred - min_pred

        return self.unlabeled_indices[margin.sort()[1][:self.budget]]

    def least_confidence(self):
        max_pred = self.pred.max(dim=1)[0]
        margin = max_pred

        return self.unlabeled_indices[margin.sort()[1][:self.budget]]

    def random(self):
        shuffled_indices = np.random.choice(self.unlabeled_indices, len(self.unlabeled_indices), replace=False)
        return shuffled_indices[:self.budget]

    def kmeans(self):
        inference_feats = self.get_feats(self.model, self.device)
        return fast_kmeans(inference_feats, self.budget, self.num_images)

    def coreset(self):
        unlabeled_feats = self.get_feats(self.model, self.device)
        labeled_feats = self.get_feats(self.model, self.device, self.labeled_loader)
        return core_set(unlabeled_feats, labeled_feats, self.unlabeled_indices, self.budget)

    def badge(self):
        gradEmbedding = self._get_grad_embedding(self.unlabeled_indices)
        chosen = init_centers(gradEmbedding, self.budget)
        return self.unlabeled_indices[chosen]


