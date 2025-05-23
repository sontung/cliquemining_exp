import faiss
import numpy as np
import scipy
import torch
from PIL import Image
from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from torchvision import transforms
from tqdm import trange


IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=0.5, binary=False, cosine=False):
        print("Using Contrastive Loss")
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.binary = binary
        self.cosine = cosine
        self.distance = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, out0, out1, label):
        if self.binary:
            label[label > 0] = 1
        gt = label.float()
        if self.cosine:
            dist = 1 - self.distance(out0, out1).float().squeeze()
        else:
            dist = -torch.sum(out0 * out1, dim=1)
        loss = gt * 0.5 * torch.pow(dist, 2) + (1 - gt) * 0.5 * torch.pow(
            torch.clamp(self.margin - dist, min=0.0), 2
        )
        return loss


class MultiSimLoss(torch.nn.Module):

    def __init__(self, alpha=2.0, beta=50, base=0.5):
        print("Using MultiSim Loss")
        super(MultiSimLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.distance = DotProductSimilarity()

    def forward(self, out0, out1, label):
        gt = label.float()
        mat = self.distance(out0, out1)  # cosine similarity
        cond1 = gt > 0
        cond2 = gt == 0
        pos_exp = self.distance.margin(mat, self.base)
        neg_exp = self.distance.margin(self.base, mat)
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        indices = torch.arange(mat.size(0))
        pos_mask[indices, indices] = cond1.float()
        neg_mask[indices, indices] = cond2.float()
        pos_loss = (1.0 / self.alpha) * lmu.logsumexp(
            self.alpha * pos_exp, keep_mask=pos_mask.bool(), add_one=True
        )
        neg_loss = (1.0 / self.beta) * lmu.logsumexp(
            self.beta * neg_exp, keep_mask=neg_mask.bool(), add_one=True
        )
        return pos_loss + neg_loss


def normalize(x, eps=1e-8):
    return (x - x.min()) / (x.max() - x.min() + eps)


class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        pose_graph_file,
        batch_size=256,
        nb_iterations=1000,
        hard_mining=True,
        train=False,
    ):
        graph = scipy.sparse.load_npz(pose_graph_file).tocoo()
        print(f"Loaded graph file at {pose_graph_file}")
        self.edge_index = torch.tensor(
            np.stack([graph.row, graph.col]), dtype=torch.long
        )
        self.data = graph.data

        self.positive_pairs = self.edge_index[:, self.data > 0.5]
        self.positive_scores = self.data[self.data > 0.5]
        self.soft_negative_pairs = self.edge_index[
            :, (self.data >= 0.25) & (self.data <= 0.5)
        ]
        self.soft_negative_scores = self.data[(self.data >= 0.25) & (self.data <= 0.5)]
        self.hard_mining = hard_mining
        if hard_mining:
            batch_size = batch_size // 2

        self.batch_size = batch_size
        self.nb_positives = batch_size // 3
        self.nb_negatives = batch_size // 3
        self.all_indices = torch.arange(np.max(graph.row) + 1)
        self.csr_arr = graph.tocsr()
        self.nb_iterations = nb_iterations

        rgb_dir = root_dir / "rgb"
        self.root_dir = str(root_dir)
        self.ori_ds_dir = str(root_dir / "../../aachen_source/images_upright")

        self.rgb_files = sorted(rgb_dir.iterdir())

        self.salad_db_desc = np.load("../covis_graph/checkpoints/desc_salad_db.npy")

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]
                ),
            ]
        )
        if train:
            index = faiss.IndexFlatL2(self.salad_db_desc.shape[1])  # build the index
            index.add(self.salad_db_desc.astype(np.float32))  # add vectors to the index
            distances, indices = index.search(self.salad_db_desc.astype(np.float32), 10)
            self.thresh_neg = np.max(distances)

            self.salad_db_desc = torch.tensor(self.salad_db_desc)
            dist_mat = self.compute_dist_matrix()
            dist_mat = normalize(dist_mat)
            sim_score = normalize(self.data)

            prob = torch.tensor(sim_score) * dist_mat
            self.prob = prob / prob.sum()

            self.num_pairs = len(self.all_indices) * len(self.all_indices)
            self.all_images = self.read_all_images()

    def compute_dist_matrix(self):
        results = []
        B = 4096
        for i in trange(0, self.edge_index.shape[1], B):
            idx0 = self.edge_index[0, i : i + B]
            idx1 = self.edge_index[1, i : i + B]
            desc0 = self.salad_db_desc[idx0]
            desc1 = self.salad_db_desc[idx1]
            sq_norm = ((desc0 - desc1) ** 2).sum(dim=1)
            results.append(sq_norm)
        distance_matrix = torch.cat(results, dim=0)
        return distance_matrix

    def __len__(self):
        return self.nb_iterations

    def sample_random_pair(self):

        # Sample a random index
        rand_idx = torch.randint(0, self.num_pairs, (1,)).item()

        # Compute indices in list_a and list_b
        i = rand_idx // len(self.all_indices)
        j = rand_idx % len(self.all_indices)

        pair = (self.all_indices[i].item(), self.all_indices[j].item())
        return pair

    def read_all_images(self):
        mat = torch.zeros((len(self.rgb_files), 3, 224, 224), dtype=torch.float32)
        for idx in trange(len(self.rgb_files)):
            frame_path = str(self.rgb_files[idx])

            if "query" in frame_path:
                parts = frame_path.split("/")[-1].split("_")
                part0 = "/".join(parts[:3])
                part1 = "_".join(parts[3:])

                frame_path1 = f"{self.ori_ds_dir}/{part0}/{part1}"
                image_ori = Image.open(frame_path1)
            else:
                base_key = "/".join(
                    frame_path.split("/")[-1].split(".png")[0].split("_")
                )
                image_ori = Image.open(f"{self.ori_ds_dir}/{base_key}")
            image = self.image_transform(image_ori)
            mat[idx] = image
        return mat

    def read_image(self, indices):
        imgs = []
        for idx2 in indices:
            image = self.all_images[idx2]
            imgs.append(image)
        return torch.stack(imgs)

    def compute_dist(self, i0, i1):
        dist = self.salad_db_desc[i0] - self.salad_db_desc[i1]
        return torch.dot(dist, dist)

    def sample_hard_pairs(self):
        random_ints = torch.multinomial(self.prob, self.batch_size)
        batch = []

        pos_pairs = self.edge_index[:, random_ints]
        pos_scores = self.data[random_ints]

        all_scores = []
        while len(batch) < self.nb_negatives:
            i0, i1 = self.sample_random_pair()
            if i0 == i1:
                continue
            if (
                self.csr_arr[i0, i1] == 0.0
                and self.compute_dist(i0, i1) < self.thresh_neg
            ):
                batch.append([i0, i1])
                all_scores.append(0.0)
        batch.extend(pos_pairs.T)
        all_scores.extend(pos_scores)
        return batch, all_scores

    def __getitem__(self, idx):
        if self.hard_mining:
            batch, all_scores = self.sample_hard_pairs()
        else:
            batch, all_scores = [], []
        random_ints = torch.randint(
            0, self.positive_pairs.size(1), (self.nb_positives,)
        )
        pos_pairs = self.positive_pairs[:, random_ints]
        pos_scores = self.positive_scores[random_ints]
        random_ints = torch.randint(
            0, self.soft_negative_pairs.size(1), (self.nb_negatives,)
        )
        neg_pairs = self.soft_negative_pairs[:, random_ints]
        neg_scores = self.soft_negative_scores[random_ints]
        while len(batch) < self.nb_negatives:
            i0, i1 = self.sample_random_pair()
            if i0 == i1:
                continue
            if (
                self.csr_arr[i0, i1]
                == 0.0
                # and self.compute_dist(i0, i1) < self.thresh_neg
            ):
                batch.append([i0, i1])
                all_scores.append(0.0)
        batch.extend(pos_pairs.T)
        batch.extend(neg_pairs.T)
        all_scores.extend(pos_scores)
        all_scores.extend(neg_scores)
        batch = torch.tensor(batch)
        all_scores = torch.tensor(all_scores)
        img_indices = torch.unique(batch)

        sorted_indices, sorted_pos = img_indices.sort()
        mapping = torch.full(
            (sorted_indices.max().item() + 1,),
            -1,
            dtype=torch.long,
            device=img_indices.device,
        )
        mapping[sorted_indices] = torch.arange(
            len(sorted_indices), device=img_indices.device
        )
        remapped_pairs = mapping[batch]
        images = self.read_image(img_indices)
        return remapped_pairs, all_scores, images, img_indices
