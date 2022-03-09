from torch_geometric.data import Data, Dataset
from scipy.spatial import Delaunay
import torch

import os
import pickle


class FaceDataset(Dataset):
    def __init__(self,
                 root,
                 filename,
                 mode,
                 transform=None,
                 pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.mode = mode
        self.filename = filename
        self.length = None
        super(FaceDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        return "not_implemented.pt"

    def download(self):
        pass

    def __triangle2edges(self, t):
        edges = torch.tensor([
            [t[0], t[1]],
            [t[1], t[0]],
            [t[0], t[2]],
            [t[2], t[0]],
            [t[1], t[2]],
            [t[2], t[1]],
        ])
        return edges

    def __triangle2edge_features(self, edges, graph):
        points = graph[edges]

        # compute distances
        dist = torch.norm(points[..., 0] - points[..., -1],
                          p=2,
                          dim=-1,
                          keepdim=True)
        return dist

    def __rescale(self, x, w, h):

        x[..., 0] -= 100
        x[..., 1] -= 150

        x[..., 0] /= w
        x[..., 1] /= h
        return x

    def get_landmarks(self, l):

        # select last:

        landmarks = l[:, -1, :, :]  # - l[:, 0, :, :]
        # looks like taking the 9-8-7th image helps for certain emotions
        mouv1 = l[:, -1, :, :] - l[:, 0, :, :]
        mouv2 = l[:, -2, :, :] - l[:, 0, :, :]
        mouv3 = l[:, -3, :, :] - l[:, 0, :, :]

        mouv = torch.cat([mouv1, mouv2, mouv3], -1)

        # create weighted context
        w = torch.linspace(0, 3, 3).view(-1, 1, 1)
        ctx = l[:, -4:-1, :, :]
        w_ctx = ctx * w
        m_ctx = torch.mean(w_ctx, dim=1)

        width = torch.abs(landmarks[:, 0, 0] -
                          landmarks[:, 16, 0]).unsqueeze(-1)
        heigth = 2 * torch.abs(landmarks[:, 8, 1] -
                               landmarks[:, 29, 1]).unsqueeze(-1)

        # rescale

        landmarks = self.__rescale(landmarks, width, heigth)
        m_ctx = self.__rescale(landmarks, width, heigth)

        # select landmarks we want to keep (previously identitfied with random forest)
        idx = [
            50, 49, 59, 56, 57, 58, 61, 60, 64, 10, 34, 48, 55, 41, 36, 31, 13,
            46, 9, 0, 63, 14, 54, 6, 1, 38, 67, 47, 35, 45, 12, 44, 11, 40, 2,
            53, 52, 16, 62, 4, 15, 51, 66, 32, 65, 3, 37, 5, 42, 24, 22, 17,
            43, 18, 33, 8, 23, 26, 30, 25, 19, 21, 27, 7, 20, 39, 29, 28
        ]

        return landmarks[:, idx, :], mouv[:, idx, :], m_ctx[:, idx, :]

    def process(self):
        data = pickle.load(open(self.raw_paths[0], "rb"))

        # only keep the last image of each sequence
        landmarks = torch.tensor(data["landmarks"]).float()
        landmarks, mouv, ctx = self.get_landmarks(landmarks)

        labels = torch.tensor(data["labels"]).float()

        self.length = len(landmarks)

        for i in range(self.length):
            graph = landmarks[i]
            graph_mouv = mouv[i]
            graph_ctx = ctx[i]
            label = labels[i]

            x = torch.cat((graph, graph_mouv, graph_ctx), -1)

            # compute triangle mesh
            tri = Delaunay(graph)

            edges = torch.empty(0)
            # convert triangle to edges:
            for t in tri.simplices:
                edges = torch.cat((edges, self.__triangle2edges(t)), -2)

            edges = edges.long()
            edge_features = self.__triangle2edge_features(edges, graph)

            # Create data object
            data = Data(
                x=x,
                edge_index=edges.t().contiguous().long(),
                edge_attr=edge_features,
                y=label.long(),
            )

            torch.save(
                data,
                os.path.join(self.processed_dir, f'data_{self.mode}_{i}.pt'))

    def len(self):
        return self.length

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """

        data = torch.load(
            os.path.join(self.processed_dir, f'data_{self.mode}_{idx}.pt'))
        return data