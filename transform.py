import torch
from torch_geometric.transforms import BaseTransform
from skimage.segmentation import slic
from torch_geometric.data import Data
import torchvision.transforms.functional as F
from torch_geometric.nn import radius_graph
from torch_geometric.utils import scatter

class ToImageGraph(BaseTransform):
    def __init__(self, n_segments, radius):
        self.n_segments = n_segments
        self.radius = radius
    
    def forward(self, img):
        # pytorch_geometric implementation of slic
        img = F.to_tensor(img)
        img = img.permute(1, 2, 0)
        h, w, c = img.size()
        seg = slic(img.to(torch.double).numpy(),n_segments=self.n_segments, start_label=0)
        seg = torch.from_numpy(seg)
        x = scatter(img.view(h * w, c), seg.view(h * w), dim=0, reduce='mean')
        pos_y = torch.arange(h, dtype=torch.float)
        pos_y = pos_y.view(-1, 1).repeat(1, w).view(h * w)
        pos_x = torch.arange(w, dtype=torch.float)
        pos_x = pos_x.view(1, -1).repeat(h, 1).view(h * w)
        pos = torch.stack([pos_x, pos_y], dim=-1)
        pos = scatter(pos, seg.view(h * w), dim=0, reduce='mean')
        edge_index = radius_graph(x, r=self.radius, loop=False)
        data = Data(x=x, pos=pos, edge_index=edge_index)
        return data
    
    def __call__(self, img):
        return self.forward(img)