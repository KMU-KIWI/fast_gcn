from torch_geometric.transforms import BaseTransform

from fast_gcn.data.samplers import sample_frames


class SampleFrames(BaseTransform):
    def __init__(self, length):
        self.length = length

    def __call__(self, data):
        data.x = sample_frames(data.x, self.length)
        return data
