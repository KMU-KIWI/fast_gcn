import torch

from torch_geometric.transforms import BaseTransform

from fast_gcn.data.samplers import sample_frames


class SampleFrames(BaseTransform):
    def __init__(self, length, num_clips=1):
        self.length = length
        self.num_clips = num_clips

    def __call__(self, data):
        clips = []
        for _ in range(self.num_clips):
            clips.append(sample_frames(data.x, self.length))

        if self.num_clips > 1:
            data.x = torch.stack(clips, dim=0)
            data.y = torch.tensor([data.y]).unsqueeze(0).repeat(self.num_clips, 1)
        else:
            data.x = clips[0]

        return data
