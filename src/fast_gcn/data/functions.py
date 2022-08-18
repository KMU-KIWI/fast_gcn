import torch
from torch import Tensor


def sample_frames(x: Tensor, length: int):
    T, M, J, C = x.size()
    if T < length:
        sampled_frames = torch.zeros(length, M, J, C).float()

        sampled_frames[:T] = x

        return sampled_frames
    else:
        chunks = x.split(T // length)

        sampled_frames = []
        for chunk in chunks:
            chunk_frames = chunk.size(0)
            index = torch.randint(chunk_frames)
            sampled_frames.append(chunk_frames[index])

        return torch.stack(sampled_frames, dim=0)
