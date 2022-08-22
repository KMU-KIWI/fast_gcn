import torch
from torch import Tensor


def sample_frames(x: Tensor, length: int):
    T, M, J, C = x.size()

    if T == length:
        return x

    if T < length:
        sampled_frames = torch.zeros(length, M, J, C).float()

        sampled_frames[:T] = x

        return sampled_frames
    else:
        chunks = x.tensor_split(length)

        sampled_frames = []
        for chunk in chunks:
            chunk_length = chunk.size(0)
            index = torch.randint(chunk_length, (1,))
            sampled_frames.append(chunk[index])

        return torch.cat(sampled_frames, dim=0)
