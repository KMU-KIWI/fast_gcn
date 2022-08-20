from typing import List, Dict

import torch
from torch import Tensor

from ._dataclasses import Bodies


def filter_by_length(bodies: Bodies, threshold: int):
    body_ids = list(bodies.body_ids)
    for body_id in body_ids:
        length = bodies.indices[body_id].size(0)

        if length <= threshold:
            bodies.remove_body(body_id)

    return bodies


def filter_by_spread(bodies: Bodies, threshold: float):
    body_ids = list(bodies.body_ids)
    for body_id in body_ids:
        frames = bodies.frames[body_id]
        max_xy = frames[:, :, 0:2].max(dim=-1).values
        min_xy = frames[:, :, 0:2].min(dim=-1).values

        x = max_xy[:, 0] - min_xy[:, 0]
        y = max_xy[:, 1] - min_xy[:, 1]

        mask = x <= threshold * y
        ratio = mask.float().mean().cpu().item()

        if ratio < threshold:
            bodies.remove_body(body_id)

    return bodies


def filter_by_motion(bodies: Bodies, max_bodies: int):
    motion = {}
    for body_id in bodies.body_ids:
        motion[body_id] = torch.sum(torch.var(bodies.frames[body_id], dim=0))

    motion = torch.stack(list(motion.values()), dim=0)
    sorted_motion = motion.sort(descending=True)

    indices = sorted_motion.indices[:max_bodies]
    filtered_ids = [bodies.body_ids[i] for i in indices]

    body_ids = list(bodies.body_ids)
    for body_id in body_ids:
        if body_id not in filtered_ids:
            bodies.remove_body(body_id)

    return bodies


def pad_frames(
    raw_frames: List[Dict[str, Tensor]],
    max_bodies: int,
    num_joints: int,
    num_features: int,
):
    bodies = Bodies.from_list(raw_frames)

    length = bodies.max_length()
    frames = torch.zeros(length, max_bodies, num_joints, num_features).float()

    for body_id in bodies.body_ids:
        frames[bodies.indices[body_id]] = bodies.frames[body_id]

    return frames


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
            chunk_length = chunk.size(0)
            index = torch.randint(chunk_length, (1,))
            sampled_frames.append(chunk[index])

        return torch.cat(sampled_frames, dim=0)
