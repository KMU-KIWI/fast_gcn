import os.path as osp
import zipfile

import torch
from torch import Tensor

from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class Joint:
    x: float
    y: float
    z: float
    depth_x: float
    depth_y: float
    color_x: float
    color_y: float
    orientation_w: float
    orientation_x: float
    orientation_y: float
    orientation_z: float
    tracking_state: int

    def __init__(self, line):
        (
            x,
            y,
            z,
            depth_x,
            depth_y,
            color_x,
            color_y,
            orientation_w,
            orientation_x,
            orientation_y,
            orientation_z,
            tracking_state,
        ) = line.split()

        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.depth_x = float(depth_x)
        self.depth_y = float(depth_y)
        self.color_x = float(color_x)
        self.color_y = float(color_y)
        self.orientation_w = float(orientation_w)
        self.orientation_x = float(orientation_x)
        self.orientation_y = float(orientation_y)
        self.orientation_z = float(orientation_z)
        self.tracking_state = int(tracking_state)

    def to_3d(self):
        return [self.x, self.y, self.z]

    def to_color_2d(self):
        return [self.color_x, self.color_y]

    def to_depth_2d(self):
        return [self.depth_x, self.depth_y]


@dataclass
class Skeleton:
    body_id: str
    clipped_edges: int
    left_hand_confidence: int
    left_hand_state: int
    right_hand_confidence: int
    right_hand_state: int
    restricted: int
    lean_x: float
    lean_y: float
    tracking_state: int
    num_joints: int
    joints: List[Joint]

    def __init__(self, lines):
        (
            body_id,
            clipped_edges,
            left_hand_confidence,
            left_hand_state,
            right_hand_confidence,
            right_hand_state,
            restricted,
            lean_x,
            lean_y,
            tracking_state,
        ) = lines.pop(0).split()

        self.body_id = body_id
        self.clipped_edges = int(clipped_edges)
        self.left_hand_confidence = int(left_hand_confidence)
        self.left_hand_state = int(left_hand_state)
        self.right_hand_confidence = int(right_hand_confidence)
        self.right_hand_state = int(right_hand_state)
        self.restricted = int(restricted)
        self.lean_x = float(lean_x)
        self.lean_y = float(lean_y)
        self.tracking_state = int(tracking_state)

        self.num_joints = int(lines.pop(0))

        self.joints = [Joint(lines.pop(0)) for _ in range(self.num_joints)]


@dataclass
class Frame:
    num_skeletons: int
    skeletons: List[Skeleton]

    def __init__(self, lines):
        self.num_skeletons = int(lines.pop(0))
        self.skeletons = [Skeleton(lines) for _ in range(self.num_skeletons)]


@dataclass
class SkeletonSequence:
    num_frames: int
    frames: List[Frame]

    def __init__(self, lines):
        self.num_frames = int(lines.pop(0))
        self.frames = [Frame(lines) for _ in range(self.num_frames)]


@dataclass(unsafe_hash=True)
class Bodies:
    frames: Dict[str, Tensor] = field(default_factory=dict)
    indices: Dict[str, Tensor] = field(default_factory=dict)
    body_ids: List[str] = field(default_factory=list)
    num_bodies: int = 0

    @staticmethod
    def from_list(raw_frames=[]):
        if len(raw_frames) == 0:
            return Bodies()
        else:
            body_ids = set()
            for frame in raw_frames:
                for body_id in frame.keys():
                    body_ids.add(body_id)

            body_ids = list(body_ids)
            num_bodies = len(body_ids)

            frames = dict()
            indices = dict()

            for body_id in body_ids:
                frame_list = []
                index_list = []
                for i, raw_frame in enumerate(raw_frames):
                    if body_id in raw_frame.keys():
                        frame_list.append(raw_frame[body_id])
                        index_list.append(i)

                frames[body_id] = torch.stack(frame_list, dim=0)
                indices[body_id] = torch.tensor(index_list)

            return Bodies(frames, indices, body_ids, num_bodies)

    def remove_body(self, body_id):
        assert body_id in self.body_ids

        frames = self.frames[body_id]
        indices = self.indices[body_id]

        del self.frames[body_id]
        del self.indices[body_id]

        self.body_ids.remove(body_id)
        self.num_bodies -= 1

        return frames, indices

    def to_list(self):
        max_length = self.max_length()

        frames = []
        for i in range(max_length):
            frame = {}
            for body_id in self.body_ids:
                indices = self.indices[body_id]

                if i in indices:
                    frame[body_id] = self.frames[body_id][indices == i]

            frames.append(frame)

        return frames

    def max_length(self):
        max_length = 0
        for body_id in self.body_ids:
            max_length = max(max_length, max(self.indices[body_id]) + 1)

        return max_length


def zip_extracted(search_root: str, zip_path: str, at: str = ""):
    zipfile_path = zipfile.Path(zip_path, at)
    for path in zipfile_path.iterdir():
        if osp.exists(osp.join(search_root, path.name)) is False:
            return False
    return True
