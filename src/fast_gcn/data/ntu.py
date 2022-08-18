import os
import os.path as osp

import zipfile

from dataclasses import dataclass

from multiprocessing import Pool

import torch

from torch_geometric.data import Dataset, Data, extract_zip

from typing import List, Union


class NTU60(Dataset):
    train_subjects = [
        int(c)
        for c in (
            "1,2,4,5,8,9,"
            "13,14,15,16,17,18,19,"
            "25,27,28,"
            "31,34,35,38,"
            "45,46,47,49,"
            "50,52,53,54,55,56,57,58,59,"
            "70,74,78,"
            "80,81,82,83,84,85,86,89,"
            "91,92,93,94,95,97,98,"
            "100,103"
        ).split(",")
    ]

    train_cameras = [2, 3]

    urls = ["https://rose1.ntu.edu.sg/dataset/actionRecognition/download/157"]

    def __init__(
        self,
        root,
        benchmark: str = "xsub",
        split: str = "train",
        joint_type: Union[List[str], str] = "3d",
        max_bodies: int = 2,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        debug=False,
    ):
        self.root = root
        self.benchmark = benchmark
        self.split = split

        self.joint_type = joint_type
        self.max_bodies = max_bodies

        self.debug = debug

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data_list = torch.load(self.processed_paths[0])

    @property
    def num_joints(self) -> int:
        return 25

    @property
    def num_features(self) -> int:
        if self.joint_type == "3d":
            return 3
        else:
            return 2

    @property
    def raw_file_names(self) -> List[str]:
        return ["nturgbd_skeletons_s001_to_s017.zip"]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "NTURGBD", "raw")

    @property
    def extract_dir(self) -> str:
        return osp.join(self.raw_dir, "nturgb+d_skeletons")

    @property
    def processed_dir(self) -> str:
        return osp.join(
            self.root,
            "NTURGBD",
            "processed",
            f"ntu60_{self.benchmark}_{self.joint_type}_{self.max_bodies}_{self.split}",
        )

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        session = None
        for url, file_name in zip(self.urls, self.raw_file_names):
            if not osp.exists(osp.join(self.raw_dir, file_name)):
                if session is None:
                    session = login_rose()

                download_rose_url(session, url, self.raw_dir)

    def extract(self):
        ntu60_file_path = osp.join(self.raw_dir, self.raw_file_names[0])

        if self.check_zip(ntu60_file_path, "nturgb+d_skeletons/"):
            extract_zip(ntu60_file_path, self.raw_dir)

        return

    def check_zip(self, zip_path, at=""):
        zipfile_path = zipfile.Path(zip_path, at)
        for path in zipfile_path.iterdir():
            if osp.exists(osp.join(self.extract_dir, path.name)) is False:
                return True
        return False

    def process(self):
        self.extract()

        self.edge_index = (
            torch.tensor(
                [
                    [1, 2],
                    [2, 21],
                    [3, 21],
                    [4, 3],
                    [5, 21],
                    [6, 5],
                    [7, 6],
                    [8, 7],
                    [9, 21],
                    [10, 9],
                    [11, 10],
                    [12, 11],
                    [13, 1],
                    [14, 13],
                    [15, 14],
                    [16, 15],
                    [17, 1],
                    [18, 17],
                    [19, 18],
                    [20, 19],
                    [22, 23],
                    [23, 8],
                    [24, 25],
                    [25, 12],
                ],
                dtype=torch.long,
            )
            .t()
            .contiguous()
        )

        file_list = []
        for filename in os.listdir(self.extract_dir):
            if filename.split(".")[-1] != "skeleton":
                continue

            if self.benchmark == "xsub":
                eval_id = int(filename[9:12])
                train_group = self.train_subjects
            if self.benchmark == "xview":
                eval_id = int(filename[5:8])
                train_group = self.train_cameras

            if eval_id in train_group:
                curr_split = "train"
            else:
                curr_split = "val"

            if curr_split == self.split:
                file_list.append(osp.join(self.extract_dir, filename))

        if self.debug:
            data_list_with_nones = map(self.process_path, file_list)
        else:
            with Pool(os.cpu_count()) as p:
                data_list_with_nones = p.map(self.process_path, file_list)

        data_list = []
        for data in data_list_with_nones:
            if data is not None:
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(data_list, self.processed_paths[0])

    def process_path(self, path):
        with open(path, "r") as f:
            lines = f.read().splitlines()

        raw_skeleton_sequence = SkeletonSequence(lines)

        filename = osp.basename(path)
        label = int(filename[17:20]) - 1

        raw_frames = []
        for frame in raw_skeleton_sequence.frames:
            if frame.num_skeletons == 0:
                continue

            skeletons = {}
            for skeleton in frame.skeletons:
                joints = []
                for joint in skeleton.joints:
                    joint_feature = []
                    if "3d" in self.joint_type:
                        joint_feature.extend(joint.to_3d())
                    if "color_2d" in self.joint_type:
                        joint_feature.extend(joint.to_color_2d())
                    if "depth_2d" in self.joint_type:
                        joint_feature.extend(joint.to_depth_2d())

                    joints.append(torch.tensor(joint_feature, dtype=torch.float16))

                skeletons[skeleton.body_id] = torch.stack(joints, dim=0)

            raw_frames.append(skeletons)

        body_ids = self.extract_body_ids(raw_frames)
        num_bodies = len(body_ids)

        if num_bodies == 0:
            return None

        if num_bodies <= self.max_bodies:
            frames = raw_frames
            x = self.pad_frames(raw_frames)

        if num_bodies > self.max_bodies:
            raw_bodies = self.group_by_bodies(raw_frames)
            num_bodies = len(raw_bodies.keys())
            if num_bodies == 0:
                return None

            # denoise based on frame length
            bodies = self.filter_by_length(raw_bodies, 10)
            num_bodies = len(bodies.keys())
            if num_bodies == 0:
                return None

            # denoise based on spread
            bodies = self.filter_by_spread(bodies, 0.8)
            num_bodies = len(bodies.keys())
            if num_bodies == 0:
                return None

            # add motion data
            frames = self.filter_by_motion(bodies)

            body_ids = self.extract_body_ids(frames)
            num_bodies = len(body_ids)
            if num_bodies == 0:
                return None

            x = self.pad_frames(frames)

        assert x.size(1) == self.max_bodies, breakpoint()

        return Data(x=x, edge_index=self.edge_index, y=label)

    def group_by_bodies(self, raw_frames):
        body_ids = self.extract_body_ids(raw_frames)

        bodies = {body_id: {"frames": [], "indices": []} for body_id in body_ids}
        for body_id in body_ids:
            frames = []
            indices = []
            for i, raw_frame in enumerate(raw_frames):
                if body_id in raw_frame.keys():
                    frames.append(raw_frame[body_id])
                    indices.append(i)

            bodies[body_id] = {
                "frames": torch.stack(frames, dim=0),
                "indices": torch.tensor(indices),
            }

        return bodies

    def ungroup_to_frames(self, bodies):
        max_length = 0
        for body_id, body in bodies.items():
            max_length = max(max_length, max(body["indices"]))

        ungrouped_frames = []
        for i in range(max_length):
            frame = {}
            for body_id, body in bodies.items():
                frames = body["frames"]
                indices = body["indices"]

                if i in indices:
                    frame[body_id] = frames[indices == i]

            ungrouped_frames.append(frame)

        return ungrouped_frames

    def extract_body_ids(self, frames):
        body_ids = set()
        for frame in frames:
            for body_id in frame.keys():
                body_ids.add(body_id)

        return body_ids

    def pad_frames(self, raw_frames):
        body_ids = self.extract_body_ids(raw_frames)

        frames = []
        for frame in raw_frames:
            skeletons = torch.zeros(
                self.max_bodies, self.num_joints, self.num_features
            ).float()

            for i, body_id in enumerate(body_ids):
                if body_id in frame.keys():
                    skeletons[i] = frame[body_id]

            frames.append(skeletons)

        frames = torch.stack(frames, dim=0)
        return frames

    def filter_by_length(self, bodies, threshold):
        if len(bodies.keys()) <= self.max_bodies:
            return bodies

        body_ids = list(bodies.keys())
        for body_id in body_ids:
            length = bodies[body_id]["indices"].size(0)

            if length <= threshold:
                del bodies[body_id]

        return bodies

    def filter_by_spread(self, bodies, threshold):
        if len(bodies.keys()) <= self.max_bodies:
            return bodies

        body_ids = list(bodies.keys())
        for body_id in body_ids:
            frames = bodies[body_id]["frames"]

            max_xy = frames[:, :, 0:2].max(dim=-1).values
            min_xy = frames[:, :, 0:2].min(dim=-1).values

            x = max_xy[:, 0] - min_xy[:, 0]
            y = max_xy[:, 1] - min_xy[:, 1]

            mask = x <= threshold * y
            ratio = mask.float().mean().cpu().item()

            if ratio < threshold:
                del bodies[body_id]

        return bodies

    def filter_by_motion(self, bodies):
        frames = self.ungroup_to_frames(bodies)

        if len(bodies.keys()) <= self.max_bodies:
            return frames

        motion = {}
        for body_id in bodies.keys():
            motion[body_id] = torch.sum(torch.var(bodies[body_id]["frames"], dim=0))

        motion = torch.stack(list(motion.values()), dim=0)
        sorted_motion = motion.sort(descending=True)

        body_ids = list(bodies.keys())
        indices = sorted_motion.indices[: self.max_bodies]
        body_ids = [body_ids[i] for i in indices]

        for i, frame in enumerate(frames):
            frame_body_ids = list(frame.keys())
            for frame_body_id in frame_body_ids:
                if frame_body_id not in body_ids:
                    del frames[i][frame_body_id]

        return frames

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class NTU120(NTU60):
    urls = [
        "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/157",
        "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/158",
    ]

    @property
    def processed_dir(self) -> str:
        return osp.join(
            self.root, "NTURGBD", "processed", f"ntu120_{self.benchmark}_{self.split}"
        )

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "nturgbd_skeletons_s001_to_s017.zip",
            "nturgbd_skeletons_s018_to_s032.zip",
        ]

    def extract(self):
        super().extract()

        ntu120_file_path = osp.join(self.raw_dir, self.raw_file_names[1])
        extract_zip(ntu120_file_path, self.extract_dir)

        if self.check_zip(ntu120_file_path):
            extract_zip(ntu120_file_path, self.raw_dir)

    def process(self):
        super().process()


def login_rose():
    import getpass

    import requests
    from bs4 import BeautifulSoup

    s = requests.Session()
    r = s.get("https://rose1.ntu.edu.sg/login/")
    r.raise_for_status()

    soup = BeautifulSoup(r.content, features="html.parser")
    csrftoken = soup.find("input", dict(name="csrfmiddlewaretoken"))["value"]

    print(
        "Type your username and password for http://rose1.ntu.edu.sg to download dataset"
    )
    username = input("Username: ")
    password = getpass.getpass()

    r = s.post(
        "https://rose1.ntu.edu.sg/login/",
        data={
            "csrfmiddlewaretoken": csrftoken,
            "username": username,
            "password": password,
        },
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
            ),
            "Referer": "https://rose1.ntu.edu.sg/login/",
        },
    )
    r.raise_for_status()

    return s


def download_rose_url(s, url, dst):
    import cgi

    from tqdm import tqdm

    with s.get(url, stream=True) as r:
        r.raise_for_status()

        headers = r.headers
        total = int(headers.get("content-length", 0))

        string = headers.get("Content-Disposition", None)

        from_url = url.split("/")[-1]
        if string is None:
            file_name = from_url
        else:
            value, params = cgi.parse_header(string)
            file_name = params.get("filename", from_url)

        path = osp.join(dst, file_name)

        if osp.exists(path):
            print(f"{file_name} already downloaded")
            return file_name

        with tqdm.wrapattr(
            open(path, "wb"),
            "write",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            total=total,
        ) as fout:
            for chunk in r.iter_content(chunk_size=1024):
                fout.write(chunk)

        return file_name


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
