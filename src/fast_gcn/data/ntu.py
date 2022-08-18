import os
import os.path as osp

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
    ):
        self.root = root
        self.benchmark = benchmark
        self.split = split
        self.joint_type = joint_type

        self.max_bodies = max_bodies

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data_list = torch.load(self.processed_paths[0])

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
            self.root, "NTURGBD", "processed", f"ntu60_{self.benchmark}_{self.split}"
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
        ntu60_file = self.raw_file_names[0]
        extract_zip(osp.join(self.raw_dir, ntu60_file), self.raw_dir)

    def process(self):
        if not osp.exists(self.extract_dir):
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

        with Pool(os.cpu_count()) as p:
            data_list_with_nones = p.map(self.process_one, file_list)

        data_list = []
        for data in data_list_with_nones:
            if data is not None:
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(data_list, self.processed_paths[0])

    def process_one(self, path):
        with open(path, "r") as f:
            lines = f.read().splitlines()

        raw_skeleton_sequence = SkeletonSequence(lines)

        filename = osp.basename(path)
        label = int(filename[17:20]) - 1

        body_ids = set()
        frames = []
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

                body_id = skeleton.body_id
                body_ids.add(body_id)
                skeletons[body_id] = torch.stack(joints, dim=0)

            frames.append(skeletons)

        num_bodies = len(body_ids)

        if num_bodies == 0:
            return None

        if num_bodies <= self.max_bodies:
            skeleton_sequence = self.pad_bodies(frames, body_ids)

        if num_bodies > self.max_bodies:
            # denoise based on frame length
            frames = self.filter_short_bodies(frames, body_ids, 10)

            # denoise based on spread
            frames = self.filter_spread_bodies(frames, body_ids)

            motions = []
            for body_sequence in skeleton_sequence:
                motion = torch.sum(torch.var(body_sequence, dim=0))
                motions.append(motion)

        return Data(x=skeleton_sequence, edge_index=self.edge_index, y=label)

    def pad_bodies(self, frames, body_ids):
        skeleton_sequence = []
        for frame in frames:
            skeletons = []
            for body_id in body_ids:
                if body_id in frame.keys():
                    skeletons.append(frame[body_id])
                else:
                    k = list(frame.keys())[0]
                    skeletons.append(torch.zeros_like(frame[k]))

            while len(skeletons) < self.max_bodies:
                skeletons.append(torch.zeros_like(skeletons[0]))

            skeletons = torch.stack(skeletons, dim=0)
            skeleton_sequence.append(skeletons)

        skeleton_sequence = torch.stack(skeleton_sequence, dim=0)
        return skeleton_sequence

    def filter_short_bodies(self, frames, body_ids, threshold):
        for body_id in body_ids:
            length = 0
            for frame in frames:
                if body_id in frame.keys():
                    length += 1
            if length <= threshold:
                body_ids.remove(body_id)
                for i in range(len(frames)):
                    del frames[i][body_id]
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
        ntu120_file = self.raw_file_names[1]
        extract_zip(ntu120_file, self.extract_dir)

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
