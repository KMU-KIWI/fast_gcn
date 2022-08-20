import os
import os.path as osp

from typing import List, Union

import zipfile

from multiprocessing import Pool

import torch

from torch_geometric.data import Dataset, Data, extract_zip

from ._dataclasses import SkeletonSequence, Bodies
from .functions import filter_by_length, filter_by_spread, filter_by_motion, pad_frames


edge_index = [
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
]


class BatchData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "x":
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


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
        length_threshold: int = 11,
        spread_threshold: float = 0.8,
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

        self.length_threshold = length_threshold
        self.spread_threshold = spread_threshold

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

    def check_zip(self, zip_path, at=""):
        zipfile_path = zipfile.Path(zip_path, at)
        for path in zipfile_path.iterdir():
            if osp.exists(osp.join(self.extract_dir, path.name)) is False:
                return True
        return False

    def process(self):
        self.extract()

        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

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

                    joints.append(torch.tensor(joint_feature, dtype=torch.float32))

                skeletons[skeleton.body_id] = torch.stack(joints, dim=0)

            raw_frames.append(skeletons)

        bodies = Bodies.from_list(raw_frames)
        bodies = self.filter_bodies(bodies)
        if bodies is None:
            return None

        x = pad_frames(
            bodies.to_list(), self.max_bodies, self.num_joints, self.num_features
        )

        return Data(x=x, edge_index=self.edge_index, y=label)

    def filter_bodies(self, bodies: Bodies):
        if bodies.num_bodies == 0:
            return None

        if bodies.num_bodies <= self.max_bodies:
            return bodies

        if bodies.num_bodies > self.max_bodies:
            bodies = filter_by_length(bodies, self.length_threshold)

            if bodies.num_bodies == 0:
                return None

            bodies = filter_by_spread(bodies, self.spread_threshold)

            if bodies.num_bodies == 0:
                return None

            if bodies.num_bodies > self.max_bodies:
                bodies = filter_by_motion(bodies, self.max_bodies)

            assert bodies.num_bodies <= self.max_bodies, print(bodies, self.max_bodies)

            return bodies

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

        if self.check_zip(ntu120_file_path):
            extract_zip(ntu120_file_path, self.extract_dir)

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
