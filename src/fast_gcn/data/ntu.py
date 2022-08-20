import os
import os.path as osp

from typing import List, Union

from multiprocessing import Pool

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data, extract_zip, makedirs

from .utils import zip_extracted
from .utils import SkeletonSequence, Bodies

from .filters import filter_by_length, filter_by_spread, filter_by_motion, pad_frames


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


class NTU(Dataset):
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

    def __init__(
        self,
        root,
        benchmark: str = "xsub",
        split: str = "train",
        joint_type: Union[List[str], str] = "3d",
        max_bodies: int = 2,
        length_threshold: int = 11,
        spread_threshold: float = 0.8,
        download: bool = True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__()

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        self.root = root
        self.benchmark = benchmark
        self.split = split

        self.joint_type = joint_type
        self.max_bodies = max_bodies

        self.length_threshold = length_threshold
        self.spread_threshold = spread_threshold

        if download:
            self.download()
            self.extract()

        self.raw_file_paths = []
        for name in os.listdir(self.raw_dir):
            if name.split(".")[-1] != "skeleton":
                continue

            if self.benchmark == "xsub":
                eval_id = int(name[9:12])
                train_group = self.train_subjects
            if self.benchmark == "xview":
                eval_id = int(name[5:8])
                train_group = self.train_cameras

            if eval_id in train_group:
                curr_split = "train"
            else:
                curr_split = "val"

            if curr_split == self.split:
                self.raw_file_paths.append(osp.join(self.raw_dir, name))

        if not osp.exists(self.processed_file_paths[0]):
            self.data_list = self.process(root)
            if not osp.exists(self.processed_dir):
                makedirs(self.processed_dir)
            torch.save(self.data_list, self.processed_file_paths[0])
        else:

            self.data_list = torch.load(self.processed_file_paths[0])

    @property
    def urls(self) -> List[str]:
        raise NotImplementedError

    def download(self):
        session = None
        for url, file_name in zip(self.urls, self.raw_file_names):
            if not osp.exists(osp.join(self.root, file_name)):
                if session is None:
                    session = login_rose()

                download_rose_url(session, url, self.root)

    @property
    def base_dir(self) -> str:
        return osp.join(self.root, "NTU")

    @property
    def processed_dir(self) -> str:
        raise NotImplementedError

    @property
    def processed_file_paths(self) -> List[str]:
        return [osp.join(self.processed_dir, "data.pt")]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.base_dir, "skeletons")

    def process(self, root):
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # with Pool(os.cpu_count()) as p:
        raw_data_list = map(self.process_path, self.raw_file_paths)

        data_list = []
        for data in raw_data_list:
            if not data:
                continue

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        return data_list

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
        if not bodies:
            return False

        T, J, C = bodies.frames[bodies.body_ids[0]].size()
        x = pad_frames(bodies.to_list(), self.max_bodies, J, C)

        return Data(x=x, edge_index=self.edge_index, y=label)

    def filter_bodies(self, bodies: Bodies):
        if bodies.num_bodies == 0:
            return False

        if bodies.num_bodies <= self.max_bodies:
            return bodies

        if bodies.num_bodies > self.max_bodies:
            bodies = filter_by_length(bodies, self.length_threshold)

            if bodies.num_bodies == 0:
                return False

            bodies = filter_by_spread(bodies, self.spread_threshold)

            if bodies.num_bodies == 0:
                return False

            if bodies.num_bodies > self.max_bodies:
                bodies = filter_by_motion(bodies, self.max_bodies)

            assert bodies.num_bodies <= self.max_bodies, print(bodies, self.max_bodies)

            return bodies

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        if self.transform is not None:
            data = self.transform(data)

        return data.x, data.edge_index, data.y


class NTU60(NTU):
    @property
    def urls(self) -> List[str]:
        return ["https://rose1.ntu.edu.sg/dataset/actionRecognition/download/157"]

    @property
    def raw_file_names(self) -> List[str]:
        return ["nturgbd_skeletons_s001_to_s017.zip"]

    @property
    def processed_dir(self) -> str:
        return osp.join(
            self.base_dir,
            f"ntu120_{self.benchmark}_{self.split}_{self.joint_type}_{self.max_bodies}bodies",
        )

    def extract(self):
        ntu60_file_path = osp.join(self.root, self.raw_file_names[0])

        internal_dir = "nturgb+d_skeletons/"
        if not zip_extracted(self.raw_dir, ntu60_file_path, internal_dir):
            extract_zip(ntu60_file_path, self.base_dir)
            os.rename(osp.join(self.base_dir, internal_dir), self.raw_dir)


class NTU120(NTU):
    @property
    def urls(self) -> List[str]:
        return [
            "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/157",
            "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/158",
        ]

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "nturgbd_skeletons_s001_to_s017.zip",
            "nturgbd_skeletons_s018_to_s032.zip",
        ]

    @property
    def processed_dir(self) -> str:
        return osp.join(
            self.base_dir,
            f"ntu120_{self.benchmark}_{self.split}_{self.joint_type}_{self.max_bodies}bodies",
        )

    def extract(self):
        ntu60_file_path = osp.join(self.root, self.raw_file_names[0])

        internal_dir = "nturgb+d_skeletons/"
        if not zip_extracted(self.raw_dir, ntu60_file_path, internal_dir):
            extract_zip(ntu60_file_path, self.base_dir)
            os.rename(osp.join(self.base_dir, internal_dir), self.raw_dir)

        ntu120_file_path = osp.join(self.root, self.raw_file_names[1])

        if not zip_extracted(self.raw_dir, ntu120_file_path):
            extract_zip(ntu120_file_path, self.raw_dir)


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
