import zipfile
import os
from urllib.request import urlretrieve
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from truckms.api import FrameDatapoint, TargetDatapoint
import numpy as np


def get_progress_bar_hook():
    pbar = None
    downloaded = 0

    def show_progress(count, block_size, total_size):
        nonlocal pbar
        nonlocal downloaded
        if pbar is None:
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024)
            # pbar = ProgBar(total_size, stream=sys.stdout)

        downloaded += block_size
        pbar.update(block_size)
        if downloaded == total_size:
            pbar.finish()
            pbar = None
            downloaded = 0

    return show_progress

def download_data_if_not_exists(datalake_path):
    if not os.path.exists(datalake_path):
        os.mkdir(datalake_path)
    coco_val_2017_path = os.path.join(datalake_path, "coco_val_2017")
    if not os.path.exists(coco_val_2017_path):
        os.mkdir(coco_val_2017_path)
    if len(os.listdir(coco_val_2017_path)) != 4:
        print("downloading data coco_val_2017")

        zip_path = os.path.join(coco_val_2017_path, "val2017.zip")
        urlretrieve(r"http://images.cocodataset.org/zips/val2017.zip", zip_path, get_progress_bar_hook())
        z = zipfile.ZipFile(zip_path)
        z.extractall(coco_val_2017_path)

        zip_path = os.path.join(coco_val_2017_path, "annotations_trainval2017.zip")
        urlretrieve(r"http://images.cocodataset.org/annotations/annotations_trainval2017.zip", zip_path, get_progress_bar_hook())
        z = zipfile.ZipFile(zip_path)
        z.extractall(coco_val_2017_path)


def get_dataset(datalake_path):
    coco_val_2017_root_path = os.path.join(datalake_path, "coco_val_2017", "val2017")
    coco_val_2017_anno_path = os.path.join(datalake_path, "coco_val_2017", "annotations", "instances_val2017.json")
    dataset = CocoDetection(root=coco_val_2017_root_path, annFile=coco_val_2017_anno_path)
    return dataset


def gen_cocoitem2datapoints(dataset: CocoDetection):
    for i in range(len(dataset)):
        img, gt_target = dataset[i]
        fdp = FrameDatapoint(np.array(img), gt_target[0]['image_id'])
        tdp = TargetDatapoint(target={"boxes": np.array([t['bbox'] for t in gt_target]),
                                      "labels": np.array([t['category_id'] for t in gt_target])},
                              frame_id=gt_target[0]['image_id'])
        yield fdp, tdp
