# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------
import argparse
import multiprocessing as mp
import os
import random
import shutil

# fmt: off
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time

import numpy as np
import tqdm
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import Boxes, Instances
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo

from oneformer import add_common_config, add_convnext_config, add_dinat_config, add_oneformer_config, add_swin_config

# constants
WINDOW_NAME = "OneFormer Demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="oneformer demo for builtin configs")
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("../configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml"),
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--task", help="Task type", default="instance")
    parser.add_argument(
        "--input-dir",
        # help="A list of space separated input images; " "or a single glob pattern such as 'directory/*.jpg'",
        type=Path,
        default=Path("/home/gkinoshita/dataset/kitti-object-detection/training/image_2/"),
        # default=Path("/home/gkinoshita/dataset/kitti-object-detection/training/image_hard/"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/gkinoshita/dataset/kitti-object-detection/training/tmp4/"),
        # default=Path("/home/gkinoshita/dataset/kitti-object-detection/prediction/"),
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    demo = VisualizationDemo(cfg, parallel=False)

    # TODO: bboxは後で消すようにする
    # truck, carなどのmask面積のうち，数％内包されたpersonは消す＆その領域のpersonを塗りつぶす．(車の前の歩行者を消さないように注意)上３/2とかの条件も使えるかも
    # personの装飾品は全てのpersonに含まれるようにする．その領域も同一personで塗りつぶす．（handbagなど）
    # 明らかに不要なクラスは保存しないようにする．
    # 自転車でくそ横長が発生しているので対処する（面積の小さい左右半分を塗りつぶすので対処できそう）
    # HACK: 問題の残っている画像：3432.png 3408.png, 709.png, 3564.png, 3576.png, 6666.png, 7288.png, 535.png, 3210.png, 3408.png, 3617.png

    if args.output_dir:
        args.output_dir = args.output_dir / args.config_file.stem
        # HACK: output_dirの中身を全て一旦消すようにしていることに注意
        if args.output_dir.exists() and "tmp" in str(args.output_dir):
            shutil.rmtree(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm.tqdm(args.input_dir.glob("*.png")):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img, args.task)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )
        if args.output_dir:
            save_filename = args.output_dir / path.stem
            visualized_output["instance_inference"].save(save_filename)
            # out_filepath = args.output_dir / path.stem
            # instances: Instances = predictions["instances"]
            # boxes: Boxes = instances.get("pred_boxes")
            # labels: torch.Tensor = instances.get("pred_classes")
            # np.savez(out_filepath, boxes=boxes.tensor.cpu().numpy(), labels=labels.cpu().numpy())
        else:
            plt.figure(figsize=(16, 7), tight_layout=True)
            plt.imshow(visualized_output["instance_inference"].get_image())
            plt.show()

"""
x: 削除すべき
-: supercategoryがaccessoryに分類されているもの

{
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
x    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
x    "boat": 8,
x    "traffic light": 9,
x    "fire hydrant": 10,
x    "stop sign": 11,
x    "parking meter": 12,
x    "bench": 13,
x    "bird": 14,
x    "cat": 15,
x    "dog": 16,
x    "horse": 17,
x    "sheep": 18,
x    "cow": 19,
x    "elephant": 20,
x    "bear": 21,
x    "zebra": 22,
x    "giraffe": 23,
-    "backpack": 24,
-    "umbrella": 25,
-    "handbag": 26,
-    "tie": 27,
-    "suitcase": 28,
x    "frisbee": 29,
x    "skis": 30,
x    "snowboard": 31,
x    "sports ball": 32,
x    "kite": 33,
x    "baseball bat": 34,
x    "baseball glove": 35,
x    "skateboard": 36,
x    "surfboard": 37,
x    "tennis racket": 38,
x    "bottle": 39,
x    "wine glass": 40,
x    "cup": 41,
x    "fork": 42,
x    "knife": 43,
x    "spoon": 44,
x    "bowl": 45,
x    "banana": 46,
x    "apple": 47,
x    "sandwich": 48,
x    "orange": 49,
x    "broccoli": 50,
x    "carrot": 51,
x    "hot dog": 52,
x    "pizza": 53,
x    "donut": 54,
x    "cake": 55,
x    "chair": 56,
x    "couch": 57,
x    "potted plant": 58,
x    "bed": 59,
x    "dining table": 60,
x    "toilet": 61,
x    "tv": 62,
x    "laptop": 63,
x    "mouse": 64,
x    "remote": 65,
x    "keyboard": 66,
x    "cell phone": 67,
x    "microwave": 68,
x    "oven": 69,
x    "toaster": 70,
x    "sink": 71,
x    "refrigerator": 72,
x    "book": 73,
x    "clock": 74,
x    "vase": 75,
x    "scissors": 76,
x    "teddy bear": 77,
x    "hair drier": 78,
x    "toothbrush": 79,
}
"""
