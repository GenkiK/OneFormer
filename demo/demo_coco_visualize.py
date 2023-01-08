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

import cv2
import numpy as np
import tqdm
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
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
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/gkinoshita/dataset/kitti-object-detection/training/tmp/"),
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

    demo = VisualizationDemo(cfg)

    # TODO: CITYSCAPESの２パターン(念のため，ADE20k含む他のパターンも見てみてもいいかも）でとりあえずやる．並行してCOCO用も作れたらいい．くそ横長に対しては左右1/3ずつみて，マスク面積が小さい方を塗りつぶすっていうのでやる．
    # NMSはすでにやってあるっぽいけど，車が複数に分割される率高い．．そのへんはCOCOのほうがしっかりやってくれそうやけど，COCOのほうが後処理めんどいのと車の周辺部が汚い．
    # TODO: bboxは後で消すようにする
    # TODO: truck, carなどのmask面積のうち，数％内包されたpersonは消す＆その領域のpersonを塗りつぶす．(車の前の歩行者を消さないように注意)上３/2とかの条件も使えるかも
    # TODO: personの装飾品は全てのpersonに含まれるようにする．その領域も同一personで塗りつぶす．（handbagなど）
    # TODO: 明らかに不要なクラスは保存しないようにする．
    # TODO: 自転車でくそ横長が発生しているので対処する（面積の小さい左右半分を塗りつぶすので対処できそう）

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
        else:
            plt.figure(figsize=(16, 7), tight_layout=True)
            plt.imshow(visualized_output["instance_inference"].get_image())
            plt.show()
