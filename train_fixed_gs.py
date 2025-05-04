import os
import time
import scipy
import torch
import torchvision
import random
import subprocess
import numpy as np
import taichi as ti
import open3d as o3d
import trimesh as tm
from tqdm import tqdm
from simulator import Simulator
from utils.general_utils import safe_state
from pytorch3d.loss import chamfer_distance
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from new_trajectory import load_pcd_file, read_estimation_result, gen_xyz_list, render_new
from train_gs_fixed_pcd import train_gs_with_fixed_pcd
from utils.image_utils import psnr
from utils.loss_utils import ssim
from gaussian_renderer import render
from pathlib import Path

if __name__ == "__main__":
    start_time = time.time()

    parser = ArgumentParser(description="Prediction")
    parser.add_argument("--predict_frames", default=30, type=int)
    # parser.add_argument("--train_frames", type=int)#, default=14, type=int)
    parser.add_argument("--train_frames", default=14, type=int)
    parser.add_argument("--gt_path", type=str)
    parser.add_argument('-cid', '--config_id', type=int, default=0)
    model = ModelParams(parser)#, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    gs_args, phys_args = get_combined_args(parser)
    print(phys_args)
    setattr(phys_args, "config_id", gs_args.config_id)
    # print(phys_args)
    safe_state(gs_args.quiet)
    dataset = model.extract(gs_args)
    opt = op.extract(gs_args)
    pipe = pipeline.extract(gs_args)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)

    model_path = Path(dataset.model_path)
    obj_name = dataset.model_path.split('/')[-1]
    (model_path/f'{obj_name}_img_render').mkdir(exist_ok=True)
    (model_path/f'{obj_name}_img_gt').mkdir(exist_ok=True)

    # 0. Load trained pcd
    vol = load_pcd_file(dataset.model_path, gs_args.iteration)

    
    # estimation_params = Namespace(**read_estimation_result(dataset, phys_args))
    # print(estimation_params.fps)
    # simulator = Simulator(estimation_params, vol)
    # d_xyz_list = gen_xyz_list(simulator, gs_args.predict_frames, diff=True, save_ply=False, path=dataset.model_path)
    # d_xyz_list = d_xyz_list[:20]
    print(gs_args.test_iterations + list(range(10000, 40001, 1000)))
    scene = train_gs_with_fixed_pcd(
        vol, 
        dataset, 
        opt, 
        pipe, 
        gs_args.test_iterations + list(range(10000, 40001, 1000)), 
        gs_args.save_iterations, 
        None,
        fps = 24,
        # estimation_params.fps, 
        force_train=True, 
        # grid_size=estimation_params.density_grid_size
    )

