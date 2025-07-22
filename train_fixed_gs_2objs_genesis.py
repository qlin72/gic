import os
import time
import scipy
import torch
import torchvision
import random
import subprocess
import json
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
from new_trajectory import load_pcd_file, read_estimation_result, gen_xyz_list, render_new, load_sample_and_save_pcd_with_velocity
from train_gs_fixed_pcd import train_gs_with_fixed_pcd_2_objs
from utils.image_utils import psnr
from utils.loss_utils import ssim
from gaussian_renderer import render
from pathlib import Path



def load_object_vel_and_color_from_json(json_path):
    """
    Load the initial velocity and surface color of obj1 and obj2 from a JSON file.

    Args:
        json_path (str): Path to the JSON configuration file.

    Returns:
        Tuple[list, list, list, list]:
            - velocity_obj1: List of 3 floats representing initial velocity of obj1
            - velocity_obj2: List of 3 floats representing initial velocity of obj2
            - color_obj1: List of 3 floats representing RGB color of obj1
            - color_obj2: List of 3 floats representing RGB color of obj2
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    velocity_obj1 = data['obj1']['initial_velocity']
    velocity_obj2 = data['obj2']['initial_velocity']
    color_obj1 = data['obj1']['surface_color']
    color_obj2 = data['obj2']['surface_color']

    return velocity_obj1, velocity_obj2, color_obj1, color_obj2



if __name__ == "__main__":
    start_time = time.time()

    parser = ArgumentParser(description="Prediction")
    parser.add_argument("--predict_frames", default=30, type=int)
    # parser.add_argument("--train_frames", type=int)#, default=14, type=int)
    parser.add_argument("--train_frames", default=14, type=int)
    # parser.add_argument("--gt_path", type=str)
    # parser.add_argument('-cid', '--config_id', type=int, default=0)
    
    # parser.add_argument('--source_path', type=str)
    # parser.add_argument('--config_path', type=str)
    # parser.add_argument('--model_path', type=str)
    
    parser.add_argument('--sim_ini_pcd1_path', type=str)
    parser.add_argument('--sim_ini_pcd2_path', type=str)
    parser.add_argument('--json_path', type=str)
    
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    
    args = parser.parse_args()
    
    # gs_args, phys_args = get_combined_args(parser)
    # print(phys_args)
    # setattr(phys_args, "config_id", gs_args.config_id)
    # print(phys_args)
    dataset = model.extract(args)
    opt = op
    pipe = pipeline
    # opt = op.extract(gs_args)
    # pipe = pipeline.extract(gs_args)
    # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)

    # model_path = Path(dataset.model_path)
    # obj_name = dataset.model_path.split('/')[-1]
    # (model_path/f'{obj_name}_img_render').mkdir(exist_ok=True)
    # (model_path/f'{obj_name}_img_gt').mkdir(exist_ok=True)

    # 0. Load trained pcd
    # vol = load_pcd_file(dataset.model_path, gs_args.iteration)
   
    
    v1, v2, c1, c2 = load_object_vel_and_color_from_json(args.json_path)
    
    print("c1",c1)
    print("c2",c2)
    
    vol,slipt_idx = load_sample_and_save_pcd_with_velocity(args.sim_ini_pcd1_path,args.sim_ini_pcd2_path,v1,v2,ratio=0.075)
    
    print("vol:",vol)
    
    
    # estimation_params = Namespace(**read_estimation_result(dataset, phys_args))
    # print(estimation_params.fps)
    # simulator = Simulator(estimation_params, vol)
    # d_xyz_list = gen_xyz_list(simulator, gs_args.predict_frames, diff=True, save_ply=False, path=dataset.model_path)
    # d_xyz_list = d_xyz_list[:20]
    # print(gs_args.test_iterations + list(range(10000, 40001, 1000)))
    scene = train_gs_with_fixed_pcd_2_objs(
        vol, 
        dataset, 
        opt, 
        pipe, 
        list(range(10000, 10001, 1000)), 
        list(range(10000, 10001, 10000)), 
        None,
        fps = 24,
        # estimation_params.fps, 
        force_train=True, 
        # grid_size=estimation_params.density_grid_size
        split_idx = slipt_idx, 
        color_obj1 = c1, 
        color_obj2 = c2 
    )

