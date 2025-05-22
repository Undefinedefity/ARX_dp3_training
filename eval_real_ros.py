
"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -f <frequency> --s <steps_per_inference>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

import sys

sys.path.append("/home/dc/mambaforge/envs/robodiff/lib/python3.9/site-packages")
sys.path.append(
    "/home/dc/Desktop/dp_ycw/follow_control/follow1/src/arm_control/scripts/KUKA-Controller"
)
import time
import math
import copy
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st



from common.precise_sleep import precise_wait
from real_world.real_inference_util import get_real_obs_resolution, get_real_obs_dict

from common.pytorch_util import dict_apply
from workspace.base_workspace import BaseWorkspace
from policy.base_image_policy import BaseImagePolicy
from common.cv2_util import get_image_transform

from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from multiprocessing.managers import SharedMemoryManager
import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from arm_control.msg import JointInformation
from arm_control.msg import JointControl
from arm_control.msg import PosCmd
from sensor_msgs.msg import Image
from threading import Lock
from cv_bridge import CvBridge,CvBridgeError

CAMERA_FX = 337.21  
CAMERA_FY = 432.97  
CAMERA_CX = 320.0  
CAMERA_CY = 240.0  
DEPTH_IMG_HEIGHT = 480 
DEPTH_IMG_WIDTH = 640  

OmegaConf.register_new_resolver("eval", eval, replace=True)
np.set_printoptions(suppress=True)

@click.command()
@click.option(
    "--input_path",
    "-i",
    required=True,
    help="Path to checkpoint",
)
@click.option(
    "--frequency",
    "-f",
    default=10,
    type=int,
    help="control frequency",
)
@click.option(
    "--steps_per_inference",
    "-s",
    default=8,
    type=int,
    help="Action horizon for inference.",
)

# @profile
def main(
    input_path,
    frequency,
    steps_per_inference,
):
    global obs_ring_buffer

    dt = 1 / frequency
    video_capture_fps = 30
    max_obs_buffer_size = 30

    # load checkpoint
    ckpt_path = input_path
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    cfg._target_ = "codebase.diffusion_policy." + cfg._target_
    cfg.policy._target_ = "codebase.diffusion_policy." + cfg.policy._target_
    cfg.ema._target_ = "codebase.diffusion_policy." + cfg.ema._target_

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # policy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device = torch.device("cuda:0")
    policy.eval().to(device)
    policy.reset()

    ## set inference params
    policy.num_inference_steps = 16  # DDIM inference iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    # buffer
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    examples = dict()
    point_cloud_shape_from_cfg = cfg.task.shape_meta.obs.point_cloud.shape
    examples["point_cloud"] = np.empty(shape=tuple(point_cloud_shape_from_cfg), dtype=np.float32)
    examples["agent_pos"] = np.empty(shape=(7,), dtype=np.float64)
    examples["timestamp"] = 0.0
    obs_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
        shm_manager=shm_manager,
        examples=examples,
        get_max_k=max_obs_buffer_size,
        get_time_budget=0.2,
        put_desired_frequency=video_capture_fps,
    )

    # ros config
    rospy.init_node("eval_real_ros")
    agent_pos = Subscriber("joint_information2", JointInformation)
    depth = Subscriber("mid_depth_camera", Image)
    control_robot2 = rospy.Publisher("test_right", JointControl, queue_size=10)
    ats = ApproximateTimeSynchronizer(
        [agent_pos, depth], queue_size=10, slop=0.1
    )
    ats.registerCallback(callback)
    rate = rospy.Rate(frequency)

    # data
    last_data = None
    right_control = JointControl()
    

    # start episode
    
    start_delay = 1.0
    eval_t_start = time.time() + start_delay
    t_start = time.monotonic() + start_delay
    frame_latency = 1/30
    precise_wait(eval_t_start - frame_latency, time_func=time.time)
    print("Started!")
    iter_idx = 0

    # inference loop
    while not rospy.is_shutdown():
        test_t_start = time.perf_counter()
        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

        # get observation
        k = math.ceil(n_obs_steps * (video_capture_fps / frequency))
        last_data = obs_ring_buffer.get_last_k(k=k, out=last_data)
        last_timestamp = last_data["timestamp"][-1]
        obs_align_timestamps = last_timestamp - (np.arange(n_obs_steps)[::-1] * dt)

        obs_dict = dict()
        this_timestamps = last_data["timestamp"]
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps <= t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)
        for key in last_data.keys():
            obs_dict[key] = last_data[key][this_idxs]

        obs_timestamps = obs_dict["timestamp"]
        print("Got Observation!")

        # run inference
        with torch.no_grad():
            obs_dict_np = get_real_obs_dict(
                env_obs=obs_dict, shape_meta=cfg.task.shape_meta)
            obs_dict = dict_apply(obs_dict_np,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(torch.device("cuda:0")))

            result = policy.predict_action(obs_dict)

            action = result["action"][0].detach().to("cpu").numpy()

        # preprocess action
        action = action[:steps_per_inference, :]
        action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]

        action_exec_latency = 0.01
        curr_time = time.time()
        is_new = action_timestamps > (curr_time + action_exec_latency)

        if np.sum(is_new) == 0:
            action = action[[-1]]
            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
            action_timestamp = eval_t_start + (next_step_idx) * dt
            print("Over budget", action_timestamp - curr_time)
            action_timestamps = np.array([action_timestamp])
        else:
            action = action[is_new]
            action_timestamps = action_timestamps[is_new]
        # execute actions
        print("Execute Action!")
        for item in action:
            right_control.joint_pos = item
            control_robot2.publish(right_control)
            rate.sleep()

        precise_wait(t_cycle_end - frame_latency)
        iter_idx += steps_per_inference

        print(f"Inference Actual frequency {steps_per_inference/(time.perf_counter() - test_t_start)}")


def callback(agent_pos, depth_msg):
    global obs_ring_buffer

    bridge = CvBridge()
    receive_time = time.time()

    obs_data = dict()
    obs_data["agent_pos"] = agent_pos.joint_pos
    # process depth observation to point cloud
    try:
        # 假设深度图像是16位单通道 (e.g., '16UC1') 或者 32位浮点单通道 ('32FC1')
        # 如果是毫米单位的 uint16:
        depth_cv2 = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
        # 如果已经是米单位的 float32:
        # depth_cv2 = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")
        return

    # 确保深度图的分辨率与内参匹配
    if depth_cv2.shape[0] != DEPTH_IMG_HEIGHT or depth_cv2.shape[1] != DEPTH_IMG_WIDTH:
        # 如果不匹配，可能需要缩放深度图，但这会引入误差，最好相机输出就匹配
        # 或者调整 CAMERA_CX, CAMERA_CY, DEPTH_IMG_HEIGHT, DEPTH_IMG_WIDTH
        rospy.logwarn_throttle(10, f"Depth image resolution ({depth_cv2.shape}) "
                                   f"does not match expected ({DEPTH_IMG_HEIGHT},{DEPTH_IMG_WIDTH}). "
                                   "Point cloud accuracy may be affected.")
        # 简单的缩放示例 (如果必须)
        depth_cv2 = cv2.resize(depth_cv2, (DEPTH_IMG_WIDTH, DEPTH_IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)


    # 将深度图转换为点云
    # 注意：depth_to_pointcloud 内部应处理单位转换 (如毫米到米) 和采样
    point_cloud_data = depth_to_pointcloud(depth_cv2,
                                           CAMERA_FX, CAMERA_FY,
                                           CAMERA_CX, CAMERA_CY)

    # obs_data["depth"] = transform(depth_cv2) # 旧的，transform 是为RGB设计的
    obs_data["point_cloud"] = point_cloud_data # 新的
    obs_data["timestamp"] = receive_time

    put_data = obs_data
    obs_ring_buffer.put(put_data, wait=False)


def transform(data, video_capture_resolution=(640, 480), obs_image_resolution=(160, 120)):
    color_tf = get_image_transform(
                input_res=video_capture_resolution,
                output_res=obs_image_resolution,
                # obs output rgb
                bgr_to_rgb=True,
            )
    
    tf_data = color_tf(data)
    return tf_data

def depth_to_pointcloud(depth_image_cv2, fx, fy, cx, cy):
    """
    Converts a depth image to a point cloud.
    Args:
        depth_image_cv2: (H, W) numpy array, depth in meters (or needs conversion to meters).
        fx, fy: focal lengths.
        cx, cy: principal points.
    Returns:
        point_cloud: (N, 3) numpy array of XYZ coordinates, or (H*W, 3) if dense.
                     Invalid points (e.g., depth=0) should be filtered out or handled.
    """
    h, w = depth_image_cv2.shape
    # 创建像素坐标网格
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

    # 深度值 (确保是米单位，如果原始是毫米，需要 /1000.0)
    # 假设 depth_image_cv2 已经是浮点型并且单位是米
    # 如果是 uint16 毫米，需要先转换: depth_in_meters = depth_image_cv2.astype(np.float32) / 1000.0
    depth_in_meters = depth_image_cv2.astype(np.float32) # 假设已经是米
    if depth_image_cv2.dtype == np.uint16: # 如果是毫米的uint16
        depth_in_meters = depth_image_cv2.astype(np.float32) / 1000.0


    # 计算 X, Y, Z
    # Z 是深度值本身
    Z = depth_in_meters
    X = (i - cx) * Z / fx
    Y = (j - cy) * Z / fy

    # 堆叠成 (H, W, 3)
    point_cloud_image = np.stack((X, Y, Z), axis=-1)

    # 展平成 (H*W, 3) 并移除无效点 (例如深度为0或非常大的点)
    points = point_cloud_image.reshape(-1, 3)
    valid_depth_mask = (Z.ravel() > 0.1) & (Z.ravel() < 5.0) # 示例有效深度范围 0.1m to 5m
    valid_points = points[valid_depth_mask]

    # --- 重要：点云采样/处理 ---
    # 实际应用中，你可能不想要所有的 H*W 个点。
    # 你可能需要：
    # 1. 固定数量采样：例如，随机采样 N 个点，或者使用 Farthest Point Sampling (FPS)
    #    e.g., if len(valid_points) > target_num_points:
    #             indices = np.random.choice(len(valid_points), target_num_points, replace=False)
    #             sampled_points = valid_points[indices]
    #          else:
    #             # 处理点数不足的情况 (填充或报错)
    #             sampled_points = valid_points # 或者填充
    # 2. 体素下采样
    # 这里的实现返回了所有有效点，你需要根据你的策略模型期望的输入来调整。
    # 假设你的 'shape_meta' 中定义的点云形状是 (N, 3)
    TARGET_NUM_POINTS = 1024 # 从你的配置中获取，或者硬编码
    if len(valid_points) >= TARGET_NUM_POINTS:
        indices = np.random.choice(len(valid_points), TARGET_NUM_POINTS, replace=False)
        sampled_points = valid_points[indices]
    elif len(valid_points) > 0 : # 点数不足，用重复填充
        indices = np.random.choice(len(valid_points), TARGET_NUM_POINTS, replace=True)
        sampled_points = valid_points[indices]
    else: # 没有有效点
        sampled_points = np.zeros((TARGET_NUM_POINTS, 3), dtype=np.float32)

    return sampled_points

if __name__ == "__main__":
    main()
