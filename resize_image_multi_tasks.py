import h5py
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Tuple, List, Dict
import argparse
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageResizer:
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        """
        初始化图像处理器
        
        Args:
            target_size: 目标图像尺寸，默认为(128, 128)
        """
        self.target_size = target_size
        
    def resize_single_image(self, image: np.ndarray) -> np.ndarray:
        """
        调整单张图片的大小
        
        Args:
            image: 输入图像数组
            
        Returns:
            调整大小后的图像数组
        """
        try:
            pil_image = Image.fromarray(image)
            resized_image = pil_image.resize(self.target_size, Image.LANCZOS)
            return np.array(resized_image)
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            raise
            
    def resize_batch(self, images: np.ndarray) -> np.ndarray:
        """
        批量调整图片大小
        
        Args:
            images: 输入图像数组批次
            
        Returns:
            调整大小后的图像数组批次
        """
        return np.array([self.resize_single_image(img) for img in images])

class HDF5Handler:
    def __init__(self, input_path: str, output_path: str):
        """
        初始化HDF5文件处理器
        
        Args:
            input_path: 数据集基础路径
            output_path: 输出数据集基础路径
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
    def get_file_path(self, episode_num: int) -> Tuple[Path, Path]:
        """
        获取特定episode的文件路径
        
        Args:
            episode_num: episode编号
            
        Returns:
            输入文件路径和输出文件路径
        """
        input_file_path = self.input_path / f"episode_{episode_num}.hdf5"
        output_file_path = self.output_path / f"episode_{episode_num}.hdf5"
        return input_file_path, output_file_path
    
    def process_single_file(self, input_file_path: Path, output_file_path: Path, resizer: ImageResizer, task_emb: np.ndarray) -> None:
        """
        处理单个HDF5文件
        
        Args:
            input_file_path: 输入HDF5文件路径
            output_file_path: 输出HDF5文件路径
            resizer: 图像处理器实例
            task_emb: 任务嵌入向量
        """
        try:
            # 确保输出目录存在
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 读取输入文件并创建新的输出文件
            with h5py.File(input_file_path, 'r') as f_in, h5py.File(output_file_path, 'w') as f_out:
                # 获取action的长度
                action_length = f_in['action'].shape[0]
                
                # 复制所有原始数据
                for key in f_in.keys():
                    if key != 'observations':
                        f_in.copy(key, f_out)
                
                # 确保observations组存在
                if 'observations' not in f_out:
                    f_out.create_group('observations')
                
                # 复制observations中除images外的所有数据
                obs_group = f_in['observations']
                for key in obs_group.keys():
                    if key != 'images':
                        obs_group.copy(key, f_out['observations'])
                
                # 处理图像
                if 'images' not in f_out['observations']:
                    f_out['observations'].create_group('images')
                    
                for camera in ['mid', 'right']:
                    # 读取原始图像
                    images = f_in[f'/observations/images/{camera}'][:]
                    
                    # 调整图像大小
                    resized_images = resizer.resize_batch(images)
                    
                    # 创建新的数据集
                    f_out.create_dataset(f'/observations/images/{camera}', data=resized_images, dtype='uint8')
                
                # 添加任务嵌入向量，复制与action长度相同的次数
                repeated_task_emb = np.tile(task_emb, (action_length, 1))
                f_out.create_dataset('observations/task_emb', data=repeated_task_emb)
                
            logger.info(f"Successfully processed {input_file_path} to {output_file_path}")
            
        except Exception as e:
            logger.error(f"Error processing file {input_file_path}: {e}")
            raise

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Resize images in HDF5 files.")
    parser.add_argument('--input_path', type=str, required=True, help='输入数据集基础路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出数据集基础路径')
    parser.add_argument('--num_episodes', type=int, required=True, help='需要处理的episode数量')
    parser.add_argument('--target_size', type=int, nargs=2, default=(128, 128), help='目标图像尺寸，格式为: width height')
    
    return parser.parse_args()

def get_task_emb_real(TASK):
    emb = np.load('/data/ouyangzikai/Data/ATM_real_data/ATM_0327_45_25mm_multi_tasks/task_embedding_caches_sponge/task_emb_bert.npy', allow_pickle=True).item() 
    if TASK == "sponge":
        task = 'put the sponge onto the plate'
    elif TASK == "carrot":
        task = 'put the carrot into the basket'
    elif TASK == "banana":
        task = 'put the banana into the basket'
    else:
        print(f"ERROR: TASK {TASK} not in task list")
        exit()
    assert task in emb.keys(), f'Task "{task}" is not in the bert embedding cache. Please check.'
    task_emb = emb[task]
    return task_emb

def process_all_subtasks(input_base_path: str, output_base_path: str, num_episodes: int, target_size: Tuple[int, int]):
    """
    Process all subtask directories and resize images, storing task embeddings.
    
    Args:
        input_base_path: Base path containing subtask directories.
        output_base_path: Base path to store processed files.
        num_episodes: Number of episodes to process per subtask.
        target_size: Target size for image resizing.
    """
    resizer = ImageResizer(target_size)
    episode_counter = 0
    
    for subtask_dir in Path(input_base_path).iterdir():
        if subtask_dir.is_dir():
            task_name = subtask_dir.name
            task_emb = get_task_emb_real(task_name)
            
            hdf5_handler = HDF5Handler(subtask_dir, output_base_path)
            
            for episode_num in tqdm(range(num_episodes), desc=f"Processing {task_name} episodes"):
                input_file_path, _ = hdf5_handler.get_file_path(episode_num)
                
                # 使用累计计数生成输出文件名
                output_file_path = Path(output_base_path) / f"episode_{episode_counter}.hdf5"
                episode_counter += 1
                
                hdf5_handler.process_single_file(input_file_path, output_file_path, resizer, task_emb)
                
            logger.info(f"All episodes for {task_name} processed successfully")

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    try:
        process_all_subtasks(args.input_path, args.output_path, args.num_episodes, tuple(args.target_size))
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
