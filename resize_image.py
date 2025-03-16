import h5py
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Tuple, List, Dict
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageResizer:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        初始化图像处理器
        
        Args:
            target_size: 目标图像尺寸，默认为(224, 224)
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
    
    def process_single_file(self, input_file_path: Path, output_file_path: Path, resizer: ImageResizer) -> None:
        """
        处理单个HDF5文件
        
        Args:
            input_file_path: 输入HDF5文件路径
            output_file_path: 输出HDF5文件路径
            resizer: 图像处理器实例
        """
        try:
            with h5py.File(input_file_path, 'r+') as f:
                for camera in ['mid', 'right']:
                    # 读取原始图像
                    images = f[f'/observations/images/{camera}'][:]
                    
                    # 调整图像大小
                    resized_images = resizer.resize_batch(images)
                    
                    # 更新数据集
                    dataset_path = f'/observations/images/{camera}'
                    if dataset_path in f:
                        del f[dataset_path]
                    f.create_dataset(dataset_path, data=resized_images, dtype='uint8')
                    
                logger.info(f"Successfully processed {input_file_path}")
                
            # 移动文件
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            input_file_path.rename(output_file_path)
                
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

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    try:
        # 初始化处理器
        resizer = ImageResizer(tuple(args.target_size))
        hdf5_handler = HDF5Handler(args.input_path, args.output_path)
        
        # 处理所有文件
        for episode_num in tqdm(range(args.num_episodes), desc="Processing episodes"):
            input_file_path, output_file_path = hdf5_handler.get_file_path(episode_num)
            hdf5_handler.process_single_file(input_file_path, output_file_path, resizer)
            
        logger.info("All episodes processed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
