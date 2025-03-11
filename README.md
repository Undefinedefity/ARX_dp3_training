## Installation

```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf  
$ conda env create -f conda_environment.yaml
```

## Path determination

1. dataset_path in config/task
2. task in config(workspace)

## Image Resizing

To resize images stored in HDF5 files, use the `resize_image.py` script with the following command-line options:

```shell
$ python resize_image.py --input_path /path/to/input --output_path /path/to/output --num_episodes 100 --target_size 224 224
```

- `--input_path`: The base path to the input dataset.
- `--output_path`: The base path where the resized images will be saved.
- `--num_episodes`: The number of episodes to process.
- `--target_size`: The target size for the images, specified as width and height.

## Training

```shell
$ conda activate robodiff
$ wandb login
$ python train.py --config-name=train_diffusion_unet_real_image_workspace.yaml
```
