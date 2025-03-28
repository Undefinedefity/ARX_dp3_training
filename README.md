## Installation

```console 
$ conda env create -f conda_environment.yaml
$ conda activate robodiff
```

## Image Resizing

To resize images stored in HDF5 files, use the `resize_image.py` script with the following command-line options:

```shell
$ python resize_image.py --input_path /path/to/input --output_path /path/to/output --num_episodes 50 --target_size 128 128
```

- `--input_path`: The base path to the input dataset.
- `--output_path`: The base path where the resized images will be saved.
- `--num_episodes`: The number of episodes to process.
- `--target_size`: The target size for the images, specified as width and height.

## Path determination

1. **Task Configuration**: Modify the following parameters in `diffusion_policy/conf/task/{task_name}.yaml`:
   - `image_shape`: Specify the shape of the images.
   - `dataset_dir`: Set the directory for the dataset.
   - `num_episodes`: Define the number of episodes to process.

2. **Workspace Configuration**: Update these parameters in `diffusion_policy/conf/xxx_workspace.yaml`:
   - `default.task`: Set the default task.
   - `crop_shape`: Specify the crop shape, which should be smaller than the target size.
   - `num_epochs`: Define the number of epochs for training.

## Training

```shell
$ conda activate robodiff
$ wandb login
$ python train.py --config-name=train_diffusion_unet_real_image_workspace.yaml
```
