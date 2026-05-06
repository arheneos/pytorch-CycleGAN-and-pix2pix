# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

PyTorch implementations of **CycleGAN** (unpaired image-to-image translation) and **pix2pix** (paired image-to-image translation). Supports Python 3.11 and PyTorch 2.4+, with DDP for multi-GPU training.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate pytorch-img2img
```

Key dependencies: PyTorch 2.4, torchvision 0.19, numpy, scikit-image, dominate, Pillow, wandb.

## Common Commands

**Train CycleGAN:**
```bash
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```

**Train pix2pix:**
```bash
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```

**Test a model:**
```bash
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
# Results saved to ./results/<name>/latest_test/index.html
```

**Resume training:**
```bash
python train.py ... --continue_train
```

**Multi-GPU training (DDP):**
```bash
torchrun --nproc_per_node=4 train.py ... --norm sync_instance
```
Note: `--norm batch` is incompatible with DDP; use `sync_instance` or `syncbatch`.

**Download datasets:**
```bash
bash ./datasets/download_cyclegan_dataset.sh <dataset_name>
bash ./datasets/download_pix2pix_dataset.sh <dataset_name>
```

**Download pretrained models:**
```bash
bash ./scripts/download_cyclegan_model.sh horse2zebra
bash ./scripts/download_pix2pix_model.sh facades_label2photo
```

## Testing and Linting

**Before pushing:**
```bash
flake8 --ignore E501 .
pytest scripts/test_before_push.py -v
```

The test suite downloads mini datasets automatically and verifies train+test pipelines for CycleGAN, pix2pix, colorization, and the template model.

## Architecture

### Plugin Pattern (Models and Datasets)

Both `models/` and `data/` use a **dynamic import pattern**: `--model cycle_gan` imports `models/cycle_gan_model.py` and finds the class `CycleGANModel`; `--dataset_mode aligned` imports `data/aligned_dataset.py` and finds `AlignedDataset`. Class names must match `<name>Model` / `<name>Dataset` (case-insensitive).

Adding a new model: create `models/<name>_model.py` with a `<Name>Model(BaseModel)` subclass. Adding a new dataset: create `data/<name>_dataset.py` with `<Name>Dataset(BaseDataset)`.

### Model Lifecycle

`BaseModel` (`models/base_model.py`) defines the interface. Each model subclass must implement:
- `set_input(data)` — unpack batch from dataloader
- `forward()` — run the forward pass
- `optimize_parameters()` — compute losses, backprop, update weights
- (optional) `modify_commandline_options(parser, is_train)` — add model-specific CLI args

During `setup()`, networks are initialized, optionally loaded from checkpoints, moved to device, and wrapped with `DistributedDataParallel` if DDP is active. Only rank-0 saves checkpoints.

Model subclasses declare four lists in `__init__`:
- `self.loss_names` — losses to log
- `self.model_names` — network names (networks stored as `self.net<Name>`)
- `self.visual_names` — tensors to visualize (stored as `self.<name>`)
- `self.optimizers` — for LR scheduling

### Key Models

- **`cycle_gan_model.py`** — Two generators (G_A: A→B, G_B: B→A) and two discriminators (D_A, D_B). Cycle-consistency and identity losses. Uses `ImagePool` to store 500 past generated images for discriminator updates.
- **`pix2pix_model.py`** — Generator + discriminator with paired data. Uses U-Net generator and PatchGAN discriminator.
- **`colorization_model.py`** — Subclass of pix2pix; maps L channel → ab channels (Lab color space).
- **`test_model.py`** — Inference-only wrapper for single-direction CycleGAN results.

### Networks (`models/networks.py`)

Defines all architecture components: ResNet generators, U-Net generators, PatchGAN discriminators (including attention variant), normalization layers, weight initialization, LR schedulers, and GAN loss functions (`vanilla`, `lsgan`, `wgangp`).

### Options

`BaseOptions` → `TrainOptions` / `TestOptions`. Options are gathered in two passes: base options first, then model/dataset `modify_commandline_options` hooks add model-specific args. Default image size: `load_size=120`, `crop_size=64`, `batch_size=64`.

### Data

- `unaligned_dataset.py` — expects `trainA/`, `trainB/` subdirectories (CycleGAN)
- `aligned_dataset.py` — expects side-by-side image pairs in `train/` (pix2pix)
- `single_dataset.py` — single directory, used with `--model test`
- `colorization_dataset.py` — RGB images converted to Lab pairs

DDP: `CustomDatasetDataLoader` automatically uses `DistributedSampler` when `LOCAL_RANK` is set in the environment.

### Outputs

- Checkpoints: `./checkpoints/<name>/` — saved as `<epoch>_net_<Name>.pth` and `latest_net_<Name>.pth`
- Results: `./results/<name>/<phase>_<epoch>/` — images + `index.html`
- Intermediate visuals during training: `./checkpoints/<name>/web/index.html`

### Custom Scripts in This Repo

`test2.py` is a standalone inference script (not the standard `test.py`) that loads `.bin` files from `train/` and `.npy` files from `real/`, applies a saved `latest_net_G_A.pth` generator, and displays results with OpenCV. It uses a custom piecewise normalization (`normalize_real_range` / `restore_real_range`) for non-image float data.
