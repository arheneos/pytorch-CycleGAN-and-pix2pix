set -ex
torchrun --nproc_per_node=2 train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --pool_size 50 --no_dropout --norm sync_batch  --use_wandb
```
