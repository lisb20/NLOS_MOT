
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from data.dataset import split_dataset
from utils.vis import *
from utils.tools import load_model
from utils.trainer import _set_seed
from matplotlib import pyplot as plt
import cv2


plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots

## srun -p optimal --quotatype=auto --gres=gpu:1 -J NLOS_test python test.py
# life save magic code
print(f"torch version: {torch.__version__}")
use_cuda = torch.cuda.is_available()
if use_cuda:
    GPU_nums = torch.cuda.device_count()
    GPU = torch.cuda.get_device_properties(0)
    print(f"There are {GPU_nums} GPUs in total.\nThe first GPU: {GPU}")
    print(f"CUDA version: {torch.version.cuda}")
device = torch.device(f"cuda:0" if use_cuda else "cpu")
print(f"Using {device} now!")

# # Load Model & Data


# Fill your run name and log dir!
run_name = "2023_07_13_17_38_22"
log_dir = "./log"
model = load_model(run_name, log_dir,ckpt_name="epoch_70").to(device)
# _set_seed(seed=1026, deterministic=True)

train_dataset, val_dataset = split_dataset(
    dataset_root='/mnt/petrelfs/share_data/lisibo/NLOS/data_render_mot/1', # Fill your dataset root!
    # dataset_root = "./dataset/real_shot_new/1",
    # dataset_root = "./dataset/render",
    train_ratio=0,
    route_len=160,
    data_type = "real_shot"
    )

loader_kwargs = {
    'batch_size' : 3,
    'num_workers': 4,
    'pin_memory': True,
    'prefetch_factor': 4,
    'persistent_workers': True
}
# train_loader = DataLoader(train_dataset, **loader_kwargs)
# iter_train_loader = iter(train_loader)
val_loader = DataLoader(val_dataset, **loader_kwargs)
iter_val_loader = iter(val_loader)
frames, gt_routes, map_sizes = next(iter_val_loader)
frames = frames.to(device)
gt_routes = gt_routes.to(device)
print(frames.shape, gt_routes.shape)

# %%
with torch.no_grad():
    with autocast():
        pred_routes = model.vis_forward((frames, gt_routes))
for idx, (gt, pred) in enumerate(zip(gt_routes.cpu().numpy(), pred_routes.cpu().numpy())):
    # fig = draw_routes_mot(routes=(gt, pred), return_mode = "fig_array")
    fig = draw_routes(routes=(gt, pred), return_mode = "fig_array")
    img = cv2.cvtColor(fig, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(f"./test/test_render/{idx}.png", fig)


