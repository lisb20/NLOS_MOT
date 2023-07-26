import time

from numpy import ndarray
import similaritymeasures

from torch import Tensor, no_grad
from torch.nn import Module
from torch.cuda.amp import autocast
from tqdm import tqdm

from .tools import get_device, AverageMeter


def test_metrics(model: Module, loader, warm_up: int = None):
    device = get_device(model)
    model.eval()  # set model to evaluation mode
    tic = time.time()

    pcm_recorder = AverageMeter()
    area_recorder = AverageMeter()
    dtw_recorder = AverageMeter()

    with no_grad():
        with autocast():
            for batch_idx, (X, Y) in tqdm(enumerate(loader)):
                X = X.to(device)  # move to device, e.g. GPU
                Y = Y.to(device)

                batch_size, T = Y.shape[:2]

                _, preds = model((X, Y))
                if warm_up is not None:
                    T = warm_up - T
                    Y = Y[:, T:]
                    preds = preds[:, T:]
                pcm, area, dtw = compute_batch_metrics(preds, Y)
                pcm_recorder.update(pcm.item(), batch_size)
                area_recorder.update(area.item(), batch_size)
                dtw_recorder.update(dtw.item(), batch_size)

    print(f"Infer time: {time.time() - tic}")
    print(f"pcm: {pcm_recorder.avg:.4f}\n"
          f"area: {area_recorder.avg:.4f}\n"
          f"dtw: {dtw_recorder.avg:.4f}\n")


def compute_track_metrics(pred: ndarray, gt: ndarray):
    ## pred gt: t*10
    T = gt.shape[0]
    pcm_tot = 0
    area_tot = 0
    dtw_tot = 0
    cnt = 0
    for p in range(3):
        id1 = 2*p
        id2 = id1+2
        if all(gt[:,id1] == 0.5):
            break
        cnt += 1
        pcm = similaritymeasures.pcm(gt[:,id1:id2], pred[:,id1:id2])
        area = similaritymeasures.area_between_two_curves(gt[:,id1:id2], pred[:,id1:id2])
        dtw = similaritymeasures.dtw(gt[:,id1:id2], pred[:,id1:id2])[0]
        pcm_tot += pcm
        area_tot += area
        dtw_tot += dtw

    return pcm_tot / (T*cnt), area_tot / (T*cnt), dtw_tot / (T*cnt)


def compute_batch_metrics(preds: Tensor, labels: Tensor):
    pcm, area, dtw = 0, 0, 0
    B = labels.shape[0]
    for gt, pred in zip(labels.cpu().numpy(), preds.detach().cpu().numpy()):
        metrics = compute_track_metrics(gt, pred)
        pcm += metrics[0]
        area += metrics[1]
        dtw += metrics[2]
    return pcm/B, area/B, dtw/B
