import gc
import h5py
import time
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from data.dataset import TiledDataset
from data.transforms import build_transforms
from training_module import SegmentationModule

def load_data(h5_path, patch_size,  batch_size, indices=None, stride=None, transform=None, num_workers=2):
    dataset = TiledDataset(h5_path, patch_size=patch_size, transform=transform,
                            use_cache=True, stride=stride, indices=indices)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True,
                            prefetch_factor=4, persistent_workers=True)
    return dataloader

def load_model(ckpt_path, device="cuda"):
    model = SegmentationModule.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()
    return model

# Comment to add unadded changes to git.

def build_eval_transform():
    data_cfg = {"normalization": "imagenet",}
    transform = build_transforms(data_cfg, mode="val")
    return transform

def update_pr_auc_counts(pred_mask, gt_mask, thresholds, tp, fp, fn,):
    # Flatten to 1D (P = H * W)
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten().bool()
    
    # thresholds[:, None] -> shape (T, 1)
    # pred_flat[None, :]  -> shape (1, P)
    # Result: (T, P) boolean matrix where row t is (prob >= threshold[t])
    pred_binary = thresholds[:, None] <= pred_flat[None, :]
    
    # gt_flat[None, :] -> shape (1, P) broadcasts to (T, P)
    gt_binary = gt_flat[None, :]
    
    # Sum along axis=1 (across pixels) gives shape (T,)
    tp += torch.sum(pred_binary & gt_binary, axis=1)
    fp += torch.sum(pred_binary & ~gt_binary, axis=1)
    fn += torch.sum(~pred_binary & gt_binary, axis=1)


def make_predictions_and_count(loader, model, h5_path, patch_size, threshold=0.5, compute_pr_auc=False):
    tp = fp = fn = tn = 0

    if compute_pr_auc:
        thresholds = torch.linspace(0, 1, 101)
        tp_auc = torch.zeros(101, dtype=torch.int64)
        fp_auc = torch.zeros(101, dtype=torch.int64)
        fn_auc = torch.zeros(101, dtype=torch.int64)

    current_img = None
    full_pred = None
    count_map = None

    loader_time = infer_time = stitch_time = 0
    batch_end = time.time()

    with h5py.File(h5_path, 'r') as f:
        masks = f["masks"]
        pad_h, pad_w = loader.dataset.pad_h, loader.dataset.pad_w
        with torch.inference_mode():
            for batch in tqdm(loader, desc="Predicting patches", unit="batch", total=len(loader)):
                loader_time += time.time() - batch_end

                images, _, img_idx, y, x = batch
                images = images.cuda(non_blocking=True)

                torch.cuda.synchronize()

                t1 = time.time()

                preds = torch.sigmoid(model(images))

                torch.cuda.synchronize()
                infer_time += time.time() - t1

                preds = preds.cpu()

                t2 = time.time()
                for i in range(preds.shape[0]):
                    img = img_idx[i].item()
                    yi = y[i].item()
                    xi = x[i].item()

                    # NEW IMAGE → finalize previous one
                    if current_img is not None and img != current_img:
                        avg = full_pred / torch.clamp(count_map, min=1)
                        avg = avg[:orig_h, :orig_w]

                        gt = torch.from_numpy(masks[current_img]).float()

                        if compute_pr_auc:
                            update_pr_auc_counts(avg, gt, thresholds, tp_auc, fp_auc, fn_auc)
                        else:
                            pred_mask = (avg >= threshold)

                            tp += ((pred_mask == 1) & (gt == 1)).sum().item()
                            fp += ((pred_mask == 1) & (gt == 0)).sum().item()
                            fn += ((pred_mask == 0) & (gt == 1)).sum().item()
                            tn += ((pred_mask == 0) & (gt == 0)).sum().item()

                        # cleanup
                        del full_pred, count_map

                    # initialize new image
                    if img != current_img:
                        current_img = img

                        orig_h, orig_w = masks[img].shape
                        padded_h = orig_h + pad_h
                        padded_w = orig_w + pad_w

                        full_pred = torch.zeros(padded_h, padded_w, dtype=torch.float16)
                        count_map = torch.zeros(padded_h, padded_w, dtype=torch.float16)

                    patch = preds[i].squeeze()

                    full_pred[yi:yi+patch_size, xi:xi+patch_size] += patch
                    count_map[yi:yi+patch_size, xi:xi+patch_size] += 1

                del images, preds

                stitch_time += time.time() - t2
                batch_end = time.time()


        # finalize last image
        if current_img is not None:
            avg = full_pred / torch.clamp(count_map, min=1)
            avg = avg[:orig_h, :orig_w]

            gt = torch.from_numpy(masks[current_img]).float()

            if compute_pr_auc:
                update_pr_auc_counts(avg, gt, thresholds, tp_auc, fp_auc, fn_auc)
            else:
                pred_mask = (avg > threshold)
                tp += ((pred_mask == 1) & (gt == 1)).sum().item()
                fp += ((pred_mask == 1) & (gt == 0)).sum().item()
                fn += ((pred_mask == 0) & (gt == 1)).sum().item()
                tn += ((pred_mask == 0) & (gt == 0)).sum().item()

    del masks
    gc.collect()
    print(f"Loading time: {loader_time:.2f}s, Inference time: {infer_time:.2f}s, Stitching time: {stitch_time:.2f}s")

    if compute_pr_auc:
        return {"tp": tp_auc, "fp": fp_auc, "fn": fn_auc, "thresholds": thresholds}

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
