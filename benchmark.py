import os
import gc
import h5py
import argparse
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
import time

import torch
from torch.utils.data import DataLoader

from utils import compute_metrics
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


def make_predictions_and_count(loader, model, h5_path, patch_size):
    tp = fp = fn = tn = 0

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
                        pred_mask = (avg > 0.5)

                        gt = torch.from_numpy(masks[current_img]).float()

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
            pred_mask = (avg > 0.5)

            gt = torch.from_numpy(masks[current_img]).float()

            tp += ((pred_mask == 1) & (gt == 1)).sum().item()
            fp += ((pred_mask == 1) & (gt == 0)).sum().item()
            fn += ((pred_mask == 0) & (gt == 1)).sum().item()
            tn += ((pred_mask == 0) & (gt == 0)).sum().item()

    del masks
    gc.collect()
    print(f"Loading time: {loader_time:.2f}s, Inference time: {infer_time:.2f}s, Stitching time: {stitch_time:.2f}s")

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def run_benchmark(model, h5_root, dataset_dict, patch_size, batch_size, stride, dataset_flags):
    results = {}

    for enabled, (name, config) in zip(dataset_flags, dataset_dict.items()):
        if enabled != "1":
            continue

        dataset_path = h5_root / config["file"]

        data_cfg = {"normalization": "imagenet",}
        transform = build_transforms(data_cfg, mode="val")

        loader = load_data(
            dataset_path,
            patch_size=patch_size,
            batch_size=batch_size,
            transform=transform,
            stride=stride,
            indices=config.get("indices", None)
        )

        counts = make_predictions_and_count(
            loader,
            model,
            dataset_path,
            patch_size
        )

        results[name] = compute_metrics(
            counts["tp"],
            counts["fp"],
            counts["fn"],
            counts["tn"]
        )

        del loader
        torch.cuda.empty_cache()
        gc.collect()

    return results


def get_inria_city_indices(h5_path):
    cities = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]

    with h5py.File(h5_path, "r") as f:
        filenames = f["filenames"][:]

    city_indices = {city: [] for city in cities}

    for idx, filename in enumerate(filenames):
        filename = filename.decode() if isinstance(filename, bytes) else filename

        for city in cities:
            if filename.startswith(city):
                city_indices[city].append(idx)
                break

    return city_indices


def save_results_to_csv(results, config_name, csv_path="benchmark_results.csv"):
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    rows = [
        {
            "config": config_name,
            "dataset": dataset,
            "timestamp": timestamp,
            **metrics
        }
        for dataset, metrics in results.items()
    ]

    df = pd.DataFrame(rows)

    header = (
        not os.path.exists(csv_path)
        or os.path.getsize(csv_path) == 0
    )

    df.to_csv(
        csv_path,
        mode="a",
        header=header,
        index=False
    )

def main():
    parser = argparse.ArgumentParser(description="Benchmarking script for WHU building segmentation.")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to the HDF5 dataset.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--patch_size", type=int, required=True, help="Patch size for testing.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for testing.")
    parser.add_argument("--dataset_flags", type=str, default="111", help="Which datasets to evaluate, show with a 3-digit boolean code,"
                        " first digit for WHU Test, second for Massachusetts, third for INRIA. E.g. '110' means evaluate only WHU Test and Massachusetts.")
    parser.add_argument("--stride", type=int, default=None, help="Stride for testing.")
    parser.add_argument("--dest_dir", type=str, help="Directory to save results CSV.")
    parser.add_argument("--dest_file1", type="str", default="benchmark_results.csv", help="File name to store WHU and Massachusetts benchmarks.")
    parser.add_argument("--dest_file2", type="str", default="inria_benchmark_results.csv", help="File name to store INRIA benchmarks.")

    
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    h5_path = Path(args.h5_path)
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        print(f"Checkpoint path {ckpt_path} does not exist.")
        return

    model = load_model(args.ckpt_path)

    results = inria_results = None

    if args.dataset_flags[:2] != "00":
        dataset_dict = {
            "WHU Test": {
                "file": "whu_test.h5"
            },
            "Massachusetts": {
                "file": "mas_test.h5"
            }
        }

        results = run_benchmark(model, h5_path, dataset_dict, args.patch_size,
                                 args.batch_size, args.stride, args.dataset_flags[:2])
    
        save_results_to_csv(results, config_name=ckpt_path.stem, csv_path=dest_dir / args.dest_file1)
    
    if args.dataset_flags[2] == "1":
        city_indices = get_inria_city_indices(h5_path / "inria_val.h5")

        inria_datasets = {
            "austin": {
                "file": "inria_val.h5",
                "indices": city_indices["austin"]
            },
            "chicago": {
                "file": "inria_val.h5",
                "indices": city_indices["chicago"]
            },
            "kitsap": {
                "file": "inria_val.h5",
                "indices": city_indices["kitsap"]
            },
            "tyrolw": {
                "file": "inria_val.h5",
                "indices": city_indices["tyrol-w"]
            },
            "vienna": {
                "file": "inria_val.h5",
                "indices": city_indices["vienna"]
            }
        }

        inria_results = run_benchmark(model, h5_path, inria_datasets, args.patch_size,
                                       args.batch_size, args.stride, "1"*len(inria_datasets))
        
        cf = {i: sum([inria_results[j][i] for j in inria_results]) for i in ["tp", "fp", "fn", "tn"]}
        inria_results["overall"] = compute_metrics(**cf)

        save_results_to_csv(inria_results, config_name=ckpt_path.stem, csv_path=dest_dir / args.dest_file2)


if __name__ == "__main__":
    main()

