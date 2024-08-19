from argparse import ArgumentParser
import os
import json
import time
import numpy as np
import pandas as pd
from data import create_dataset_with_args
from models import create_model
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from models.utils.config import OptConfig


def eval_miss(model, val_iter):
    model.eval()
    total_pred = []
    total_label = []
    total_miss_type = []
    batch_times = []
    for _, data in enumerate(val_iter):  # inner loop within one epoch
        start = time.time()
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        end = time.time()
        label = data["label"]
        miss_type = np.array(data["miss_type"])
        total_pred.append(pred)
        total_label.append(label)
        total_miss_type.append(miss_type)
        batch_times.append(end - start)

    # calculate metrics
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    total_miss_type = np.concatenate(total_miss_type)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average="macro")
    f1 = f1_score(total_label, total_pred, average="weighted")
    cm = confusion_matrix(total_label, total_pred)

    print(f"Total acc:{acc:.4f} uar:{uar:.4f} f1:{f1:.4f}")
    data = {}
    for part_name in ["azz", "zvz", "zzl", "avz", "azl", "zvl", "avl"]:
        part_index = np.where(total_miss_type == part_name)
        part_pred = total_pred[part_index]
        part_label = total_label[part_index]
        acc_part = accuracy_score(part_label, part_pred)
        uar_part = recall_score(part_label, part_pred, average="macro")
        f1_part = f1_score(part_label, part_pred, average="weighted")
        print(f"{part_name}, acc:{acc_part:.4f}, {uar_part:.4f}, {f1_part:.4f}")
        data[part_name] = [acc_part, uar_part, f1_part]

    print(f"Average batch time: {np.mean(batch_times):.4f}")
    return acc, uar, f1, cm, data


def eval_all(model, val_iter):
    model.eval()
    total_pred = []
    total_label = []
    for _, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data["label"]
        total_pred.append(pred)
        total_label.append(label)

    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average="macro")
    f1 = f1_score(total_label, total_pred, average="weighted")
    cm = confusion_matrix(total_label, total_pred)

    print(f"Total acc:{acc:.4f} uar:{uar:.4f} f1:{f1:.4f}")
    return acc, uar, f1, cm


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
    )
    args = parser.parse_args()
    test_miss = True
    test_base = False
    in_men = True
    total_cv = 1
    gpu_ids = [0]
    ckpt_path = args.ckpt_path
    print(os.listdir(ckpt_path))
    config = json.load(open(os.path.join(ckpt_path, "train_opt.conf")))
    opt = OptConfig()
    opt.load(config)
    if test_base:
        opt.dataset_mode = "multimodal"
    if test_miss:
        opt.dataset_mode = "cmu_mosei_miss"

    opt.gpu_ids = gpu_ids
    setattr(opt, "in_mem", in_men)
    opt.ckpt_path = ckpt_path
    model = create_model(opt)
    model.setup(opt)
    results = []
    for cv in range(1, 1 + total_cv):
        opt.cvNo = cv
        tst_dataloader = create_dataset_with_args(opt, set_name="tst")
        model.load_networks_cv(os.path.join(ckpt_path, str(cv)))
        model.eval()
        if test_base:
            acc, uar, f1, cm, data = eval_all(model, tst_dataloader)
        if test_miss:
            acc, uar, f1, cm, data = eval_miss(model, tst_dataloader)
        results.append(data)

    ## Save results
    ## need to average the results across all the cross-validation folds
    ## while accounting for the each condition in the results dict

    # Function to process a single dictionary
    def process_dict(d):
        rows = []
        for condition, metrics in d.items():
            if len(metrics) >= 3:  # Ensure we have at least 3 metrics
                rows.append(
                    {
                        "condition": condition,
                        "acc": metrics[0],
                        "uar": metrics[1],
                        "f1_weighted": metrics[2],
                    }
                )
        return rows

    # Process all dictionaries
    all_rows = []
    for result in results:
        all_rows.extend(process_dict(result))

    # Create DataFrame from processed data
    df = pd.DataFrame(all_rows)

    # Group by condition and calculate mean for each metric
    df_final = (
        df.groupby("condition")
        .agg({"acc": "mean", "uar": "mean", "f1_weighted": "mean"})
        .reset_index()
    )

    # Display the final result
    print(df_final)

    # Save the final result to a CSV file
    df_final.to_csv(os.path.join(args.ckpt_path, "results.csv"), index=False)
