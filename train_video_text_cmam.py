from argparse import ArgumentParser
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from cmam_loss import CMAMLoss
from cmams import BimodalCMAM
from data import create_dataset_with_args
from models import create_model
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.utils.config import OptConfig
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def find_net_in_dir(dir_path, net_name):
    for file in os.listdir(dir_path):
        if net_name in file:
            return file
    return None


def eval(model, dataloader, cmam, missing="video"):
    model.eval()
    total_pred = []
    total_label = []
    total_miss_type = []

    with torch.no_grad():
        for _, data in enumerate(dataloader):  # inner loop within one epoch

            generated = cmam(
                data["V_feat"].float().to(device), data["L_feat"].float().to(device)
            )

            data["A_feat"] = generated
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.test(a_in=generated)

            miss_type = np.array(data["miss_type"])

            pred = model.pred.argmax(dim=1).detach().cpu().numpy()
            label = data["label"]
            total_pred.append(pred)
            total_label.append(label)
            total_miss_type.append(miss_type)
        total_miss_type = np.concatenate(total_miss_type)

        total_pred = np.concatenate(total_pred)
        total_label = np.concatenate(total_label)

        missing_results = {}
        for part_name in [
            "zvl",
        ]:
            part_index = np.where(total_miss_type == part_name)
            part_pred = total_pred[part_index]
            part_label = total_label[part_index]
            acc_part = accuracy_score(part_label, part_pred)
            uar_part = recall_score(part_label, part_pred, average="macro")
            f1_part = f1_score(part_label, part_pred, average="macro")
            f1_weighted_part = f1_score(part_label, part_pred, average="weighted")
            f1_micro_part = f1_score(part_label, part_pred, average="micro")
            recall_macro_part = recall_score(part_label, part_pred, average="macro")
            recall_weighted_part = recall_score(
                part_label, part_pred, average="weighted"
            )
            recall_micro_part = recall_score(part_label, part_pred, average="micro")
            precision_macro_part = precision_score(
                part_label, part_pred, average="macro"
            )
            precision_weighted_part = precision_score(
                part_label, part_pred, average="weighted"
            )
            precision_micro_part = precision_score(
                part_label, part_pred, average="micro"
            )

            print(
                f"{part_name}, acc:{acc_part:.4f}, {uar_part:.4f}, {f1_part:.4f}, {f1_weighted_part:.4f}"
            )
            missing_results[part_name] = [
                acc_part,
                uar_part,
                f1_part,
                f1_weighted_part,
                f1_micro_part,
                recall_macro_part,
                recall_weighted_part,
                recall_micro_part,
                precision_macro_part,
                precision_weighted_part,
                precision_micro_part,
            ]

        return (
            acc_part,
            uar_part,
            f1_part,
            f1_weighted_part,
            f1_micro_part,
            recall_macro_part,
            recall_weighted_part,
            recall_micro_part,
            precision_macro_part,
            precision_weighted_part,
            precision_micro_part,
        )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoints_path",
        type=str,
    )

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )
    parser.add_argument("--cosine_weight", type=float, default=0.75)
    parser.add_argument("--cls_weight", type=float, default=0.3)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--mae_weight", type=float, default=0.2)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--total_cv", type=int, default=10, help="Number of CVs to run")
    parser.add_argument("--save_metrics_to", default=None, type=str)
    parser.add_argument("--cmam_type", default="v1", type=str)
    parser.add_argument("--target_metric", default="loss", type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--use_pretrained_encoders", action="store_false", default=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    epochs = args.epochs
    checkpoints_path = args.checkpoints_path

    target_modality = "A_feat"
    target_net = "net_A"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for cv in range(1, 1 + args.total_cv):
        try:
            config = json.load(open(os.path.join(checkpoints_path, "train_opt.conf")))
        except Exception as e:
            print(f"Could not find train_opt.conf in {checkpoints_path} {e}")
            exit(1)
        opt = OptConfig()
        opt.load(config)

        trained_video_encoder = LSTMEncoder(
            opt.input_dim_v, opt.embd_size_v, opt.embd_method_v
        )

        trained_text_encoder = TextCNN(opt.input_dim_l, opt.embd_size_l)

        model_dir = os.path.join(checkpoints_path, str(cv))
        if args.use_pretrained_encoders:
            trained_video_encoder.load_state_dict(
                torch.load(os.path.join(model_dir, find_net_in_dir(model_dir, "net_V")))
            )

            trained_text_encoder.load_state_dict(
                torch.load(os.path.join(model_dir, find_net_in_dir(model_dir, "net_L")))
            )

        trained_video_encoder = trained_video_encoder.to(device)
        trained_text_encoder = trained_text_encoder.to(device)

        trained_target_encoder = LSTMEncoder(
            opt.input_dim_a, opt.embd_size_a, opt.embd_method_a
        )
        trained_target_encoder.load_state_dict(
            torch.load(os.path.join(model_dir, find_net_in_dir(model_dir, target_net)))
        )

        trained_target_encoder.to(device)

        cmam = (
            BimodalCMAM(
                trained_video_encoder,
                trained_text_encoder,
                opt.embd_size_v + opt.embd_size_l,
                output_dim=opt.embd_size_a,
            )
            .to(device)
            .train()
        )

        train_dataloader = create_dataset_with_args(opt, set_name="trn")
        val_dataloader = create_dataset_with_args(opt, set_name="val")

        criterion = CMAMLoss(
            cosine_weight=args.cosine_weight,
            mae_weight=args.mae_weight,
            mse_weight=args.mse_weight,
        )
        cls_criterion = torch.nn.CrossEntropyLoss()
        optimizer = Adam(cmam.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_size = int(len(train_dataloader) / opt.batch_size)
        validation_size = int(len(val_dataloader) / opt.batch_size)
        gpu_ids = [0]
        in_men = True
        opt.gpu_ids = gpu_ids
        setattr(opt, "in_mem", in_men)
        model = create_model(opt)
        model.setup(opt)

        opt.cvNo = cv
        old_dataset_mode = opt.dataset_mode
        opt.dataset_mode = "multimodal_miss"
        tst_dataloader = create_dataset_with_args(opt, set_name="tst")
        model.load_networks_cv(os.path.join(checkpoints_path, str(cv)))
        model.eval()

        best_validation_loss = np.inf
        best_validation_f1 = 0.0
        best_validation_acc = 0.0
        best_validation_uar = 0.0
        for epoch in range(epochs):
            train_loss = 0.0
            train_targets = []
            train_generated = []
            cmam = cmam.train()
            for _, data in enumerate(train_dataloader):
                optimizer.zero_grad()
                audio = data["A_feat"].float().to(device)
                text = data["L_feat"].float().to(device)
                video = data["V_feat"].float().to(device)

                with torch.no_grad():
                    target = trained_target_encoder(audio)

                generated = cmam(video, text)
                loss = criterion(generated, target)

                data["L_feat"] = generated

                model.set_input(
                    data
                )  # unpack data from dataset and apply preprocessing
                model.test(l_in=generated)

                pred_logits = model.logits
                loss["cls_loss"] = cls_criterion(pred_logits, data["label"].to(device))
                loss["total_loss"] *= args.recon_weight
                loss["total_loss"] += args.cls_weight * loss["cls_loss"]

                loss["total_loss"].backward()
                optimizer.step()

                train_loss += loss["total_loss"].item()
                train_targets.append(target.detach().cpu().numpy())
                train_generated.append(generated.detach().cpu().numpy())

            train_loss /= (
                len(train_dataloader) / opt.batch_size
            )  ## MMIN data loader is weird
            train_targets = np.concatenate(train_targets)
            train_generated = np.concatenate(train_generated)
            train_mae = np.mean(np.abs(train_targets - train_generated))
            train_mse = np.mean((train_targets - train_generated) ** 2)
            del train_targets, train_generated
            val_loss = 0.0
            val_targets = []
            val_generated = []
            cmam = cmam.eval()
            val_losses = {}
            for _, data in enumerate(val_dataloader):
                audio = data["A_feat"].float().to(device)
                text = data["L_feat"].float().to(device)
                video = data["V_feat"].float().to(device)
                with torch.no_grad():
                    target = trained_target_encoder(audio)

                    generated = cmam(video, text)
                    loss = criterion(generated, target)
                    data["L_feat"] = generated
                    model.set_input(
                        data
                    )  # unpack data from dataset and apply preprocessing
                    model.test(l_in=generated)

                pred_logits = model.logits
                loss["cls_loss"] = cls_criterion(pred_logits, data["label"].to(device))
                loss["total_loss"] *= args.recon_weight

                loss["total_loss"] += args.cls_weight * loss["cls_loss"]
                val_loss += loss["total_loss"].item()
                val_targets.append(target.detach().cpu().numpy())
                val_generated.append(generated.detach().cpu().numpy())
                for k, v in loss.items():
                    if k not in val_losses:
                        val_losses[k] = []
                    val_losses[k].append(v.item())

            val_loss /= len(val_dataloader) / opt.batch_size
            val_losses = {k: np.mean(v) for k, v in val_losses.items()}
            val_targets = np.concatenate(val_targets)
            val_generated = np.concatenate(val_generated)
            val_mae = np.mean(np.abs(val_targets - val_generated))
            val_mse = np.mean((val_targets - val_generated) ** 2)

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                torch.save(
                    cmam.state_dict(),
                    os.path.join(model_dir, "cmam_VT_A_loss.pt"),
                )

                print(f"Best Loss model saved at {model_dir}")

            out_str = "Epoch {}/{}\nTrain Losses:".format(epoch + 1, epochs)
            for k, v in loss.items():
                out_str += f" {k}: {v:.6f} "
            out_str += "\nVal Losses:"
            for k, v in val_losses.items():
                out_str += f" {k}: {v:.6f} "

            print(out_str)
            (
                acc,
                uar,
                f1,
                f1_weighted,
                f1_micro,
                recall_macro,
                recall_weighted,
                recall_micro,
                precision_macro,
                precision_weighted,
                precision_micro,
            ) = eval(
                model,
                val_dataloader,
                cmam,
                missing="text" if target_modality == "V_feat" else "video",
            )

            if f1 > best_validation_f1:
                best_validation_f1 = f1
                torch.save(
                    cmam.state_dict(),
                    os.path.join(model_dir, "cmam_cmam_VT_A_f1.pt"),
                )
                print(f"Best F1 model saved at {model_dir}")

            if acc > best_validation_acc:
                best_validation_acc = acc
                torch.save(
                    cmam.state_dict(),
                    os.path.join(model_dir, "cmam_cmam_VT_A_acc.pt"),
                )
                print(f"Best ACC model saved at {model_dir}")

            if uar > best_validation_uar:
                best_validation_uar = uar
                torch.save(
                    cmam.state_dict(),
                    os.path.join(model_dir, "cmam_VT_A_uar.pt"),
                )
                print(f"Best UAR model saved at {model_dir}")

            print("best_validation_loss ", best_validation_loss)
            print("best_validation_f1 ", best_validation_f1)
            print("best_validation_acc ", best_validation_acc)
            print("best_validation_uar ", best_validation_uar)

            print("\n\n")

        print("Done")
        (
            acc_part,
            uar_part,
            f1_part,
            f1_weighted_part,
            f1_micro_part,
            recall_macro_part,
            recall_weighted_part,
            recall_micro_part,
            precision_macro_part,
            precision_weighted_part,
            precision_micro_part,
        ) = eval(
            model,
            tst_dataloader,
            cmam,
            missing="text" if target_modality == "V_feat" else "video",
        )
        torch.save(
            cmam.state_dict(),
            os.path.join(model_dir, "cmam_VT_A_final.pt"),
        )
        metrics = {
            "accuracy": acc_part,
            "uar": uar_part,
            "f1": f1_part,
            "f1_weighted": f1_weighted_part,
            "f1_micro": f1_micro_part,
            "recall_macro": recall_macro_part,
            "recall_weighted": recall_weighted_part,
            "recall_micro": recall_micro_part,
            "precision_macro": precision_macro_part,
            "precision_weighted": precision_weighted_part,
            "precision_micro": precision_micro_part,
        }
        metrics = {
            k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()
        }
        if args.save_metrics_to is not None:
            os.makedirs(os.path.dirname(args.save_metrics_to), exist_ok=True)
            if os.path.exists(args.save_metrics_to + ".csv"):
                df = pd.read_csv(args.save_metrics_to + ".csv")
            else:
                df = pd.DataFrame()

            df = pd.concat([df, pd.DataFrame(metrics, index=[cv])])
            df.to_csv(args.save_metrics_to + ".csv", index=False)
