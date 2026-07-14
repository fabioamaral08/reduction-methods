import sys
import os
sys.path.append(os.path.join(os.path.split(sys.path[0])[0], 'src'))

import torch
from torch.utils.data import DataLoader, Dataset
import glob
from utils import *
from Autoencoder.SINDy_Autoencoder import SINDyAutoencoderModule, loss_sindy_ae
import argparse
import optuna
from optuna import TrialPruned

# dataset that loads saved .pt snapshots
class SnapshotDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = sorted(glob.glob(os.path.join(folder, "B_*.pt")))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        A = torch.load(self.files[i]).float()
        dA = torch.load(os.path.join(self.folder, f"dB_dt_{i:05d}.pt")).float()
        return A, dA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_bool(value):
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def build_loaders(train_dir, val_dir, batch_size):
    train_ds = SnapshotDataset(train_dir)
    val_ds = SnapshotDataset(val_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_ds, val_ds, train_loader, val_loader


def train_epoch(model, loader, optimizer, lambda1, lambda2, lambda3, rec_energy, L2):
    model.train()
    total = 0.0
    for A_batch, dA_batch in loader:
        A_batch = A_batch.to(device)
        dA_batch = dA_batch.to(device)
        A_batch.requires_grad_(True)
        loss = loss_sindy_ae(A_batch, model, dA_batch, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, rec_energy=rec_energy, L2=L2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * A_batch.size(0)
    return total / len(loader.dataset)


def eval_epoch(model, loader, lambda1, lambda2, lambda3, rec_energy, L2):
    model.eval()
    total = 0.0
    with torch.enable_grad():
        for A_batch, dA_batch in loader:
            A_batch = A_batch.to(device)
            dA_batch = dA_batch.to(device)
            A_batch.requires_grad_(True)
            loss = loss_sindy_ae(A_batch, model, dA_batch, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, rec_energy=rec_energy, L2=L2)
            total += loss.item() * A_batch.size(0)
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--rec_energy", type=parse_bool, required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--eps_sindy", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--epoch_threshold", type=int, default=500)
    parser.add_argument("--epoch_ref", type=int, default=1000)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--L2", type=float, default=20)
    parser.add_argument("--include_bias", action="store_true")
    args = parser.parse_args()

    print(args.train_dir)
    sample_A, _ = SnapshotDataset(args.train_dir)[0]
    channels, H, W = sample_A.shape
    print(sample_A.shape)

    save_dir = '../ModelsTorch/SINDy_AE'

    save_dir += '_Kernel' if args.rec_energy else '_MSE'
    L2 = args.L2 if args.rec_energy else None
    def objective(trial):
        n_filters = trial.suggest_int("n_filters", 8, 16, step=8)
        n_layers = trial.suggest_int("n_layers", 1, 4)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        lambda1 = trial.suggest_float("lambda1", 1e-3, 1, log=True)
        lambda2 = trial.suggest_float("lambda2", 1e-3, 1, log=True)
        lambda3 = trial.suggest_float("lambda3", 1e-6, 1e-1, log=True)

        model = SINDyAutoencoderModule(n_filters, n_layers, args.latent_dim, (channels, H, W), args.degree, args.include_bias).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        _, _, train_loader, val_loader = build_loaders(args.train_dir, args.val_dir, args.batch_size)

        val_loss = 0.0
        for ep in range(1, min(5, args.num_epochs) + 1):
            train_epoch(model, train_loader, optimizer, lambda1, lambda2, lambda3, args.rec_energy, L2)
            val_loss = eval_epoch(model, val_loader, lambda1, lambda2, lambda3, args.rec_energy, L2)
            trial.report(val_loss, ep)
            if trial.should_prune():
                raise TrialPruned()
        return val_loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)

    best = study.best_params
    model = SINDyAutoencoderModule(best["n_filters"], best["n_layers"], args.latent_dim, (channels, H, W), args.degree, args.include_bias).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best["lr"], weight_decay=best["weight_decay"])
    _, _, train_loader, val_loader = build_loaders(args.train_dir, args.val_dir, args.batch_size)

    best_val_loss = float("inf")
    best_checkpoint_path = save_dir + "/best_sindy_ae_checkpoint.pt"

    for ep in range(1, args.num_epochs + 1):
        train_epoch(model, train_loader, optimizer, best["lambda1"], best["lambda2"], best["lambda3"], args.rec_energy, L2)
        if ep % args.epoch_threshold == 0:
            model.sindy.threshold(eps= args.eps_sindy)
        current_val = eval_epoch(model, val_loader, best["lambda1"], best["lambda2"], best["lambda3"], args.rec_energy, L2)
        print(f"Epoch {ep}: val_loss={current_val:.6f}")
        if current_val < best_val_loss:
            best_val_loss = current_val
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_params": best,
                "latent_dim": args.latent_dim,
                "rec_energy": args.rec_energy,
                "L2": L2,
                "degree": args.degree,
                "include_bias": args.include_bias,
                "input_shape": (channels, H, W),
            }, best_checkpoint_path)


    # Coeff refinement
    for ep in range(1, args.epoch_ref + 1):
        train_epoch(model, train_loader, optimizer, best["lambda1"], best["lambda2"], 0.0, args.rec_energy, L2)
        current_val = eval_epoch(model, val_loader, best["lambda1"], best["lambda2"], 0.0, args.rec_energy, L2)
        print(f"Coeff Refinement Epoch {ep}: val_loss={current_val:.6f}")
        if current_val < best_val_loss:
            best_val_loss = current_val
            torch.save({
                "epoch": args.num_epochs + ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_params": best,
                "latent_dim": args.latent_dim,
                "rec_energy": args.rec_energy,
                "L2": L2,
                "degree": args.degree,
                "include_bias": args.include_bias,
                "input_shape": (channels, H, W),
            }, best_checkpoint_path)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    study.trials_dataframe().to_csv(save_dir + "optuna_sindy_ae_trials.csv", index=False)


if __name__ == '__main__':
    main()
