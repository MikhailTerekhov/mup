import argparse
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import wandb
import pandas as pd
import matplotlib.pyplot as plt


def get_tags(name):
    tags = name.split("_")
    dim_index = tags.index("width") + 1
    lr_index = tags.index("lr") + 1
    mup_index = dim_index + 1
    lr = tags[lr_index]
    dim = tags[dim_index]
    mup = tags[mup_index]
    return (mup, dim, lr)


def get_best_metric(run, metric_name, is_min):
    best = None

    def upd_best(new):
        nonlocal best
        if best is None:
            best = new
        elif is_min:
            best = min(best, new)
        else:
            best = max(best, new)

    hist = run.scan_history()
    for entry in hist:
        if metric_name not in entry or entry[metric_name] is None:
            continue
        upd_best(entry[metric_name])
    return best


def get_wandb_summary_logs(user_name, project_name):
    api = wandb.Api()
    runs = list(api.runs(f"{user_name}/{project_name}"))

    summary_list, config_list, name_list = [], [], []
    for run in tqdm(runs, desc="Scanning runs for summaries"):
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items()
             if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
    rows = []
    for sum, conf, name, run in tqdm(zip(summary_list, config_list, name_list, runs), total=len(runs),
                                     desc="Scanning runs for best"):
        best_val_loss = get_best_metric(run, "val_loss", True)
        best_train_loss = get_best_metric(run, "train_loss", True)
        row_dict = {}
        row_dict.update(sum)
        row_dict.update(conf)
        row_dict.update({"name": name})
        row_dict["best_val_loss"] = best_val_loss
        row_dict["best_train_loss"] = best_train_loss
        mup, dim, lr = get_tags(name)
        row_dict["muP"] = mup
        if row_dict["lr_scheduler"] is None:
            row_dict["muP"] += " no scheduler"
        row_dict["dim"] = dim
        row_dict["lr"] = lr

        rows.append(row_dict)
    runs_df = pd.DataFrame(rows)

    return runs_df


def get_wandb_history_logs(user_name, project_name):
    api = wandb.Api()
    runs = api.runs(f"{user_name}/{project_name}")
    history_dict = {}
    for run in runs:
        if run.name.startswith("new_"):
            history_dict[run.name] = run.history()
    return history_dict


def plot_coord_summaries(history_dict, steps=range(0, 10)):
    norm_cols = [col for col in next(iter(history_dict.values())).columns if "norm" in col]
    dfs = []
    lrs = []
    for run_name in history_dict:
        mup, dim, lr = get_tags(run_name)
        lr = float(lr)
        run_df = history_dict[run_name]
        run_df = run_df[run_df["_step"].isin(steps)].copy()
        run_df["width"] = dim
        run_df["muP"] = mup
        if "lr_scheduler" in run_df and run_df["lr_scheduler"].iloc[0] == "none":
            run_df["mup"] += "no scheduler"
        run_df["lr"] = lr
        lrs.append(lr)
        dfs.append(run_df)
    df = pd.concat(dfs)
    mups = df["muP"].unique()

    print(f"norm_cols: {norm_cols} lr: {lrs} mups: {mups}")
    fig, axs = plt.subplots(len(mups), len(norm_cols), figsize=(30, 30))

    lr_to_plot = sorted(list(set(lrs)))[len(set(lrs)) // 2]
    for i, col in enumerate(norm_cols):
        for j, mup in enumerate(mups):
            for k in steps:
                axs[j, i].set_title(col + " - " + mup)
                axs[j, i].set_xlabel("width")
                axs[j, i].set_ylabel("abs(out).mean()")
                data = df[(df["lr"] == lr_to_plot) & (df["muP"] == mup) & (df["_step"] == k)]
                if not data.empty:
                    data.plot(x="width", y=col, ax=axs[j, i], label=f"{k}", marker="o", logy=True)
        plt.tight_layout()


# todo mup tag is not boolean anymore -- fix plotting below

def plot_val_accuracy(wandb_df):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey='row')

    wandb_df["muP"], wandb_df["dim"] = zip(*wandb_df["name"].apply(get_tags))

    # Plot for UmuP
    for dim in wandb_df["dim"].unique():
        subset = wandb_df.query("muP==True and dim==@dim")
        subset.sort_values("lr").plot(x="lr", y="val_accuracy", logx=True, label=f"Dim {dim}", ax=axs[0])

    axs[0].set_title("muP")
    axs[0].set_xlabel("Learning Rate")
    axs[0].set_ylabel("Validation Accuracy")
    axs[0].legend()
    axs[0].set_xscale("log", base=2)  # Set log scale with base 2 for x-axis

    # Plot for Standard
    for dim in wandb_df["Dim"].unique():
        subset = wandb_df.query("muP==False and Dim==@dim")
        subset.sort_values("lr").plot(x="lr", y="val_accuracy", logx=True, label=f"width {dim}", ax=axs[1])

    axs[1].set_title("Standard")
    axs[1].set_xlabel("Learning Rate")
    axs[1].set_ylabel("Validation Accuracy")
    axs[1].legend()
    axs[1].set_xscale("log", base=2)  # Set log scale with base 2 for x-axis

    plt.tight_layout()


def plot_losses(wandb_df):
    mups = wandb_df["muP"].unique()
    mups = sorted(mups)
    mups[0], mups[1] = mups[1], mups[0]  # Swap the "no scheduler" to the front
    mups = mups[-1:] + mups[:-1]  # Put the std plots to front

    fig, axs = plt.subplots(4, len(mups), figsize=(30, 30), sharey='row')
    metrics = {
        "val_loss": "Last Validation Loss",
        "best_val_loss": "Best Validation Loss",
        "train_loss": "Last Train Loss",
        "best_train_loss": "Best Train Loss"
    }
    for i, mup in enumerate(mups):
        for dim in wandb_df["dim"].unique():
            subset = wandb_df.query("muP==@mup and dim==@dim")
            if not subset.empty:
                for j, (metric, title) in enumerate(metrics.items()):
                    subset.sort_values("lr").plot(x="lr", y=metric, logx=True, label=f"Width {dim}", ax=axs[j][i])

        for j, (metric, title) in enumerate(metrics.items()):
            axs[j][i].set_title(f"{mup}\n{title}")
            axs[j][i].set_xlabel("Learning Rate")
            axs[j][i].set_ylabel(title)
            axs[j][i].set_xscale("log", base=2)  # Set log scale with base 2 for x-axis
            axs[j][i].legend()

    plt.tight_layout()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_name", type=str, default="claire-labo")
    parser.add_argument('--plot_losses', action='store_true')
    args = parser.parse_args()

    if args.plot_losses:
        cache_name = "wandb_cache.csv"
        if Path(cache_name).exists():
            wandb_df = pd.read_csv(cache_name)
        else:
            wandb_df = get_wandb_summary_logs(args.user_name, "mup-transformer-training")
            wandb_df.to_csv(cache_name)
        plot_losses(wandb_df)
    else:
        history_dict = get_wandb_history_logs(args.user_name, "mup-transformer-coordcheck")
        # Coordinate check
        plot_coord_summaries(history_dict, steps=range(0, 10))

    plt.show()
