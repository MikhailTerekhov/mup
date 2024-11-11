import pandas as pd 
import wandb
import pandas as pd
import matplotlib.pyplot as plt


def get_wandb_summary_logs(user_name,project_name):
    api = wandb.Api()
    runs = api.runs(f"{user_name}/{project_name}")

    summary_list, config_list, name_list = [], [], []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
    rows = []
    for sum,conf,name in zip(summary_list,config_list,name_list):
        row_dict = {}
        row_dict.update(sum)
        row_dict.update(conf)
        row_dict.update({"name":name})
        rows.append(row_dict)
    runs_df = pd.DataFrame(rows)

    return runs_df

def get_wandb_history_logs(user_name,project_name):
    api = wandb.Api()
    runs = api.runs(f"{user_name}/{project_name}")
    history_dict = {}
    for run in runs:
        history_dict[run.name] = run.history()
    return history_dict

def get_tags(name):
    tags = name.split("_")
    dim_index = tags.index("width")+1
    lr_index = tags.index("lr")+1
    mup_index = dim_index+1
    lr=tags[lr_index]
    dim=tags[dim_index]
    mup=tags[mup_index]
    return (mup=="muP", dim, lr)

def plot_coord_summaries(history_dict, steps=range(0,10)):
    norm_cols = [col for col in next(iter(history_dict.values())).columns if "norm" in col]
    fig,axs = plt.subplots(2,len(norm_cols), figsize=(30,10))
    dfs=[]
    lrs = []
    for run_name in history_dict:
        mup, dim, lr = get_tags(run_name)
        lr = float(lr)
        run_df = history_dict[run_name]
        run_df = run_df[run_df["_step"].isin(steps)].copy()
        run_df["width"] = dim
        run_df["muP"] = mup
        run_df["lr"] = lr
        lrs.append(lr)
        dfs.append(run_df)
    df = pd.concat(dfs)
    lr_to_plot = sorted(list(set(lrs)))[len(set(lrs))//2]
    for i, col in enumerate(norm_cols):
        for j, mup in enumerate([True,False]):
            for k in steps:
                axs[j,i].set_title(col + " - " + ("muP" if mup else "standard"))
                axs[j,i].set_xlabel("width")
                axs[j,i].set_ylabel("abs(out).mean()")
                df[(df["lr"]==lr_to_plot)&(df["muP"]==mup) & (df["_step"]==k) ].plot(x="width", y=col, ax=axs[j,i], label=f"{k}", marker="o", logy=True)
        plt.tight_layout()

def plot_val_accuracy(wandb_df):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

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
    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    wandb_df["muP"], wandb_df["dim"] = zip(*wandb_df["name"].apply(get_tags))
    # Plot for UmuP
    for dim in wandb_df["dim"].unique():
        subset = wandb_df.query("muP==True and dim==@dim")
        subset.sort_values("lr").plot(x="lr", y="val_loss", logx=True, label=f"Width {dim}", ax=axs[0][0])

    axs[0][0].set_title("muP - Validation Loss")
    axs[0][0].set_xlabel("Learning Rate")
    axs[0][0].set_ylabel("Validation Loss")
    #axs[0][0].set_ylim(4, 13)
    axs[0][0].set_xscale("log", base=2)  # Set log scale with base 2 for x-axis
    axs[0][0].legend()

    # Plot for Standard
    for dim in wandb_df["dim"].unique():
        subset = wandb_df.query("muP==False and dim==@dim")
        subset.sort_values("lr").plot(x="lr", y="val_loss", logx=True, label=f"Width {dim}", ax=axs[0][1])

    axs[0][1].set_title("Standard - Validation Loss")
    axs[0][1].set_xlabel("Learning Rate")
    axs[0][1].set_ylabel("Validation Loss")
    axs[0][1].set_xscale("log", base=2)  # Set log scale with base 2 for x-axis
    #axs[0][1].set_ylim(4, 13)
    axs[0][1].legend()

    # Plot for UmuP - Train Loss
    for dim in wandb_df["dim"].unique():
        subset = wandb_df.query("muP==True and dim==@dim")
        subset.sort_values("lr").plot(x="lr", y="train_loss", logx=True, label=f"Width {dim}", ax=axs[1][0])

    axs[1][0].set_title("muP - Train Loss")
    axs[1][0].set_xlabel("Learning Rate")
    axs[1][0].set_ylabel("Train Loss")
    axs[1][0].set_xscale("log", base=2)  # Set log scale with base 2 for x-axis
    #axs[1][0].set_ylim(4, 13)
    axs[1][0].legend()

    # Plot for Standard - Train Loss
    for dim in wandb_df["dim"].unique():
        subset = wandb_df.query("muP==False and dim==@dim")
        subset.sort_values("lr").plot(x="lr", y="train_loss", logx=True, label=f"Width {dim}", ax=axs[1][1])

    axs[1][1].set_title("Standard - Train Loss")
    axs[1][1].set_xlabel("Learning Rate")
    axs[1][1].set_ylabel("Train Loss")
    axs[1][1].set_xscale("log", base=2)  # Set log scale with base 2 for x-axis

    #axs[1][1].set_ylim(4, 13)
    axs[1][1].legend()
    plt.tight_layout()