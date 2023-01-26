import pandas as pd
from matplotlib import pyplot as plt


def plot_first_iteration(df, figpath, envname):

    print(f"Plot fitness in first 30 iterations in ppo on {envname}...", end="") 
    fig, ax = plt.subplots(figsize=(4.5, 3))
    sub_df = df.query("evaluation == 0 & iteration < 30 & level == 0").groupby("iteration")["f"].plot()
    plt.yscale("symlog")
    plt.xlabel("step")
    plt.ylabel("cumulative reward")
    plt.title(f"ppo first 30 iterations {envname}")
    plt.tight_layout()
    plt.savefig(figpath + "/ppo_first_30_iterations.pdf")
    plt.close()
    print("done")




    print(f"Plot fitness first 4 evaluations in inner on {envname}...", end="")
    fig, ax = plt.subplots(figsize=(4.5, 3))
    grouped = df.query("level == 1 & f.notna()").groupby("evaluation")
    for key in grouped.groups.keys():
        plt.plot(grouped.get_group(key)["iteration"], grouped.get_group(key)["f"])
    plt.xlabel("iteration")
    plt.ylabel("objective value")
    plt.title("Evaluation of first four morphologies")
    plt.tight_layout()
    plt.savefig(figpath+"/first_4_evaluations.pdf")
    plt.close()
    print("done")
 

    print(f"Plot higher bound lower bound first 4 evaluations in inner on {envname}...", end="")
    fig, ax = plt.subplots(figsize=(4.5, 3))
    df_minmax_and_level = df.query("level == 0 & f.notna()").groupby("iteration").agg({'f':['min', 'max'],'evaluation': 'median'})
    df_minmax_and_level.columns = ["min", "max", "evaluation"]
    df_minmax_and_level.reset_index(inplace=True)
    grouped = df_minmax_and_level.groupby("evaluation")
    for key in grouped.groups.keys():
        ax.fill_between(grouped.get_group(key)["iteration"], grouped.get_group(key)["min"], grouped.get_group(key)["max"])
    plt.xlabel("iteration")
    plt.ylabel("Cumulative reward range")
    plt.suptitle("Evaluation of first four morphologies")
    ax.set_title("The lowest and highest cumulative reward observed in each iteration")
    ax.title.set_size(8)
    ax.title.set_color('grey')
    plt.tight_layout()
    plt.savefig(figpath+"/first_4_evaluations_bounds_cumulative_reward.pdf")
    plt.close()
    print("done")
