import pandas as pd
from matplotlib import pyplot as plt
import os
from tqdm import tqdm as tqdm
import numpy as np

def bootstrap_median_and_confiance_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)


def read_comparison_parameter_csvs(csv_folder_path):
    df = pd.DataFrame(columns=["level","f_best","f","time","step","iteration","evaluation","env_name","max_steps","inner_quantity","inner_length","seed"])
    for csv_name in tqdm(os.listdir(csv_folder_path)):
        if ".txt" in csv_name:
            n_df = pd.read_csv(csv_folder_path+"/"+csv_name)
            env_name,max_steps,inner_quantity, inner_length, seed = csv_name.removesuffix(".txt").split("_")
            n_df["env_name"] = env_name
            n_df["max_steps"] = int(max_steps)
            n_df["inner_quantity"] = inner_quantity
            n_df["inner_length"] = inner_length
            n_df["seed"] = int(seed)
            df = pd.concat([df, n_df])
    return df

def plot_comparison_parameters(csv_folder_path, figpath, exp_name):
    df = read_comparison_parameter_csvs(csv_folder_path)
    max_steps = max(df["step"])
    n_seeds = len(df["seed"].unique())

    # df.iloc[df[df["steps"] < 450].groupby("seed")["score"].idxmax()]

    # # Check how many steps where computed.
    # plt.boxplot(df.groupby(["seed", "inner_quantity", "inner_length"])["step"].max())
    # plt.show()
    


    print(df.dtypes)
    # import code; code.interact(local=locals()) # start interactive shell for debugging

    # df = df[df["level"]==3]
    # df = df[df["inner_quantity"]=='0.5']
    # df = df[df["inner_length"]=='0.5']
    # for seed in range(5,19):
    #     # print(df[df["seed"]==seed])
    #     # print(df.query(f"seed == {seed}"))
    #     plt.plot(df.query(f"seed == {seed}")["step"], df[(df["seed"]==seed)]["f"])
    # plt.show()
    # df.reset_index()
    # print(df)
    # exit(1)
    

    i=-1
    for group_name, df_group in df.groupby(["inner_quantity", "inner_length"]):
        i+=1
        x = []
        y_mean = []
        y_lower = []
        y_upper = []

        marker = {
            "0.5":"o",
            "1.0":"",    
        }[group_name[0]] # inner_quantity

        linestyle = {
            "0.5":"--",
            "1.0":"-",    
        }[group_name[1]] # inner_length
        df_group = df_group.reset_index()


        color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][i]

        print(marker, linestyle, color)

        step_slices = 50
        for step in np.linspace(0, max_steps, step_slices):
            selected_indices = df_group[df_group["step"] < step].groupby("seed")["f"].idxmax()
            scores = np.array(df_group.iloc[selected_indices]["f"])
            if len(scores) < 0.75*n_seeds:
                continue
            x.append(step)
            mean, lower, upper = bootstrap_median_and_confiance_interval(scores)
            y_mean.append(mean)
            y_lower.append(lower)
            y_upper.append(upper)

        plt.plot(x, y_mean, color=color, linestyle=linestyle, marker=marker, markevery=1/5)
        plt.fill_between(x, y_lower, y_upper, alpha=0.1, color=color, linestyle=linestyle)

    plt.plot([],[],label="inner_quantity = 0.5", marker="o", color="black")
    plt.plot([],[],label="inner_length = 0.5", linestyle="--",color="black")
    plt.legend()
    plt.savefig(figpath + "/comparison_cutting_controller_budget.pdf")
    plt.ylim((10,11))
    plt.savefig(figpath + "/comparison_cutting_controller_budget_zoom.pdf")


