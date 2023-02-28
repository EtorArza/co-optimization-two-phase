import pandas as pd
from matplotlib import pyplot as plt
import os
from tqdm import tqdm as tqdm
import numpy as np

marker_list = ["","o","x"]
linestyle_list = ["-","--","-."]
color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def bootstrap_median_and_confiance_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)


def read_comparison_parameter_csvs(csv_folder_path):
    print("Reading Dataframes...")
    dtypes={
        "level":object,
        "evaluation": np.int64,
        "f_best": np.float64,
        "f": np.float64,
        "controller_size": np.float64,
        "controller_size2": np.float64,
        "morphology_size": np.float64,
        "time": np.float64,
        "time_including_reeval": np.float64,
        "step": np.int64,
        "step_including_reeval": np.int64,
        "experiment_index": object,
        "env_name": object,
        "max_steps": int,
        "inner_quantity": object,
        "inner_length": object,
        "seed": int,
    }


    df = pd.DataFrame(columns=["level","evaluation","f_best","f","controller_size","controller_size2","morphology_size","time","time_including_reeval","step","step_including_reeval","experiment_index","env_name","max_steps","inner_quantity","inner_length","seed"])
    df = df.astype(dtype=dtypes)
    for csv_name in tqdm(os.listdir(csv_folder_path)):
        if ".txt" in csv_name:

            n_df = pd.read_csv(csv_folder_path+"/"+csv_name, header=0, dtype=dtypes)
            experiment_index,env_name,max_steps,inner_quantity, inner_length, seed = csv_name.removesuffix(".txt").split("_")
            n_df["experiment_index"] = experiment_index
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


    # Check how many steps where computed.
    indices_with_highest_step = np.array(df.groupby(by="experiment_index")["step"].idxmax())
    max_only_df = df.loc[indices_with_highest_step,]

    ax = max_only_df.boxplot(by=["inner_quantity", "inner_length"], column="step")

    ax.set_title("")
    ax.set_ylabel("step")
    plt.savefig(figpath + "/number_of_steps.pdf")
    plt.close()
    

    # # Print all seeds with certain parameters.
    # df = df[df["level"]==3]
    # df = df[df["inner_quantity"]=='0.5']
    # df = df[df["inner_length"]=='0.5']
    # for seed in range(2,19):
    #     # print(df[df["seed"]==seed])
    #     # print(df.query(f"seed == {seed}"))
    #     plt.plot(df.query(f"seed == {seed}")["step"], df[(df["seed"]==seed)]["f"], label=seed)
    # plt.legend()
    # plt.show()
    # exit(1)
    
    # Get the parameter values such that 1.0 is the first
    inner_quantity_values = sorted(list(df["inner_quantity"].unique()), key=lambda x: -4*float(x) + 2*float(x)*float(x))
    inner_length_values = sorted(list(df["inner_length"].unique()), key=lambda x: -4*float(x) + 2*float(x)*float(x))



    print("Generating plot...")
    step_slices = 30
    i=-1
    for group_name, df_group in tqdm(df.groupby(["inner_quantity", "inner_length"])):
        i+=1

        x = []
        y_mean = []
        y_lower = []
        y_upper = []

        marker = marker_list[inner_quantity_values.index(group_name[0])]
        linestyle = linestyle_list[inner_quantity_values.index(group_name[1])]
        color = color_list[i]

        df_group = df_group.reset_index()



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

    for inner_quantity, marker in zip(inner_quantity_values[1:], marker_list[1:]):
        plt.plot([],[],label=f"inner_quantity = {inner_quantity}", marker=marker, color="black")

    for inner_length, linestyle in zip(inner_length_values[1:], linestyle_list[1:]):

        plt.plot([],[],label=f"inner_length = {inner_length}", linestyle=linestyle,color="black")
    plt.legend()
    plt.savefig(figpath + "/comparison_cutting_controller_budget.pdf")
    plt.ylim((10,11))
    plt.savefig(figpath + "/comparison_cutting_controller_budget_zoom.pdf")


