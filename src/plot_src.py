import pandas as pd
from matplotlib import pyplot as plt
import os
from tqdm import tqdm as tqdm
import numpy as np

# copy figures / plots to paper dir with rsync -> 
# rsync -zarv --delete --prune-empty-dirs --include "*/"  --include="*.pdf" --exclude="*" "results" "../paper/results"


def removesuffix(txt:str, suffix:str)-> str: 
    if len(txt) >= len(suffix) and txt[-len(suffix):] == suffix:
        return txt[:-len(suffix)]
    else:
        return txt


def translate_to_plot_labels(param, experiment_name):

    assert experiment_name in ("reevaleachvsend", "adaptstepspermorphology")

    if param == "innerquantity_or_targetprob":
        if experiment_name == "reevaleachvsend":
            return "Controllers per morphology (wr to default params)"
        elif experiment_name == "adaptstepspermorphology":
            return "Target probability that best candidate is still best after reeval"
        else:
            raise ValueError("experiment_name=",experiment_name,"not recognized.")

    elif param == "innerlength_or_startquantity":
        if experiment_name == "reevaleachvsend":
            return "Episode length (wr to default params)"
        elif experiment_name == "adaptstepspermorphology":
            return "Initial and minimum 'Controllers per morphology' param"
        else:
            raise ValueError("experiment_name=",experiment_name,"not recognized.")
    else:
        raise ValueError(f"param={param} not found.")


marker_list = ["","o","x","s","d","2","^","*"]
linestyle_list = ["-","--","-.", ":",(0, (3, 5, 1, 5, 1, 5)),(5, (10, 3)), (0, (3, 1, 1, 1))]
color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def bootstrap_median_and_confiance_interval(data,bootstrap_iterations=2000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)


def read_comparison_parameter_csvs(csv_folder_path):

    # assert resumable_dimension in ("length", "quantity", "neither", None) 

    """
    What is the parameter reusable_dimension?

    Computation time can be saved in special cases when either innerlength_or_startquantity or innerquantity_or_targetprob are 1.0
    For example, imagine the learning algorithm PPO usually runs for 1000 iterations. If we train set 
    innerquantity_or_targetprob = 0.5 during training, it will only train for 500 iterations. If we have considered 
    the default episode length (with innerlength_or_startquantity = 1.0), then, when reevaluating, we only need to finish
    the next 500 iterations. In this case, the resumable dimension would be quantity.  

    quantity -> number of controllers tested
    length   -> the length of the episode

    With the current frameworks, the resumable dimensions are as follows:

    evogym      -> resumable_dimension = quantity
    RoboGrammar -> resumable_dimension = length

    When resumable == "neither", only innerlength_or_startquantity == innerquantity_or_targetprob == 1.0 gets a bonus of not having to
    reevaluate at all.
    """

    print(f"Reading csv_folder_path...")
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
        "innerquantity_or_targetprob": object,
        "innerlength_or_startquantity": object,
        "seed": int,
    }


    df = pd.DataFrame(columns=["env_name","experiment_index","experiment_name","level","evaluation","f_best","f","controller_size","controller_size2","morphology_size","time","time_including_reeval","step","step_including_reeval","innerquantity_or_targetprob","innerlength_or_startquantity","seed"])
    df = df.astype(dtype=dtypes)
    for csv_name in tqdm(os.listdir(csv_folder_path)):
        if ".txt" in csv_name:

            # omit_which = "0.5"
            # if omit_which in csv_name:
            #     continue


            n_df = pd.read_csv(csv_folder_path+"/"+csv_name, header=0, dtype=dtypes)
            experiment_name,experiment_index,env_name,innerquantity_or_targetprob, innerlength_or_startquantity, seed = removesuffix(csv_name,".txt").split("_")
            # if innerlength_or_startquantity != '1.0':
            #     continue
            n_df["experiment_name"] = experiment_name
            n_df["experiment_index"] = experiment_index
            n_df["env_name"] = env_name
            n_df["innerquantity_or_targetprob"] = float(innerquantity_or_targetprob)
            n_df["innerlength_or_startquantity"] = float(innerlength_or_startquantity)
            n_df["seed"] = int(seed)
            df = pd.concat([df, n_df], ignore_index=True)
    
    # # Adjust runtimes based on resumable dimension:
    # if resumable_dimension == 'quantity':
    #     sub_df = df.query("innerlength_or_startquantity == 1.0 & level == '3'")
    #     sub_df["step_including_reeval"] = sub_df["step_including_reeval"] - (sub_df["step_including_reeval"] - sub_df["step"]) * sub_df["innerquantity_or_targetprob"]
    #     cols = list(df.columns) 
    #     df.loc[sub_df.index, cols] = sub_df[cols]
    # elif resumable_dimension == 'length':
    #     sub_df = df.query("innerquantity_or_targetprob == 1.0 & level == '3'")
    #     sub_df["step_including_reeval"] = sub_df["step_including_reeval"] - (sub_df["step_including_reeval"] - sub_df["step"]) * sub_df["innerlength_or_startquantity"]
    #     cols = list(df.columns) 
    #     df.loc[sub_df.index, cols] = sub_df[cols]
    # elif resumable_dimension == 'neither':
    #     sub_df = df.query("innerquantity_or_targetprob == 1.0 & innerquantity_or_targetprob == 1.0 & level == '3'")
    #     sub_df["step_including_reeval"] = sub_df["step"]
    #     cols = list(df.columns) 
    #     df.loc[sub_df.index, cols] = sub_df[cols]

    return df


def _plot_performance(plotname, df: pd.DataFrame, figpath, scorelevel, param, score_label, resources):

    print(param)
    assert param in ["innerquantity_or_targetprob", "innerlength_or_startquantity"]
    assert scorelevel in ["reeval", "no_reeval"]


    max_steps = max(df["step"])
    n_seeds = len(df["seed"].unique())

    # Check how many steps where computed.
    indices_with_highest_step = np.array(df.groupby(by="experiment_index")["step"].idxmax())
    max_only_df = df.loc[indices_with_highest_step,]

    ax = max_only_df.boxplot(by=[param], column="step")

    ax.set_title("")
    ax.set_ylabel("step")
    plt.savefig(figpath + f"/number_of_steps_{param}.pdf")
    plt.close()
    
    if scorelevel == 'reeval':
        df = df.query("level == '3'")
    elif scorelevel == 'no_reeval':
        df = df.query("level == '2'")
    else:
        raise ValueError(f"scorelevel {scorelevel} not valid.")

    # Remove all ocurrences in which param_equal_1 parameter is not 1.0
    param_equal_1 = ["innerquantity_or_targetprob", "innerlength_or_startquantity"]
    param_equal_1.remove(param)
    param_equal_1 = param_equal_1[0]
    df = df[df[param_equal_1]==1.0]

    # # Print all seeds with certain parameters.
    # df = df[df["level"]=="3"]
    # df = df[df["innerquantity_or_targetprob"]==0.5]
    # df = df[df["innerlength_or_startquantity"]==0.5]
    # for seed in range(2,19):
    #     # print(df[df["seed"]==seed])
    #     # print(df.query(f"seed == {seed}"))
    #     plt.plot(df.query(f"seed == {seed}")["step"], df[(df["seed"]==seed)]["f"], label=seed)
    # plt.legend()
    # plt.show()
    # exit(1)
    
    # Get the parameter values such that 1.0 is the first
    innerquantity_or_targetprob_values = sorted(list(df["innerquantity_or_targetprob"].unique()), key=lambda x: -4*float(x) + 2*float(x)*float(x))
    innerlength_or_startquantity_values = sorted(list(df["innerlength_or_startquantity"].unique()), key=lambda x: -4*float(x) + 2*float(x)*float(x))

    step_slices = 100
    i=-1
    for group_name, df_group in tqdm(df.groupby(["innerquantity_or_targetprob", "innerlength_or_startquantity"])):
        i+=1

        x = []
        y_mean = []
        y_lower = []
        y_upper = []


        marker = marker_list[innerquantity_or_targetprob_values.index(group_name[0])]
        linestyle = linestyle_list[innerlength_or_startquantity_values.index(group_name[1])]
        color = color_list[i]

        df_group = df_group.reset_index()

        # base = 10
        # start = 0
        # end = np.log10(max_steps)
        # for step in np.logspace(start=start, stop=end, base=base, num=step_slices):


        for step in np.linspace(0, max_steps, step_slices):
            step = int(step)
            selected_indices = df_group[df_group[resources] < step].groupby("seed")[resources].idxmax()
            scores = np.array(df_group.loc[selected_indices,][score_label])
            if len(scores) < 0.75*n_seeds:
                continue
            x.append(step)
            mean, lower, upper = bootstrap_median_and_confiance_interval(scores)
            y_mean.append(mean)
            y_lower.append(lower)
            y_upper.append(upper)

        plt.plot(x, y_mean, color=color, linestyle=linestyle, marker=marker, markevery=1/5)
        plt.fill_between(x, y_lower, y_upper, alpha=0.1, color=color, linestyle=linestyle)

    for innerquantity_or_targetprob, marker in zip(innerquantity_or_targetprob_values[1:], marker_list[1:]):
        plt.plot([],[],label=f"innerquantity_or_targetprob = {innerquantity_or_targetprob}", marker=marker, color="black")

    for innerlength_or_startquantity, linestyle in zip(innerlength_or_startquantity_values[1:], linestyle_list[1:]):
        plt.plot([],[],label=f"innerlength_or_startquantity = {innerlength_or_startquantity}", linestyle=linestyle,color="black")
    # plt.xscale("log")
    plt.legend()
    plt.savefig(figpath + f"/performance_{plotname}.pdf")
    plt.close()


def _plot_stability(df: pd.DataFrame, figpath):

    for fitness_metric in ["f", "f_best"]:

        max_steps = max(df["step"])
        n_seeds = len(df["seed"].unique())

        df = df.query("level == '3'")
        pd.pandas.set_option('display.max_columns', None)

        print("TODO: The animation and the objective value might not be the same!")

        indices_with_highest_step = np.array(df.groupby(by="experiment_index")["step"].idxmax())
        max_only_df = df.loc[indices_with_highest_step,]

        labels = []
        dfs = []
        index_of_lowest_each_group = []
        for group_name, df_group in sorted(list(max_only_df.groupby(["innerquantity_or_targetprob", "innerlength_or_startquantity"])), key=lambda x: (x[0][1], x[0][0])):
            labels.append(group_name)
            dfs.append(np.array(df_group[fitness_metric]))
            index_of_lowest_each_group.append(df_group[fitness_metric].idxmin())

        
        plt.figure(figsize=(6.5,3))
        plt.boxplot(dfs, labels=labels)

        # Add labels to outliers.
        for x_text, idx in zip(plt.xticks()[0], index_of_lowest_each_group):
            y_text = max_only_df.loc[idx, fitness_metric]
            text_label = max_only_df.loc[idx, "experiment_index"]
            plt.text(x_text, y_text, text_label)

        # for group_name, df_group in sorted(list(max_only_df.groupby(["innerquantity_or_targetprob", "innerlength_or_startquantity"])), key=lambda x: (x[0][1], x[0][0])):
        #     plt.text()

        plt.xlabel("(innerquantity_or_targetprob, innerlength_or_startquantity)")
        plt.title("")
        plt.ylim((0,11))
        plt.ylabel("objective value")
        plt.tight_layout()
        plt.savefig(figpath + f"/final_fitness_{fitness_metric}.pdf")
        plt.close()
    

def _plot_complexity(df: pd.DataFrame, figpath, complexity_metric):
    fitness_metric = "f_best"


    df = df.query("level == '3'")
    pd.pandas.set_option('display.max_columns', None)

    indices_with_highest_step = np.array(df.groupby(by="experiment_index")["step"].idxmax())
    max_only_df = df.loc[indices_with_highest_step,]

    labels = []
    dfs = []
    index_of_lowest_each_group = []
    for group_name, df_group in sorted(list(max_only_df.groupby(["innerquantity_or_targetprob", "innerlength_or_startquantity"])), key=lambda x: (x[0][1], x[0][0])):
        labels.append(group_name)
        dfs.append(np.array(df_group[complexity_metric]))



    
    plt.figure(figsize=(6.5,3))
    plt.boxplot(dfs, labels=labels)    
    plt.xlabel("(innerquantity_or_targetprob, innerlength_or_startquantity)")
    plt.title("")
    plt.ylabel(f"sol. complexity as {complexity_metric}")
    plt.tight_layout()
    plt.savefig(figpath + f"/solution_complexity_{complexity_metric}.pdf")
    plt.close()


def plot_comparison_parameters(csv_folder_path, figpath):
    df = read_comparison_parameter_csvs(csv_folder_path)

    for param, param_preffix in zip(["innerquantity_or_targetprob", "innerlength_or_startquantity"], ["quantity", "length"]):
        _plot_performance(f"{param_preffix}_reevalend",      df.copy(), figpath, "reeval",    param, "f", "step")
        _plot_performance(f"{param_preffix}_reevalbest",     df.copy(), figpath, "reeval",    param, "f_best", "step_including_reeval")
        _plot_performance(f"{param_preffix}_noreeval",       df.copy(), figpath, "no_reeval", param, "f_best", "step")
        _plot_performance(f"{param_preffix}_controllersize", df.copy(), figpath, "reeval",    param, "controller_size", "step_including_reeval")
        _plot_performance(f"{param_preffix}_controllersize2",df.copy(), figpath, "reeval",    param, "controller_size2", "step_including_reeval")
        _plot_performance(f"{param_preffix}_morphologysize", df.copy(), figpath, "reeval",    param, "morphology_size", "step_including_reeval")

    _plot_stability(df.copy(), figpath)
    for complexity_metric in ["controller_size", "controller_size2", "morphology_size"]:
        _plot_complexity(df.copy(), figpath, complexity_metric)



    



