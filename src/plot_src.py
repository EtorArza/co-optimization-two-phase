import pandas as pd
pd.set_option('mode.chained_assignment','raise')

from matplotlib import pyplot as plt
import os
from tqdm import tqdm as tqdm
import numpy as np


stopping_criterion=int(8e6)

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

def bootstrap_mean_and_confiance_interval(data,bootstrap_iterations=2000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)


def read_comparison_parameter_csvs(csv_folder_path):


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
            if "param" in csv_name:
                continue
            if "proposedmethod" in csv_name:
                continue

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

            # import code; code.interact(local=locals()) # Start interactive shell for debug debugging
            # import tabloo; tabloo.show(n_df) # View pandas dataframe or table in browser

            df = pd.concat([df, n_df.copy()], ignore_index=True)


    # Check how many steps where computed.
    indices_with_highest_step = np.array(df.groupby(by="experiment_index")["step"].idxmax())
    max_only_df = df.loc[indices_with_highest_step,]
    acceptable_rows = max_only_df[max_only_df["step"] > np.mean(max_only_df["step"]) * 0.95]

    print("Number of rows per class: ", np.array(max_only_df.groupby(by=["innerquantity_or_targetprob", "innerlength_or_startquantity"]).count()["env_name"]))
    print("Number of rows per class after pruning crashed exp.: ", np.array(acceptable_rows.groupby(by=["innerquantity_or_targetprob", "innerlength_or_startquantity"]).count()["env_name"]))

    df = df[df["experiment_index"].isin(acceptable_rows["experiment_index"])]


    # # Adjust runtimes based on resumable dimension:
    # assert resumable_dimension in ("length", "quantity", "neither", None) 
    
    # """
    # What is the parameter reusable_dimension?

    # Computation time can be saved in special cases when either innerlength_or_startquantity or innerquantity_or_targetprob are 1.0
    # For example, imagine the learning algorithm PPO usually runs for 1000 iterations. If we train set 
    # innerquantity_or_targetprob = 0.5 during training, it will only train for 500 iterations. If we have considered 
    # the default episode length (with innerlength_or_startquantity = 1.0), then, when reevaluating, we only need to finish
    # the next 500 iterations. In this case, the resumable dimension would be quantity.  

    # quantity -> number of controllers tested
    # length   -> the length of the episode

    # With the current frameworks, the resumable dimensions are as follows:

    # evogym      -> resumable_dimension = quantity
    # RoboGrammar -> resumable_dimension = length

    # When resumable == "neither", only innerlength_or_startquantity == innerquantity_or_targetprob == 1.0 gets a bonus of not having to
    # reevaluate at all.
    # """
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


def _plot_probability_of_choosing_best_morphology(plotname, df:pd.DataFrame, figpath, param):
    from NestedOptimization import Parameters
    max_steps = Parameters("evogym",0).max_frames
    nseeds = Parameters("evogym",0).nseeds

    param_name = ["quantity", "length"][["innerquantity_or_targetprob", "innerlength_or_startquantity"].index(param)]

    df = df.query("level == '3'")


    # Remove all ocurrences in which param_equal_1 parameter is not 1.0
    param_equal_1 = ["innerquantity_or_targetprob", "innerlength_or_startquantity"]
    param_equal_1.remove(param)
    param_equal_1 = param_equal_1[0]


    param_values_list = list(sorted(df[param].unique(), reverse=True))
    for param_idx, param_value in enumerate(tqdm(param_values_list)):



        step_slices = 100
        x = np.linspace(0,max_steps,step_slices)
        y = np.ones((nseeds,step_slices))
        y *= np.nan
        for seed in range(2,2+nseeds):
            df_reeval_end   = df.query(f"step < {max_steps}                  and {param} == {param_value} and {param_equal_1} == 1.0 and level == '3' and seed == {seed}")

            best_correctly_identified = 0
            for i, step in enumerate(x):
                if df_reeval_end[df_reeval_end["step"] < step]["step"].size == 0:
                    continue
                idx_end = int(df_reeval_end[df_reeval_end["step"] < step]["step"].idxmax())
                row = df_reeval_end.loc[int(idx_end),][["step","f_best","f"]]
                y[(seed-2,i)] = int(row["f"] == row["f_best"])
        y = np.nan_to_num(y,copy=True,nan=0)
        y_mean, y_lower, y_upper = np.apply_along_axis(bootstrap_mean_and_confiance_interval, 0, y)
        
        marker = marker_list[param_idx]
        color = color_list[param_idx]

        plt.plot(x, y_mean, color=color, marker=marker, markevery=1/10, label=f"{param_value}")
        plt.fill_between(x, y_lower, y_upper, alpha=0.1, color=color)
    plt.axhline(0, color="black", linestyle="--")
    plt.ylabel("Probability of identifying best")
    plt.xlabel("steps")
    plt.legend(title=param_name)
    plt.title(plotname)
    plt.text(1,-3,"Explaination")
    plt.savefig(figpath + f"/{plotname}.pdf")
    plt.close()



# Compare reevaluating every best solution vs reevaluating at the end.
def _plot_performance_reeval_every_best_vs_end(plotname, df:pd.DataFrame, figpath, param):

    from NestedOptimization import Parameters
    max_steps = stopping_criterion
    nseeds = Parameters("evogym",0).nseeds

    param_name = ["quantity", "length"][["innerquantity_or_targetprob", "innerlength_or_startquantity"].index(param)]

    df3 = df.query("level == '3'")


    # Remove all ocurrences in which param_equal_1 parameter is not 1.0
    param_equal_1 = ["innerquantity_or_targetprob", "innerlength_or_startquantity"]
    param_equal_1.remove(param)
    param_equal_1 = param_equal_1[0]



    plt.figure(figsize=(4, 3))
    for param_idx, param_value in enumerate(sorted(df3[param].unique(), reverse=True)):



        step_slices = 100
        x = np.linspace(0,max_steps,step_slices)
        y = np.ones((nseeds,step_slices))
        y *= np.nan


        print("Add the runtime that reevaluating at the end takes, which is not otherwise taken into account.")
        for seed in range(2,2+nseeds):
            df_reeval_end   = df3.query(f"step < {max_steps}                  and {param} == {param_value} and {param_equal_1} == 1.0 and level == '3' and seed == {seed}")
            df_reeval_every = df3.query(f"step_including_reeval < {max_steps} and {param} == {param_value} and {param_equal_1} == 1.0 and level == '3' and seed == {seed}")

            if len(df.index) == 0 or len(df3.query(f"level == '3' and seed == {seed}").index) == 0:
                print("DataFrame has no rows")
                continue
 
            indx_first_lvl_3 = df3.query(f"level == '3' and seed == {seed}").index[0]


            difference = df.loc[indx_first_lvl_3, 'step_including_reeval'] - df.loc[indx_first_lvl_3 - 1, 'step_including_reeval']

            for i, step in enumerate(x):
                if df_reeval_end[df_reeval_end["step"] < step]["step"].size == 0:
                    continue
                if df_reeval_every[df_reeval_every["step_including_reeval"] < step]["step_including_reeval"].size == 0:
                    continue
                


                idx_end = df_reeval_end[df_reeval_end["step"] < step]["step"].idxmax()
                idx_every = df_reeval_every[df_reeval_every["step_including_reeval"] < step]["step_including_reeval"].idxmax()

                y[(seed-2,i)] = df_reeval_every.loc[idx_every,]["f_best"] - df_reeval_end.loc[idx_end,]["f"]

        y = np.nan_to_num(y,copy=True,nan=0)
        y_mean, y_lower, y_upper = np.apply_along_axis(bootstrap_mean_and_confiance_interval, 0, y)
        
        marker = marker_list[param_idx]
        color = color_list[param_idx]

        plt.plot(x, y_mean, color=color, marker=marker, markevery=1/10, label=f"{param_value}")
        plt.fill_between(x, y_lower, y_upper, alpha=0.1, color=color)
    plt.axhline(0, color="black", linestyle="--")
    plt.ylabel("- reevaluate_end              + reevaluate_every_new_best")
    plt.xlabel("steps")
    plt.legend(title=param_name)
    plt.title("Reeval every new best - reevaluate end")
    # plt.text(1,-3,"quantity = 1.0 is in the negative side, because with \n reevaluate every we waste time\n reevaluating with no benefit.")
    # plt.show()
    plt.tight_layout()
    plt.savefig(figpath + f"/{plotname}.pdf")
    plt.close()

def _plot_performance(plotname, df: pd.DataFrame, figpath, scorelevel, param, score_label, resources):

    print(param)
    assert param in ["innerquantity_or_targetprob", "innerlength_or_startquantity"]
    assert scorelevel in ["reeval", "no_reeval"]

    from NestedOptimization import Parameters
    max_steps = stopping_criterion
    nseeds = Parameters("evogym",0).nseeds


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

    
    # Get the parameter values such that 1.0 is the first
    innerquantity_or_targetprob_values = sorted(list(df["innerquantity_or_targetprob"].unique()), key=lambda x: -4*float(x) + 2*float(x)*float(x))
    innerlength_or_startquantity_values = sorted(list(df["innerlength_or_startquantity"].unique()), key=lambda x: -4*float(x) + 2*float(x)*float(x))

    param_values = sorted(list(set(innerlength_or_startquantity_values + innerlength_or_startquantity_values)), key=lambda x: -float(x))
    print("param_values", param_values)
    assert len(innerquantity_or_targetprob_values) == 1 or len(innerlength_or_startquantity_values) == 1

    legendtitle = {
        "innerquantity_or_targetprob":"quantity",
        "innerlength_or_startquantity":"length",
    }[param]

    step_slices = 100
    i=-1
    plt.figure(figsize=(4, 3))
    for group_name, df_group in tqdm(sorted(df.groupby(param), key=lambda x: -float(x[0]))):
        i+=1

        x = []
        y_mean = []
        y_lower = []
        y_upper = []


        marker = marker_list[i]
        linestyle = linestyle_list[i]
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
            if len(scores) < 0.75*nseeds:
                continue
            x.append(step)
            mean, lower, upper = bootstrap_mean_and_confiance_interval(scores)
            y_mean.append(mean)
            y_lower.append(lower)
            y_upper.append(upper)

        plt.plot(x, y_mean, color=color, linestyle=linestyle, marker=marker, markevery=1/10, label=group_name)
        plt.fill_between(x, y_lower, y_upper, alpha=0.1, color=color, linestyle=linestyle)

    # for innerquantity_or_targetprob, marker in zip(innerquantity_or_targetprob_values[1:], marker_list[1:]):
    #     plt.plot([],[],label=f"innerquantity_or_targetprob = {innerquantity_or_targetprob}", marker=marker, color="black")

    # for innerlength_or_startquantity, linestyle in zip(innerlength_or_startquantity_values[1:], linestyle_list[1:]):
    #     plt.plot([],[],label=f"innerlength_or_startquantity = {innerlength_or_startquantity}", linestyle=linestyle,color="black")
    # plt.xscale("log")
    plt.legend(title=legendtitle)
    plt.tight_layout()
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


def plot_tune(data_dir, fig_dir):
    import os
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt

    def find_between(s, start, end): # find substring between two strings
        return (s.split(start))[1].split(end)[0]


    rows = []
    for csv_name in os.listdir(data_dir):
        if ".txt" in csv_name and "paramtuning" in csv_name:
            df = pd.read_csv(data_dir + "/" + csv_name)
            df = df.query(f"step <= {stopping_criterion}") # set max steps to stopping_criterion
            f = df.query("level == 2")["f_best"].iloc[-1]
            step = df.query("level == 2")["step"].iloc[-1]
            nrows = df.query("level == 2").shape[0]
            innerquantity = int(find_between(csv_name, "paramtuning_","_"))
            seed = int(find_between(csv_name, "_",".txt"))
            rows.append([innerquantity, seed, f, nrows, step])
    df = pd.DataFrame(rows, columns=["innerquantity", "seed", "f", "nrows","step"])

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    inner_quantity_list = sorted(df["innerquantity"].unique())
    print("inner_quantity_list =", inner_quantity_list)

    # # https://stackoverflow.com/questions/43345599/process-pandas-dataframe-into-violinplot
    # fig, axes = plt.subplots()
    # axes.violinplot(dataset = [df[df.innerquantity == el]["f"].values for el in inner_quantity_list],showmedians=True)
    # axes.set_title('Day Ahead Market')
    # axes.yaxis.grid(True)
    # axes.set_xlabel('Scenario')
    # axes.set_ylabel('LMP ($/MWh)')
    # plt.show()
    # plt.close()

    def set_axis_style(ax, labels):
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Controllers evaluated per morphology')

    plt.figure(figsize=(4, 3))
    boxplot = plt.boxplot([df[df.innerquantity == el]["f"].values for el in inner_quantity_list], showmeans=True)

    # Add legend handles and labels
    legend_handles = [boxplot["medians"][0], boxplot["means"][0]]
    legend_labels = ["Median", "Mean"]
    plt.legend(legend_handles, legend_labels)

    set_axis_style(plt.gca(), [str(el) for el in inner_quantity_list])
    plt.title("")
    plt.tight_layout()

    plt.savefig(fig_dir+r"/f_tune.pdf")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.violinplot(dataset = [df[df.innerquantity == el]["nrows"].values for el in inner_quantity_list],showmedians=True)
    set_axis_style(plt.gca(), [str(el) for el in inner_quantity_list])
    plt.title("nrows")
    plt.yscale("log")
    plt.tight_layout()

    plt.savefig(fig_dir+r"/nrows_tune.pdf")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.violinplot(dataset = [df[df.innerquantity == el]["step"].values for el in inner_quantity_list],showmedians=True)
    set_axis_style(plt.gca(), [str(el) for el in inner_quantity_list])
    plt.title("step")
    plt.yscale("log")
    plt.tight_layout()

    plt.savefig(fig_dir+r"/step_tune.pdf")
    plt.close()


    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(4, 3))


    markers = ['o', 's', 'D', 'v', '^', 'p', 'P']
    line_styles = ['-', '--', '-.', ':']
    line_colors = ['black', 'gray', 'darkgray', 'lightgray']
    num_colors = len(inner_quantity_list)
    import matplotlib.cm as cm
    cmap = cm.get_cmap('Greys', num_colors+1)  # Linear colormap from light to dark
    marker_step = 10  # Set the step size for markers

    for idx, inner_quantity in enumerate(inner_quantity_list):
        data = df[df.innerquantity == inner_quantity]["f"].values
        kde = gaussian_kde(data)
        x = np.linspace(min(data), max(data), 100)
        density = kde(x)
        cumulative_density = np.cumsum(density) / np.sum(density)  # Compute cumulative distribution
        color = cmap(idx+1)  # Get color from the colormap
        ax.plot(x, cumulative_density, marker=markers[idx], linestyle=line_styles[idx % len(line_styles)],
                color=color, markevery=marker_step, label=str(inner_quantity))

    ax.set_xlabel('Objective value')
    ax.set_ylabel('Cumulative Distribution')
    ax.set_title('f_cumulative')
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir + r"/f_tune_cumulative.pdf")
    # plt.show()
    plt.close()


def plot_proposedmethod(data_dir, fig_dir):
    
    # Create an empty DataFrame to store the combined data
    df = pd.DataFrame()


    file_names = [csv_name for csv_name in os.listdir(data_dir) if "proposedmethod" in csv_name and ".txt" in csv_name]
    sorted_file_names = sorted(file_names, key=lambda x: int(x.split("_")[1]))


    for csv_name in sorted_file_names:
        if "proposedmethod" in csv_name and ".txt" in csv_name:
            experiment_index = int(csv_name.split("_")[1])
            method = csv_name.split("_")[3]
            seed = int(csv_name.split("_")[-1].split(".")[0])
            csv_path = os.path.join(data_dir, csv_name)
            tmp_df = pd.read_csv(csv_path, header=0)
            tmp_df["experiment_index"] = experiment_index
            tmp_df["method"] = method
            tmp_df["seed"] = seed
            tmp_df['f_best'] = tmp_df['f'].cummax()

            df = pd.concat([df, tmp_df], ignore_index=True)

    methods = df["method"].unique()
    seeds = sorted(df["seed"].unique())    
    step_slices = 100
    max_steps = stopping_criterion
    print(methods)
    for meth_idx, method in enumerate(methods):
        sub_df: pd.DataFrame = df.copy()[df["method"]==method]
        sub_df = sub_df.reset_index(drop=True)


        x = []
        y_mean = []
        y_lower = []
        y_upper = []


        marker = marker_list[meth_idx]
        linestyle = linestyle_list[meth_idx]
        color = color_list[meth_idx]


        for step in np.linspace(0, max_steps, step_slices):
            step = int(step)
            selected_indices = sub_df[sub_df["step"] < step].groupby("seed")["step"].idxmax()
            scores = np.array(sub_df.loc[selected_indices,]["f_best"])

            if len(scores) < 0.75*len(seeds):
                continue
            x.append(step)
            mean, lower, upper = bootstrap_mean_and_confiance_interval(scores)
            y_mean.append(mean)
            y_lower.append(lower)
            y_upper.append(upper)

        plt.plot(x, y_mean, color=color, linestyle=linestyle, marker=marker, markevery=1/10, label=method)
        plt.legend()
        plt.fill_between(x, y_lower, y_upper, alpha=0.1, color=color, linestyle=linestyle)
    # plt.show()



def plot_comparison_parameters(data_dir, fig_dir):

    df = read_comparison_parameter_csvs(data_dir)
    _plot_performance_reeval_every_best_vs_end("compare_reeval_every_minus_end_quantity", df.copy(), fig_dir, "innerquantity_or_targetprob")
    
    _plot_probability_of_choosing_best_morphology("probability_reevaluated_morphology_beats_previous_best_quantity", df.copy(), fig_dir, "innerquantity_or_targetprob")
    _plot_probability_of_choosing_best_morphology("probability_reevaluated_morphology_beats_previous_best_length", df.copy(), fig_dir, "innerlength_or_startquantity")




    for param, param_preffix in zip(["innerquantity_or_targetprob", "innerlength_or_startquantity"], ["quantity", "length"]):
        _plot_performance(f"{param_preffix}_reevalend",      df.copy(), fig_dir, "reeval",    param, "f", "step")
        _plot_performance(f"{param_preffix}_reevalbest",     df.copy(), fig_dir, "reeval",    param, "f_best", "step_including_reeval")
        _plot_performance(f"{param_preffix}_noreeval",       df.copy(), fig_dir, "no_reeval", param, "f_best", "step")
        _plot_performance(f"{param_preffix}_controllersize", df.copy(), fig_dir, "reeval",    param, "controller_size", "step_including_reeval")
        _plot_performance(f"{param_preffix}_controllersize2",df.copy(), fig_dir, "reeval",    param, "controller_size2", "step_including_reeval")
        _plot_performance(f"{param_preffix}_morphologysize", df.copy(), fig_dir, "reeval",    param, "morphology_size", "step_including_reeval")

    _plot_stability(df.copy(), fig_dir)
    for complexity_metric in ["controller_size", "controller_size2", "morphology_size"]:
        _plot_complexity(df.copy(), fig_dir, complexity_metric)



    



