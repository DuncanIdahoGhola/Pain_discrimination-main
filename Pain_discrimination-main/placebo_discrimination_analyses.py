# %
import numpy as np
import pandas as pd
import os
from os.path import join as opj
import scipy
import matplotlib.pyplot as plt
# from psychopy import data
import seaborn as sns
from statsmodels.stats.weightstats import ttost_paired
import pingouin as pg
import urllib.request
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from tqdm import tqdm
from scipy.stats import norm
import math

############# Setup #############


# Find participants in the sourcedata folder
bidsroot = "sourcedata"  
# List and remove date from file names
participants = [p.split("_")[0] for p in os.listdir(bidsroot) if "sub-" in p]
participants.sort()
# Create derivatives folder if it does not exist
if not os.path.exists("derivatives"):
    os.mkdir("derivatives")
    os.mkdir("derivatives/figures")
    os.mkdir("derivatives/stats")
# Get nice font
urllib.request.urlretrieve(
    "https://github.com/gbif/analytics/raw/master/fonts/Arial%20Narrow.ttf",
    "derivatives/figures/arialnarrow.ttf",
)
fm.fontManager.addfont("derivatives/figures/arialnarrow.ttf")
prop = fm.FontProperties(fname="derivatives/figures/arialnarrow.ttf")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = prop.get_name()

############# PARAMETERS #############
# Exclude participants with perfect discrimination
exclude_perfect = True

# Exclude participants with less than X% placebo effect (0 = negative placebo, -100 keep everyone) None = Ignore this flag and keep all
exclude_placebo = 0

# Add participant to list to exclude based on other criteria
exclude_custom = []


# Create empty lists to store data
all_active, all_thresholds, all_plateaus = [], [], []
all_eval_frames, all_discrim_task = [], []
all_discrim_task_long = []

# Create empty dataframe to store wide data
wide_dat = pd.DataFrame(
    index=participants,
    columns=[
        "acc_all",
        "active_acc_all",
        "inactive_acc_all",
        "active_acc_b1",
        "inactive_acc_b1",
        "active_acc_b2",
        "inactive_acc_b2",
        "temp_pic",
        "temp_plateau",
        "temp_placebo",
        "average_placebo_all",
        "average_placebo_b1",
        "average_placebo_b2",
        "perc_placebo_b1",
        "perc_placebo_b2",
        "perc_placebo_all",
        "average_eval_inactive",
        "average_eval_active",
        "average_eval_active_b1",
        "average_eval_active_b2",
        "average_eval_inactive_b1",
        "average_eval_inactive_b2",
    ],
)

# Loop participants
for p in tqdm(participants, desc="Processing individual participants"):

    # Get participant folder and create derivatives folder
    par_fold = [c for c in os.listdir(bidsroot) if p in c]
    assert len(par_fold) == 1
    par_fold = opj(bidsroot, par_fold[0])
    deriv_path = opj("derivatives", p)
    if not os.path.exists(deriv_path):
        os.mkdir(deriv_path)
    
    # Plot staircase for peak calibration
    # ###################################################
    quest_dir = opj(par_fold, "QUEST")
    # Find main file
    quest_file = [f for f in os.listdir(quest_dir) if ".csv" in f and "trial" not in f]

    if len(quest_file) > 1:  # Make sure there is just one
        quest_file = [quest_file[-1]]
    # Make sure there is just one
    for idx, f in enumerate(quest_file):
        quest = pd.read_csv(os.path.join(par_fold, "QUEST", quest_file[idx]))
        plateau = quest["temp_plateau"].values[0]
        quest = quest[quest["pic_response"] != 'None']  
        trials_num = quest["trials.thisRepN"].dropna().values
        # PLot psychometric function
        quest.loc[quest["pic_response"].isna(), "pic_sent"] = np.nan
        quest_intensities = quest["pic_sent"].dropna().values
        quest_detections = quest["pic_response"].dropna().values

        if "sub-012" in p:
            thresh = 0.5 + plateau
        elif int(p[4:7]) < 7:
            thresh = quest["threshold"].dropna().values[0]
        else:
            thresh = quest["mean_6_reversals"].dropna().values[0] + plateau



        combinedInten = quest_intensities.astype(float)
        combinedResp = quest_detections.astype(float)
        
        # Remove to avoid psychopy dependency
        # fit = data.FitWeibull(
        #     combinedInten, combinedResp, expectedMin=0.5, guess=[0.2, 0.5]
        # )
        
        # smoothInt = np.arange(min(combinedInten), max(combinedInten), 0.001)
        # smoothResp = fit.eval(smoothInt)

        # plt.figure(figsize=(5, 5))
        # plt.plot(smoothInt, smoothResp, "-")
        # if len(quest_file) > 1:
        #     thresh = thresh + plateau
        # # pylab.plot([thresh, thresh],[0,0.8],'--'); pylab.plot([0, thresh], [0.8,0.8],'--')
        thresh = float(thresh)
        plateau = float(plateau)
        plt.title("threshold = %0.3f" % (thresh) + " plateau = %0.1f" % (plateau))
        plt.axvline(x=thresh * 10, color="k", linestyle="--")
        # plot points
        plt.plot(quest_intensities, quest_detections, "o")
        plt.ylim([-0.5, 1.5])
        plt.savefig(opj(deriv_path, p + "_quest.png"))

        # Plot temp - all trials
        temp_files = [
            s
            for s in os.listdir(quest_dir)
            if "temp" in s and s.split("_")[5][:11] in f and ".csv" in s
        ]
        plt.figure(figsize=(5, 5))
        for trial in temp_files:
            temp = pd.read_csv(opj(quest_dir, trial))
            avg_temp = np.average(temp[["z1", "z2", "z3", "z4", "z5"]], axis=1)

            plt.plot(np.arange(len(temp)), avg_temp)
            plt.xlabel("Sample")
            plt.ylabel("Temperature")
        plt.axhline(y=thresh, color="k", linestyle="--", label="threshold")
        plt.axhline(y=plateau, color="r", linestyle="--", label="plateau")
        plt.legend()
        plt.savefig(opj(deriv_path, p + "_quest_" + str(idx).zfill(2) + "_temps.png"))

        # Plot temp - individual trials
        fig, axes = plt.subplots(6, 6, figsize=(10, 10))
        axes = axes.flatten()
        for trial in temp_files:
            trial_num = int(trial.split("_")[-1].replace(".csv", ""))
            if trial_num in trials_num:
                temp = pd.read_csv(opj(quest_dir, trial))
                axes[trial_num].set_title(trial_num)
                axes[trial_num].set_ylim(42, plateau + 2)
                pic = quest.loc[trial_num, "pic_sent"] / 10
                axes[trial_num].plot(
                    np.arange(len(temp)),
                    np.average(temp[["z1", "z2", "z3", "z4", "z5"]], 1),
                    label=trial_num,
                )
                axes[trial_num].axhline(y=pic, color="k", linestyle="--", label="pic")

        fig.suptitle(p + " all quest trials")
        plt.tight_layout()
        plt.xlabel("Sample")
        plt.ylabel("Temperature")
        plt.savefig(
            opj(deriv_path, p + "_quest_" + str(idx).zfill(2) + "_temps_all.png")
        )

    ###################################################
    # Main task - Eval
    ###################################################

    # Find main file
    main_file = [
        f
        for f in os.listdir(par_fold)
        if "maintask" in f and "trial" not in f and ".csv" in f
    ]

    assert len(main_file) == 1  # Make sure there is just one

    main = pd.read_csv(os.path.join(par_fold, main_file[0]))

    # Some files are saved with a different delimiter
    if len(main.columns) == 1:
        main = pd.read_csv(os.path.join(par_fold, main_file[0]), sep=";")

    wide_dat.loc[p, "temp_pic"] = np.round(
        main["temp_pic_set"].values[0] - main["temp_flat"].values[0], 2
    )
    wide_dat.loc[p, "temp_plateau"] = np.round(main["temp_flat"].values[0], 2)
    wide_dat.loc[p, "temp_placebo"] = np.round(main["temp_active"].values[0], 2)

    # Evaluation task
    eval_task = main[
        [
            "ratingScale.response",
            "condition",
            "loop_eval_discri.thisRepN",
            "trials.thisN",
        ]
    ]
    # Remove missed trials
    eval_task = eval_task[eval_task["ratingScale.response"].isna() == False]

    # Get evaluation task trials for active and inactive
    eval_task_active_noplacebo = eval_task[
        (eval_task["condition"] == "active")
        & (eval_task["loop_eval_discri.thisRepN"].isna())
    ].reset_index(drop=True)
    eval_task_inactive_noplacebo = eval_task[
        (eval_task["condition"] == "inactive")
        & (eval_task["loop_eval_discri.thisRepN"].isna())
    ].reset_index(drop=True)

    # Get evaluation task trials for active and inactive during placebo
    eval_task_active_placebo = eval_task[
        (eval_task["condition"] == "active")
        & (~eval_task["loop_eval_discri.thisRepN"].isna())
    ].reset_index(drop=True)
    eval_task_inactive_placebo = eval_task[
        (eval_task["condition"] == "inactive")
        & ~(eval_task["loop_eval_discri.thisRepN"].isna())
    ].reset_index(drop=True)

    # Get evaluation task trials for active and inactive
    eval_task_active = eval_task[(eval_task["condition"] == "active")].reset_index(
        drop=True
    )
    eval_task_inactive = eval_task[(eval_task["condition"] == "inactive")].reset_index(
        drop=True
    )

    # Add eval task to wide_dat
    wide_dat.loc[p, "average_eval_active"] = np.mean(
        eval_task_active_noplacebo["ratingScale.response"].values
    )
    wide_dat.loc[p, "average_eval_inactive"] = np.mean(
        eval_task_inactive_noplacebo["ratingScale.response"].values
    )

    wide_dat.loc[p, "average_placebo_eval_active"] = np.mean(
        eval_task_active_placebo["ratingScale.response"].values
    )
    wide_dat.loc[p, "average_placebo_eval_inactive"] = np.mean(
        eval_task_inactive_placebo["ratingScale.response"].values
    )

    # Calculate placebo effect
    placebo_diff = np.mean(
        eval_task_inactive_placebo["ratingScale.response"].values
        - eval_task_active_placebo["ratingScale.response"].values
    )
    # Calculate placebo effect in percentage
    placebo_diff_perc = (
        placebo_diff
        / np.nanmean( eval_task_inactive_placebo["ratingScale.response"].values)
        * 100
    )

    # Add placebo to wide_dat
    wide_dat.loc[p, "average_placebo_all"] = placebo_diff
    wide_dat.loc[p, "perc_placebo_all"] = placebo_diff_perc

    # Do the same separately for block 1 and 2
    eval_task_b1 = eval_task[eval_task["trials.thisN"] == 0]
    eval_task_b2 = eval_task[eval_task["trials.thisN"] == 1]

    eval_task_active_noplacebo_b1 = eval_task_b1[
        (eval_task_b1["condition"] == "active")
        & (eval_task_b1["loop_eval_discri.thisRepN"].isna())
    ].reset_index(drop=True)
    eval_task_active_noplacebo_b2 = eval_task_b2[
        (eval_task_b2["condition"] == "active")
        & (eval_task_b2["loop_eval_discri.thisRepN"].isna())
    ].reset_index(drop=True)

    wide_dat.loc[p, "average_eval_active_b1"] = np.mean(
        eval_task_active_noplacebo_b1["ratingScale.response"].values
    )
    wide_dat.loc[p, "average_eval_active_b2"] = np.mean(
        eval_task_active_noplacebo_b2["ratingScale.response"].values
    )

    eval_task_inactive_noplacebo_b1 = eval_task_b1[
        (eval_task_b1["condition"] == "inactive")
        & (eval_task_b1["loop_eval_discri.thisRepN"].isna())
    ].reset_index(drop=True)
    eval_task_inactive_noplacebo_b2 = eval_task_b2[
        (eval_task_b2["condition"] == "inactive")
        & (eval_task_b2["loop_eval_discri.thisRepN"].isna())
    ].reset_index(drop=True)

    wide_dat.loc[p, "average_eval_inactive"] = np.mean(
        eval_task_inactive_noplacebo["ratingScale.response"].values
    )
    wide_dat.loc[p, "average_eval_inactive"] = np.mean(
        eval_task_inactive_noplacebo["ratingScale.response"].values
    )

    placebo_diff_b1 = np.mean(
        eval_task_inactive_placebo[eval_task_inactive_placebo["trials.thisN"] == 0][
            "ratingScale.response"
        ].values
        - eval_task_active_placebo[eval_task_active_placebo["trials.thisN"] == 0][
            "ratingScale.response"
        ].values
    )
    placebo_diff_b2 = np.mean(
        eval_task_inactive_placebo[eval_task_inactive_placebo["trials.thisN"] == 1][
            "ratingScale.response"
        ].values
        - eval_task_active_placebo[eval_task_active_placebo["trials.thisN"] == 1][
            "ratingScale.response"
        ].values
    )

    placebo_diff_all_b1 = (
        eval_task_inactive_placebo[eval_task_inactive_placebo["trials.thisN"] == 0][
            "ratingScale.response"
        ].values
        - eval_task_active_placebo[eval_task_active_placebo["trials.thisN"] == 0][
            "ratingScale.response"
        ].values
    )
    placebo_diff_all_b2 = (
        eval_task_inactive_placebo[eval_task_inactive_placebo["trials.thisN"] == 1][
            "ratingScale.response"
        ].values
        - eval_task_active_placebo[eval_task_active_placebo["trials.thisN"] == 1][
            "ratingScale.response"
        ].values
    )

    for i in range(len(placebo_diff_all_b1)):
        wide_dat.loc[p, "placebo_b1_" + str(i + 1)] = placebo_diff_all_b1[i]
    for i in range(len(placebo_diff_all_b2)):
        wide_dat.loc[p, "placebo_b2_" + str(i + 1)] = placebo_diff_all_b2[i]

    wide_dat.loc[p, "average_placebo_eval_active_b1"] = np.mean(
        eval_task_active_placebo[eval_task_active_placebo["trials.thisN"] == 0][
            "ratingScale.response"
        ].values
    )
    wide_dat.loc[p, "average_placebo_eval_inactive_b1"] = np.mean(
        eval_task_inactive_placebo[eval_task_inactive_placebo["trials.thisN"] == 0][
            "ratingScale.response"
        ].values
    )

    wide_dat.loc[p, "average_placebo_eval_inactive_b2"] = np.mean(
        eval_task_active_placebo[eval_task_active_placebo["trials.thisN"] == 1][
            "ratingScale.response"
        ].values
    )
    wide_dat.loc[p, "average_placebo_eval_inactive_b2"] = np.mean(
        eval_task_inactive_placebo[eval_task_inactive_placebo["trials.thisN"] == 1][
            "ratingScale.response"
        ].values
    )

    wide_dat.loc[p, "average_placebo_b1"] = placebo_diff_b1
    wide_dat.loc[p, "average_placebo_b2"] = placebo_diff_b2

    placebo_diff_perc_b1 = (
        placebo_diff_b1
        / np.mean(
            eval_task_inactive_placebo[eval_task_inactive_placebo["trials.thisN"] == 0][
                "ratingScale.response"
            ].values
        )
        * 100
    )
    placebo_diff_perc_b2 = (
        placebo_diff_b2
        / np.mean(
            eval_task_inactive_placebo[eval_task_inactive_placebo["trials.thisN"] == 1][
                "ratingScale.response"
            ].values
        )
        * 100
    )

    wide_dat.loc[p, "perc_placebo_b1"] = placebo_diff_perc_b1
    wide_dat.loc[p, "perc_placebo_b2"] = placebo_diff_perc_b2

    # Get threshold used
    thresh = main["temp_pic_set"].values[0]

    # Plot evaluation task for this participant
    plt.figure()
    plt.plot(eval_task_active["ratingScale.response"].values, color="g")
    plt.scatter(
        x=np.arange(len(eval_task_active)),
        y=eval_task_active["ratingScale.response"].values,
        color="g",
    )

    plt.plot(eval_task_inactive["ratingScale.response"].values, color="r")
    plt.scatter(
        x=np.arange(len(eval_task_inactive)),
        y=eval_task_inactive["ratingScale.response"].values,
        color="r",
    )

    plt.axvline(x=7.5, color="k", linestyle="--")
    plt.axvline(x=11.5, color="k", linestyle="--")
    plt.axvline(x=19.5, color="k", linestyle="--")
    plt.ylim([0, 100])
    plt.xlabel("Trial")
    plt.ylabel("Pain intensity rating")
    plt.savefig(opj(deriv_path, p + "_eval_task.png"))

    # Create dataframe for evaluation task
    out_frame = pd.DataFrame(
        dict(
            ratings=list(eval_task_active["ratingScale.response"].values)
            + list(eval_task_inactive["ratingScale.response"].values),
            condition=["active"] * len(eval_task_active)
            + ["inactive"] * len(eval_task_inactive),
            trial=list(np.arange(len(eval_task_active))) * 2,
            participant=p[:7],
            block=np.where(np.asarray(list(np.arange(len(eval_task_active))) * 2) < 12, 1, 2),
        )
    )
    # Add to list of all evaluation task dataframes
    all_eval_frames.append(out_frame)

    # Plot temperature in each trial
    temp_files = [
        f
        for f in os.listdir(par_fold)
        if "maintask" in f and "temp_trial_eval" in f and ".csv" in f
    ]
    plt.figure(figsize=(5, 5))
    for trial in temp_files:
        temp = pd.read_csv(opj(par_fold, trial))
        avg_temp = np.average(temp[["z1", "z2", "z3", "z4", "z5"]], axis=1)

        plt.plot(np.arange(len(temp)), avg_temp, color="grey", alpha=0.5)
        plt.xlabel("Sample")
        plt.ylim([38, plateau + 2])
        plt.ylabel("Temperature")

    plt.axhline(
        y=main["temp_flat"].values[0], color="k", linestyle="--", label="plateau"
    )
    plt.axhline(
        y=main["temp_active"].values[0], color="r", linestyle="--", label="active"
    )
    plt.legend()
    plt.savefig(opj(deriv_path, p + "_task_temp_eval.png"))

    ###################################################
    # Main task - discrimination
    ###################################################
    discrim_task = main[
        [
            "participant",
            "condition",
            "pic_presence",
            "pic_response",
            "trials.thisN",
            "loop_eval_discri.thisN",
            "loop_discri.thisRepN",
        ]
    ]
    discrim_task = discrim_task[discrim_task["loop_discri.thisRepN"].isna() == False]

    # List for plot
    discrim_task["actual_trial"] = (
        [9] * 4
        + [10] * 4
        + [11] * 4
        + [12] * 4
        + [21] * 4
        + [22] * 4
        + [23] * 4
        + [24] * 4
    )
    
    discrim_task["pic_response"] = discrim_task["pic_response"].fillna("None")
    discrim_task = discrim_task[discrim_task["pic_response"] != "None"]
    discrim_task["pic_response"] = discrim_task["pic_response"].astype(int)
    discrim_task.reset_index(drop=True, inplace=True)

    accurate = []
    detection_type = []
    for presence, response in zip(
        discrim_task["pic_presence"].values, discrim_task["pic_response"].values
    ):
        if presence == "pic-present" and response == 1:
            accurate.append(1)
            detection_type.append("hit")
        elif presence == "pic-absent" and response == 0:
            accurate.append(1)
            detection_type.append("correct rejection")
        elif presence == "pic-present" and response == 0:
            accurate.append(0)
            detection_type.append("miss")
        elif presence == "pic-absent" and response == 1:
            accurate.append(0)
            detection_type.append("false alarm")
        


    discrim_task["accuracy"] = accurate
    discrim_task["detection_type"] = detection_type

    # Signal detection theory measures
    hits = len(discrim_task[discrim_task["detection_type"] == "hit"])
    misses = len(discrim_task[discrim_task["detection_type"] == "miss"])
    fas = len(discrim_task[discrim_task["detection_type"] == "false alarm"])
    crs = len(discrim_task[discrim_task["detection_type"] == "correct rejection"])

    Z = norm.ppf
    # Calculate d prime for this participant
    def SDT(hits, misses, fas, crs):
        """ returns a dict with d-prime measures given hits, misses, false alarms, and correct rejections"""
        # Floors an ceilings are replaced by half hits and half FA's
        half_hit = 0.5 / (hits + misses)
        half_fa = 0.5 / (fas + crs)
    
        # Calculate hit_rate and avoid d' infinity
        hit_rate = hits / (hits + misses)
        if hit_rate == 1: 
            hit_rate = 1 - half_hit
        if hit_rate == 0: 
            hit_rate = half_hit
    
        # Calculate false alarm rate and avoid d' infinity
        fa_rate = fas / (fas + crs)
        if fa_rate == 1: 
            fa_rate = 1 - half_fa
        if fa_rate == 0: 
            fa_rate = half_fa
    
        # Return d', beta, c and Ad'
        out = {}
        out['d'] = Z(hit_rate) - Z(fa_rate)
        out['beta'] = math.exp((Z(fa_rate)**2 - Z(hit_rate)**2) / 2)
        out['c'] = -(Z(hit_rate) + Z(fa_rate)) / 2
        out['Ad'] = norm.cdf(out['d'] / math.sqrt(2))
        
        return(out)
    out = SDT(hits, misses, fas, crs)

    # Add d prime to wide_dat
    wide_dat.loc[p, "d_prime_all"] = out["d"]
    wide_dat.loc[p, "beta_all"] = out["beta"]
    wide_dat.loc[p, "c_all"] = out["c"]

    # signal detection in active and inactive conditions
    hits_active = np.sum(discrim_task[discrim_task["condition"] == "active"]["detection_type"] == "hit")
    misses_active = np.sum(discrim_task[discrim_task["condition"] == "active"]["detection_type"] == "miss")
    fas_active = np.sum(discrim_task[discrim_task["condition"] == "active"]["detection_type"] == "false alarm")
    crs_active = np.sum(discrim_task[discrim_task["condition"] == "active"]["detection_type"] == "correct rejection")
    hits_inactive = np.sum(discrim_task[discrim_task["condition"] == "inactive"]["detection_type"] == "hit")
    misses_inactive = np.sum(discrim_task[discrim_task["condition"] == "inactive"]["detection_type"] == "miss")
    fas_inactive = np.sum(discrim_task[discrim_task["condition"] == "inactive"]["detection_type"] == "false alarm")
    crs_inactive = np.sum(discrim_task[discrim_task["condition"] == "inactive"]["detection_type"] == "correct rejection")
    
    out_active = SDT(hits_active, misses_active, fas_active, crs_active)
    out_inactive = SDT(hits_inactive, misses_inactive, fas_inactive, crs_inactive)
    # Add d prime to wide_dat
    wide_dat.loc[p, "d_prime_active"] = out_active["d"]
    wide_dat.loc[p, "d_prime_inactive"] = out_inactive["d"]
    wide_dat.loc[p, "beta_active"] = out_active["beta"]
    wide_dat.loc[p, "beta_inactive"] = out_inactive["beta"]
    wide_dat.loc[p, "c_active"] = out_active["c"]
    wide_dat.loc[p, "c_inactive"] = out_inactive["c"]

    wide_dat.loc[p, "acc_all"] = np.mean(discrim_task["accuracy"].values)

    wide_dat.loc[p, "active_acc_all"] = np.mean(
        discrim_task[discrim_task["condition"] == "active"]["accuracy"].values
    )
    wide_dat.loc[p, "inactive_acc_all"] = np.mean(
        discrim_task[discrim_task["condition"] == "inactive"]["accuracy"].values
    )

    discrim_task_b1 = discrim_task[discrim_task["trials.thisN"] == 0]
    wide_dat.loc[p, "active_acc_b1"] = np.mean(
        discrim_task_b1[discrim_task_b1["condition"] == "active"]["accuracy"].values
    )
    wide_dat.loc[p, "inactive_acc_b1"] = np.mean(
        discrim_task_b1[discrim_task_b1["condition"] == "inactive"]["accuracy"].values
    )

    discrim_task_b2 = discrim_task[discrim_task["trials.thisN"] == 1]
    wide_dat.loc[p, "active_acc_b2"] = np.mean(
        discrim_task_b2[discrim_task_b2["condition"] == "active"]["accuracy"].values
    )
    wide_dat.loc[p, "inactive_acc_b2"] = np.mean(
        discrim_task_b2[discrim_task_b2["condition"] == "inactive"]["accuracy"].values
    )


    #Calculate average respone time using responser.started and responser.stopped
    reaction_time = []
    reaction_time_good_a = []
    reaction_time_bad_a = []
    for _, row in main.iterrows():
        if pd.notnull(row['discrimin_resp.rt']):
            rt = float(row["discrimin_resp.rt"])
            reaction_time.append(rt)

            if row["pic_response"] == 1 and row["pic_presence"] == "pic-present":
                reaction_time_good_a.append(rt)
            elif row["pic_response"] == 0 and row["pic_presence"] == "pic-absent":
                reaction_time_good_a.append(rt)
            else:
                reaction_time_bad_a.append(rt)

    #add all reaction times to discrim_task
    discrim_task.loc[p, "reaction_time"] = np.mean(reaction_time) if reaction_time else np.nan
    discrim_task.loc[p, "reaction_time_good_a"] = np.mean(reaction_time_good_a) if reaction_time_good_a else np.nan
    discrim_task.loc[p, "reaction_time_bad_a"] = np.mean(reaction_time_bad_a) if reaction_time_bad_a else np.nan

    discrim_task_avg = discrim_task.groupby(["condition"]).mean(numeric_only=True).reset_index()
    discrim_task_avg["participant"] = p

    all_discrim_task.append(discrim_task_avg)
    all_discrim_task_long.append(discrim_task)
    plt.figure()
    sns.catplot(x="condition", y="accuracy", data=discrim_task_avg, kind="point")
    plt.ylim([0, 1.2])
    plt.savefig(opj(deriv_path, p + "_discrim_task.png"))

    # Plot temperature pic
    temp_files = [
        f
        for f in os.listdir(par_fold)
        if "maintask" in f and "temp_trial_pic" in f and ".csv" in f
    ]
    plt.figure(figsize=(5, 5))
    for trial in temp_files:
        temp = pd.read_csv(opj(par_fold, trial))
        avg_temp = np.average(temp[["z1", "z2", "z3", "z4", "z5"]], axis=1)

        plt.plot(np.arange(len(temp)), avg_temp, color="grey", alpha=0.5)
        plt.xlabel("Sample")
        plt.ylabel("Temperature")
    plt.axhline(
        y=main["temp_flat"].values[0], color="k", linestyle="--", label="active"
    )
    plt.axhline(y=thresh, color="r", linestyle="--", label="plateau")
    plt.legend()
    plt.savefig(opj(deriv_path, p + "_task_temp_pic.png"))
    plt.close("all")

    

# Cocncatenate all dataframes from all participants

all_eval_frame = pd.concat(all_eval_frames)
all_discrim_task = pd.concat(all_discrim_task)
all_discrim_task_long = pd.concat(all_discrim_task_long)
wide_dat["participant"] = list(wide_dat.index)


# TODO ADD QUESTIONNAIRES SUMMARY HERE


#add questionnaires 
#iastay1
iastay1 = pd.read_csv(opj(bidsroot, "iasta_y1.csv"))
#iastay2
iastay2 = pd.read_csv(opj(bidsroot, "iasta_y2.csv"))
#pcs
pcs = pd.read_csv(opj(bidsroot, "pcs.csv"))
#results
results_df = pd.DataFrame(index=iastay1.index)
# Add BDI, IASTA, and PCS scores for each participant
for p in iastay1.index:
    # IASTA Y2
    all_iasta2 = []
    for c in range(2, len(iastay2.columns)):
        try:
            results_df.loc[p, "qiastay2_" + list(iastay2.columns)[c]] = int(
                str(iastay2.loc[p, list(iastay2.columns)[c]])[0]
            )
        except:
            results_df.loc[p, "qiastay2_" + list(iastay2.columns)[c]] = "nan"
        try:
            all_iasta2.append(int(str(iastay2.loc[p, list(iastay2.columns)[c]])[0]))
        except:
            all_iasta2.append(np.nan)
    assert len(all_iasta2) == 20
    # Invert scores for some columns [0, 2, 5, 6, 9, 12, 13, 15, 18]
    all_iasta2 = np.asarray(all_iasta2)
    all_iasta2[[0, 2, 5, 6, 9, 12, 13, 15, 18]] = (
        5 - all_iasta2[[0, 2, 5, 6, 9, 12, 13, 15, 18]]
    )
    # Add total
    results_df.loc[p, "iastay2_total"] = np.nansum(all_iasta2)

    # IASTA Y1
    all_iasta1 = []
    for c in range(2, len(iastay1.columns)):
        try:
            results_df.loc[p, "qiastay1_" + list(iastay1.columns)[c]] = int(
                str(iastay1.loc[p, list(iastay1.columns)[c]])[0]
            )
        except:
            results_df.loc[p, "qiastay1_" + list(iastay1.columns)[c]] = "nan"
        try:
            all_iasta1.append(int(str(iastay1.loc[p, list(iastay1.columns)[c]])[0]))
        except:
            all_iasta1.append(np.nan)
    assert len(all_iasta1) == 20

    # Invert scores for some columns [0,  1,  4,  7,  9, 10, 14, 15, 18, 19]
    all_iasta1 = np.asarray(all_iasta1)
    all_iasta1[[0, 1, 4, 7, 9, 10, 14, 15, 18, 19]] = (
        5 - all_iasta1[[0, 1, 4, 7, 9, 10, 14, 15, 18, 19]]
    )
    # Add total
    results_df.loc[p, "iastay1_total"] = np.nansum(all_iasta1)

    # PCS
    all_pcs = []
    for c in range(2, len(pcs.columns)):
        try:
            results_df.loc[p, "qpcs_" + list(pcs.columns)[c]] = int(
                str(pcs.loc[p, list(pcs.columns)[c]])[0]
            )
            all_pcs.append(int(str(pcs.loc[p, list(pcs.columns)[c]])[0]))
        except:
            results_df.loc[p, "qpcs_" + list(pcs.columns)[c]] = "nan"
    if len(all_pcs) == 13:
        # Add total
        results_df.loc[p, "pcs_total"] = np.nansum(all_pcs)
    else:
        results_df.loc[p, "pcs_total"] = "nan"

# Add IASTA and PCS scores to sociodemo
socio = pd.read_csv(opj(bidsroot, "sociodemo.csv"))
socio['pcs_total'] = results_df["pcs_total"].reset_index(drop=True)
socio['iastay1_total'] = results_df["iastay1_total"].reset_index(drop=True)
socio['iastay2_total'] = results_df["iastay2_total"].reset_index(drop=True)
#resave socio to csv
socio.to_csv(opj(bidsroot, "sociodemo.csv"), index=False)

#we can add the iasta and pcs scores to the wide_dat dataframe if needed

# Add sociodemo to wide_dats

socio = pd.read_csv(opj(bidsroot, "sociodemo.csv"))
socio.index = socio[socio.columns[1]]

wide_dat["age"] = np.nan
for row in socio.iterrows():
    if row[0] in wide_dat.index:
        wide_dat.loc[row[0], "age"] = int(row[1]["2. Quel est votre âge en années? "])
        wide_dat.loc[row[0], "ismale"] = (
            row[1]["4. Quel est votre genre? "] == "Masculin"
        )
        wide_dat.loc[row[0], "isfemale"] = (
            row[1]["4. Quel est votre genre? "] == "Féminin"
        )
        wide_dat.loc[row[0], "pcs_total"] = row[1]["pcs_total"]
        wide_dat.loc[row[0], "iastay1_total"] = row[1]["iastay1_total"]
        wide_dat.loc[row[0], "iastay2_total"] = row[1]["iastay2_total"]


# Convert to int
wide_dat["ismale"] = wide_dat["ismale"].astype(int)
wide_dat["isfemale"] = wide_dat["isfemale"].astype(int)

# Create a new column for the exclude flag
wide_dat['exclude'] = 0
all_eval_frame["exclude"] = 0
all_discrim_task["exclude"] = 0
all_discrim_task_long["exclude"] = 0

if exclude_perfect:
    wide_dat_perf = (
        list(wide_dat[wide_dat["acc_all"] == 1]["participant"])
    )

    # Switch exclude to 1 for participants in wide_dat_perf
    wide_dat.loc[wide_dat["participant"].isin(wide_dat_perf), 'exclude'] = 1
    all_eval_frame.loc[all_eval_frame["participant"].isin(wide_dat_perf), 'exclude'] = 1
    all_discrim_task.loc[all_discrim_task["participant"].isin(wide_dat_perf), 'exclude'] = 1
    all_discrim_task_long.loc[all_discrim_task_long["participant"].isin(wide_dat_perf), 'exclude'] = 1

    print(str(len(wide_dat_perf)) + " participants with perfect discrimination excluded," + "leaving " + str(len(wide_dat) - len(wide_dat_perf)) + " participants")


if exclude_placebo is not None:
    wide_dat_placebo = (
        list(wide_dat[wide_dat["perc_placebo_all"] < exclude_placebo]["participant"])
    )
    # Switch exclude to 1 for participants in wide_dat_placebo
    wide_dat.loc[wide_dat["participant"].isin(wide_dat_placebo), 'exclude'] = 1
    all_eval_frame.loc[all_eval_frame["participant"].isin(wide_dat_placebo), 'exclude'] = 1
    all_discrim_task.loc[all_discrim_task["participant"].isin(wide_dat_placebo), 'exclude'] = 1
    all_discrim_task_long.loc[all_discrim_task_long["participant"].isin(wide_dat_placebo), 'exclude'] = 1

    print(str(len(wide_dat_placebo)) + " participants with low placebo effect excluded," + "leaving " + str(len(wide_dat) - np.sum(wide_dat['exclude'])) + " participants")

if exclude_custom:
    # Switch exclude to 1 for participants in wide_dat_perf
    wide_dat.loc[wide_dat["participant"].isin(exclude_custom), 'exclude'] = 1
    all_eval_frame.loc[all_eval_frame["participant"].isin(exclude_custom), 'exclude'] = 1
    all_discrim_task.loc[all_discrim_task["participant"].isin(exclude_custom), 'exclude'] = 1
    all_discrim_task_long.loc[all_discrim_task_long["participant"].isin(exclude_custom), 'exclude'] = 1

    print(str(len(exclude_custom)) + " participants excluded for other reasons," + "leaving " + str(len(wide_dat) - np.sum(wide_dat['exclude'])) + " participants")

# Save full dataframes
wide_dat.to_csv(opj("derivatives", "data_wide_dat_full.csv"), index=False)
all_eval_frame.to_csv(opj("derivatives", "data_all_eval_frame_full.csv"), index=False)
all_discrim_task.to_csv(opj("derivatives", "data_all_discrim_task_full.csv"), index=False)
all_discrim_task_long.to_csv(opj("derivatives", "data_all_discrim_task_long_full.csv"), index=False)

# Remove excluded participants
wide_dat = wide_dat[wide_dat["exclude"] == 0]
all_eval_frame = all_eval_frame[all_eval_frame["exclude"] == 0]
all_discrim_task = all_discrim_task[all_discrim_task["exclude"] == 0]
all_discrim_task_long = all_discrim_task_long[all_discrim_task_long["exclude"] == 0]

# Save data with exclusions
wide_dat.to_csv(opj("derivatives", "data_wide_dat_withexcl.csv"), index=False)
all_eval_frame.to_csv(opj("derivatives", "data_all_eval_frame_withexcl.csv"), index=False)
all_discrim_task.to_csv(opj("derivatives", "data_all_discrim_task_withexcl.csv"), index=False)
all_discrim_task_long.to_csv(opj("derivatives", "data_all_discrim_task_long_withexcl.csv"), index=False)

# Make plots

# Eval by trial plot
fig, ax = plt.subplots(figsize=(7, 4))
sns.pointplot(
    x="trial",
    y="ratings",
    hue="condition",
    data=all_eval_frame,
    errorbar="se",
    alpha=0.9,
    ax=ax,
)
plt.ylim([0, 80])
ax.fill_between(
    np.arange(7.5, 11.5, 0.1), y1=80, facecolor="#90ee90", alpha=0.2, zorder=0
)
ax.fill_between(
    np.arange(19.5, 24, 0.1),
    y1=80,
    facecolor="#90ee90",
    alpha=0.2,
    label="Conditoning",
    zorder=0,
)
ax.fill_between(
    np.arange(-1, 7.5, 0.1),
    y1=80,
    facecolor="#8a8a8a",
    alpha=0.2,
    label="Placebo",
    zorder=0,
)
ax.fill_between(
    np.arange(11.5, 19.5, 0.1),
    y1=80,
    facecolor="#8a8a8a",
    alpha=0.2,
    label="Placebo",
    zorder=0,
)

l, h = ax.get_legend_handles_labels()
l.append(
    Patch(
        facecolor="#90ee90",
        edgecolor="#90ee90",
        label="Conditioning",
        alpha=1,
        zorder=999,
    )
)
l.append(
    Patch(
        facecolor="#8a8a8a", edgecolor="#8a8a8a", label="Placebo", alpha=1, zorder=999
    )
)
ax.legend(
    l,
    ["TENS on", "TENS off", "Placebo", "Conditioning"],
    fontsize=12,
    title="",
    loc="upper right",
    ncol=4,
    columnspacing=0.5,
)

plt.ylabel("Pain rating", fontsize=18)
plt.xlabel("Trials", fontsize=18)
plt.xticks(labels=np.arange(1, 25), ticks=np.arange(24))
plt.tick_params(labelsize=14)
plt.setp(ax.collections[0:2], alpha=0.4)  # for the markers
plt.setp(ax.lines, alpha=0.6)
plt.tight_layout()
plt.savefig(opj("derivatives/figures", "group_eval_task.png"), dpi=800, bbox_inches="tight")


# Same but with discrimination interspersed
all_discrim_task_long_avg = (
    all_discrim_task_long.groupby(["participant", "condition", "actual_trial"])
    .mean(numeric_only=True)
    .reset_index()
)

all_discrim_task_long_avg["actual_trial"] = (
    all_discrim_task_long_avg["actual_trial"].astype(int) - 0.8
)
all_eval_frame["trial"].astype(int)

all_eval_frame_avg = (
    all_eval_frame.groupby(["participant", "condition", "trial"]).mean(numeric_only=True).reset_index()
)


color_placebo_back = "#9fc8c8"
color_conditioning_back = "#BBBBBB"
TENS_on = "#a00000"
TENS_off = "#1a80bb"
pal = [TENS_off, TENS_on]
all_discrim_task_long_avg_p1 = all_discrim_task_long_avg[
    all_discrim_task_long_avg["actual_trial"] < 12
]
all_discrim_task_long_avg_p2 = all_discrim_task_long_avg[
    all_discrim_task_long_avg["actual_trial"] > 12
]
fig, (host2, host) = plt.subplots(
    2, 1, figsize=(7, 6), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
)
plt.sca(host)
sns.pointplot(
    x="actual_trial",
    y="accuracy",
    hue="condition",
    errorbar="se",
    data=all_discrim_task_long_avg_p1,
    alpha=0.9,
    native_scale=True,
    palette=pal,
)
sns.pointplot(
    x="actual_trial",
    y="accuracy",
    hue="condition",
    errorbar="se",
    data=all_discrim_task_long_avg_p2,
    alpha=0.9,
    native_scale=True,
    palette=pal,
)
plt.sca(host2)
sns.pointplot(
    x="trial",
    y="ratings",
    hue="condition",
    data=all_eval_frame_avg,
    errorbar="se",
    alpha=0.9,
    palette=pal,
)

host.set_ylim([0.5, 1])
host2.set_ylim([0, 70])
plt.ylabel("Pain rating", fontsize=18)
host.set_xlim([-1, 23.6])
host.set_xlabel("Trials", fontsize=18)
plt.tick_params(labelsize=14)
host.fill_between(
    np.arange(7.5, 11.5, 0.1), y1=1, facecolor=color_placebo_back, alpha=0.4, zorder=0
)
host.fill_between(
    np.arange(19.5, 24, 0.1),
    y1=1,
    facecolor=color_placebo_back,
    alpha=0.4,
    label="Conditoning",
    zorder=0,
)
host.fill_between(
    np.arange(-1, 7.5, 0.1),
    y1=1,
    facecolor=color_conditioning_back,
    alpha=0.2,
    label="Placebo",
    zorder=0,
)
host.fill_between(
    np.arange(11.5, 19.5, 0.1),
    y1=1,
    facecolor=color_conditioning_back,
    alpha=0.2,
    label="Placebo",
    zorder=0,
)

host2.fill_between(
    np.arange(7.5, 11.5, 0.1), y1=80, facecolor=color_placebo_back, alpha=0.4, zorder=0
)
host2.fill_between(
    np.arange(19.5, 24, 0.1),
    y1=80,
    facecolor=color_placebo_back,
    alpha=0.4,
    label="Conditoning",
    zorder=0,
)
host2.fill_between(
    np.arange(-1, 7.5, 0.1),
    y1=80,
    facecolor=color_conditioning_back,
    alpha=0.2,
    label="Placebo",
    zorder=0,
)
host2.fill_between(
    np.arange(11.5, 19.5, 0.1),
    y1=80,
    facecolor=color_conditioning_back,
    alpha=0.2,
    label="Placebo",
    zorder=0,
)
plt.xticks(labels=np.arange(1, 25), ticks=np.arange(24))
host2.axvline(7.5, color="gray", linestyle="--")
host.axvline(7.5, color="gray", linestyle="--")
host2.axvline(11.5, color="gray", linestyle="--")
host.axvline(11.5, color="gray", linestyle="--")

host2.axvline(19.5, color="gray", linestyle="--")
host.axvline(19.5, color="gray", linestyle="--")
host2.axvline(23.5, color="gray", linestyle="--")
host.axvline(23.5, color="gray", linestyle="--")

host.set_ylabel("Proportion correct", fontsize=18)
l, h = host.get_legend_handles_labels()
l = l[0:2]

l.append(
    Patch(
        facecolor=color_placebo_back,
        edgecolor=color_placebo_back,
        label="Placebo",
        alpha=0.4,
        zorder=999,
    )
)
l.append(
    Patch(
        facecolor=color_conditioning_back,
        edgecolor=color_conditioning_back,
        label="Conditioning",
        alpha=0.2,
        zorder=999,
    )
)
host2.legend(
    l,
    ["TENS on", "TENS off", "Placebo", "Conditioning"],
    fontsize=12,
    title="",
    loc="lower left",
    ncol=4,
    columnspacing=0.5,
)
host.tick_params(labelsize=14)
host2.tick_params(labelsize=14)
host.legend().remove()

plt.setp(ax.collections, alpha=0.6)  # for the markers
plt.setp(ax.lines, alpha=0.6)
plt.tight_layout()
plt.savefig(opj("derivatives/figures", "group_discrim_task.png"), dpi=800, bbox_inches="tight")


plt.ylim([0, 80])
ax.fill_between(
    np.arange(7.5, 11.5, 0.1), y1=80, facecolor="#d3d3d3", alpha=0.2, zorder=0
)
ax.fill_between(
    np.arange(19.5, 23.5, 0.1),
    y1=80,
    facecolor="#d3d3d3",
    alpha=0.2,
    label="Placebo",
    zorder=0,
)
l, h = ax.get_legend_handles_labels()
l.append(
    Patch(
        facecolor="#d3d3d3", edgecolor="#d3d3d3", label="Placebo", alpha=1, zorder=999
    )
)
ax.legend(
    l, ["TENS on", "TENS off", "Placebo"], fontsize=12, title="", loc="upper right"
)

plt.ylabel("Pain rating", fontsize=18)
plt.xlabel("Trials", fontsize=18)
plt.xticks(labels=np.arange(1, 25), ticks=np.arange(24))

plt.tight_layout()
plt.savefig(opj("derivatives/figures", "group_eval_task.png"), dpi=800)


fig, ax = plt.subplots(figsize=(4, 4))
all_eval_frame_placeb = all_eval_frame[
    all_eval_frame.trial.isin([8, 9, 10, 11, 20, 21, 22, 23])
]
all_eval_frame_placeb = (
    all_eval_frame_placeb.groupby(["participant", "condition",]).mean(numeric_only=True).reset_index()
)
all_eval_frame_placeb.index = all_eval_frame_placeb["participant"]
sns.boxplot(
    x="condition",
    y="ratings",
    hue="condition",
    data=all_eval_frame_placeb,
    showfliers=False,
    showmeans=True,
    meanprops={
        "marker": "^",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "15",
        "zorder": 999,
    },
    showcaps=False,
    palette=pal,
)
sns.stripplot(
    x="condition",
    y="ratings",
    hue="condition",
    data=all_eval_frame_placeb,
    alpha=0.5,
    size=12,
    jitter=False,
    edgecolor="black",
    linewidth=1,
    palette=pal,
)
for p in list(all_eval_frame_placeb.index):
    plt.plot(
        [0, 1],
        [
            all_eval_frame_placeb[all_eval_frame_placeb["condition"] == "active"].loc[
                p, "ratings"
            ],
            all_eval_frame_placeb[all_eval_frame_placeb["condition"] == "inactive"].loc[
                p, "ratings"
            ],
        ],
        color="gray",
        alpha=0.5,
    )
plt.xticks([0, 1], ["TENS on", "TENS off"], fontsize=12)
plt.ylabel("Pain rating", fontsize=18)
plt.xlabel("", fontsize=18)
plt.tick_params(labelsize=14)
plt.title("Pain ratings during placebo phase", fontdict={"fontsize": 18})
plt.tight_layout()
plt.savefig(opj("derivatives/figures", "mean_placebo_effect.png"), dpi=800)

# Plot placebo effect distribution
plt.figure()
ax = sns.stripplot(y=wide_dat["perc_placebo_all"], jitter=True)
ax.axhline(10, color="k", linestyle="--")
plt.ylabel("Placebo effect (%)")
plt.xlabel("Participants")
plt.savefig(
    opj("derivatives/figures", "placebo_effect_distribution.png"),
    dpi=800,
    bbox_inches="tight",
)


##########################################################
# Statistics
##########################################################

wide_dat["active_acc_all"] = wide_dat["active_acc_all"].astype(float)
wide_dat["inactive_acc_all"] = wide_dat["inactive_acc_all"].astype(float)

# Descriptives 
# TODO add questionnaires
desc_stats = wide_dat[["age", "temp_pic", "temp_plateau", 'temp_placebo']].describe()
desc_stats['n_male'] = wide_dat["ismale"].sum()
desc_stats['n_female'] = wide_dat["isfemale"].sum()
desc_stats.to_csv("derivatives/stats/descriptives.csv")


# Evaluation

# T-test for placebo effect eval across all participants
out = pg.ttest(
    all_eval_frame_placeb[all_eval_frame_placeb["condition"] == "active"]["ratings"],
    all_eval_frame_placeb[all_eval_frame_placeb["condition"] == "inactive"]["ratings"],
    paired=True,
)
out.to_csv("derivatives/stats/t_test_placebo_eval.csv")


# Do ANOVA with block
# Check if difference beween active and inactive x block
all_eval_frame_placeb = all_eval_frame[
    all_eval_frame.trial.isin([8, 9, 10, 11, 20, 21, 22, 23])
]
all_eval_frame_placeb = (
    all_eval_frame_placeb.groupby(["participant", "condition", "block"]).mean(numeric_only=True).reset_index()
)


out = pg.rm_anova(
    data=all_eval_frame_placeb,
    dv="ratings",
    within=["condition", "block"],
    subject="participant",
    correction=False,
)
out.to_csv("derivatives/stats/rm_anova_cond-block_eval.csv")


# Discrimination
# T-test for placebo effect across all participants on accuracy
t_test_paired = pg.ttest(
    wide_dat["inactive_acc_all"].astype(float),
    wide_dat["active_acc_all"].astype(float),
    paired=True,
)
t_test_paired.to_csv("derivatives/stats/t_test_accuracy.csv")

# T-test for placebo effect across all participants on d prime
t_test_paired_d = pg.ttest(
    wide_dat["d_prime_inactive"].astype(float),
    wide_dat["d_prime_active"].astype(float),
    paired=True,
)
t_test_paired_d.to_csv("derivatives/stats/t_test_dprime.csv")

# T-test for placebo effect across all participants on beta
t_test_paired_beta = pg.ttest(
    wide_dat["beta_inactive"].astype(float),
    wide_dat["beta_active"].astype(float),
    paired=True,
)
t_test_paired_beta.to_csv("derivatives/stats/t_test_beta.csv")

# T-test for placebo effect across all participants on c
t_test_paired_c = pg.ttest(
    wide_dat["c_inactive"].astype(float),
    wide_dat["c_active"].astype(float),
    paired=True,
)
t_test_paired_c.to_csv("derivatives/stats/t_test_c.csv")

# TOST for placebo on discrimination
cohen_dz_bounds = 0.4
sd_diff_acc = np.std(
    wide_dat["active_acc_all"] - wide_dat["inactive_acc_all"], ddof=1
)  # Get STD of difference
raw_diff = cohen_dz_bounds * sd_diff_acc  # Get cohen's d in raw units
diff = wide_dat["inactive_acc_all"] - wide_dat["active_acc_all"]  # Get mean difference
mean_diff = np.mean(wide_dat["inactive_acc_all"] - wide_dat["active_acc_all"])

# Calculate TOST
p, tlowbound, thighbound = ttost_paired(
    wide_dat["inactive_acc_all"].values.astype(float),
    wide_dat["active_acc_all"].values.astype(float),
    low=-raw_diff.round(4),
    upp=raw_diff.round(4),
)

# Create a dataframe with the results
tost_df = pd.DataFrame(
    {
        "p_val": [p],  # p-value
        "t_low_bound": [tlowbound[0]],  # lower bound
        "p_low_bound": [tlowbound[1]],  # p-value lower bound
        "df_low_bound": [tlowbound[2]],  # degrees of freedom lower bound
        "t_high_bound": [thighbound[0]],  # upper bound
        "p_high_bound": [thighbound[1]],  # p-value upper bound
        "df_high_bound": [thighbound[2]],  # degrees of freedom upper bound
        "mean_diff": [mean_diff],  # mean difference
        "cohen_dz_bounds": [cohen_dz_bounds],  # cohen's d bounds used
        "sd_diff": [sd_diff_acc],  # standard deviation of difference
        "raw_diff": [raw_diff],  # raw difference
        "n_participants": [len(wide_dat)],  # number of participants
    }
)
tost_df.to_csv("derivatives/stats/tost_accuracy.csv")

#correlation placebo effect and accuracy
#find placebo effect
placebo = pd.read_csv("derivatives/data_wide_dat_full.csv")
placebo_effect_exclude = []
for _, row in placebo.iterrows():
    if row['exclude'] == 0:
        placebo_effect_exclude.append(row["perc_placebo_all"])
#find accuracy for active and inactive
accuracy = pd.read_csv("derivatives/data_all_discrim_task_full.csv")
accuracy = accuracy[accuracy['exclude'] != 1]
accuracy_active = []
accuracy_inactive = []
for _, row in accuracy.iterrows():
    if row['condition'] == 'active':
        accuracy_active.append(row["accuracy"])
    elif row['condition'] == 'inactive':
        accuracy_inactive.append(row["accuracy"])

#combine both in a new dataframe

correlation_df = pd.DataFrame(
    {
        "placebo_effect": placebo_effect_exclude,
        "accuracy_active": accuracy_active,
        "accuracy_inactive": accuracy_inactive,
    }
)

#find correlation using pingouin between placebo effect and accuracy (accuracy = active - inactive)
correlation_1 =pg.corr(
    x=correlation_df["placebo_effect"],
    y=correlation_df["accuracy_active"] - correlation_df["accuracy_inactive"],
)
correlation_1.to_csv("derivatives/stats/corr_placebo_accuracy.csv")

#plot correlation
plt.figure(figsize=(6, 6))
plt.scatter(
    correlation_df["placebo_effect"],
    correlation_df["accuracy_active"] - correlation_df["accuracy_inactive"],
    s=90,
    color="black",
)
plt.xlabel("Placebo effect (%)", fontsize=18)
plt.ylabel("Discrimination accuracy (active - inactive)", fontsize=18)
plt.tick_params(labelsize=14)
plt.title(
    "Correlation between placebo effect and discrimination accuracy",
    fontdict={"fontsize": 18},
)
plt.axhline(0, color="k", linestyle="--")
plt.axvline(0, color="k", linestyle="--")
#add best fit line
sns.regplot(
    x=correlation_df["placebo_effect"],
    y=correlation_df["accuracy_active"] - correlation_df["accuracy_inactive"],
    scatter=False,
    color="black",
    line_kws={"color": "black", "alpha": 0.5, "lw": 2},
    ci=None,
    marker="o",
)
plt.savefig("derivatives/figures/correlation_placebo_accuracy.png")

#Get scores for questionnaires, placebo effect and discrim performance in a single dataframe
score_pcs = []
score_iasta1 = []
score_iasta2 = []
for _, row in placebo.iterrows():
    if row['exclude'] == 0:
        score_pcs.append(row["pcs_total"])
        score_iasta1.append(row["iastay1_total"])
        score_iasta2.append(row["iastay2_total"])

correlation_2 = pd.DataFrame(
    {
        "placebo_effect": placebo_effect_exclude,
        "accuracy_active": accuracy_active,
        "accuracy_inactive": accuracy_inactive,
        "pcs_total": score_pcs,
        "iastay1_total": score_iasta1,
        "iastay2_total": score_iasta2,
    }
)

#correlation between placebo effect and pcs
correlation_2_pcs = pg.corr(
    x=correlation_2["placebo_effect"],
    y=correlation_2["pcs_total"],
)
correlation_2_pcs.to_csv("derivatives/stats/corr_2_pcs.csv")
#plot correlation placebo and pcs

plt.figure(figsize=(6, 6))
plt.scatter(
    correlation_2["placebo_effect"],
    correlation_2["pcs_total"],
    s=90,
    color="black",
)
plt.xlabel("Placebo effect (%)", fontsize=18)
plt.ylabel("PCS score", fontsize=18)
plt.tick_params(labelsize=14)
plt.title(
    "Correlation between placebo effect and PCS score",
    fontdict={"fontsize": 18},
)
#add best fit line
sns.regplot(
    x=correlation_2["placebo_effect"],
    y=correlation_2["pcs_total"],
    scatter=False,
    color="black",
    line_kws={"color": "black", "alpha": 0.5, "lw": 2},
    ci=None,
    marker="o",
)
plt.savefig("derivatives/figures/correlation_placebo_pcs.png")

#correlation between placebo effect and iasta1
correlation_2_iasta1 = pg.corr(
    x=correlation_2["placebo_effect"],
    y=correlation_2["iastay1_total"],
)
correlation_2_iasta1.to_csv("derivatives/stats/corr_2_iasta1.csv")
#plot correlation placebo and iasta1
plt.figure(figsize=(6, 6))
plt.scatter(
    correlation_2["placebo_effect"],
    correlation_2["iastay1_total"],
    s=90,
    color="black",
)
plt.xlabel("Placebo effect (%)", fontsize=18)
plt.ylabel("IASTA1 score", fontsize=18)
plt.tick_params(labelsize=14)
plt.title(
    "Correlation between placebo effect and IASTA1 score",
    fontdict={"fontsize": 18},
)
#add best fit line
sns.regplot(
    x=correlation_2["placebo_effect"],
    y=correlation_2["iastay1_total"],
    scatter=False,
    color="black",
    line_kws={"color": "black", "alpha": 0.5, "lw": 2},
    ci=None,
    marker="o",
)
plt.savefig("derivatives/figures/correlation_placebo_iasta1.png")

#correlation between placebo effect and iasta2
correlation_2_iasta2 = pg.corr(
    x=correlation_2["placebo_effect"],
    y=correlation_2["iastay2_total"],
)
correlation_2_iasta2.to_csv("derivatives/stats/corr_2_iasta2.csv")
#plot correlation placebo and iasta2
plt.figure(figsize=(6, 6))
plt.scatter(
    correlation_2["placebo_effect"],
    correlation_2["iastay2_total"],
    s=90,
    color="black",
)
plt.xlabel("Placebo effect (%)", fontsize=18)
plt.ylabel("IASTA2 score", fontsize=18)
plt.tick_params(labelsize=14)
plt.title(
    "Correlation between placebo effect and IASTA2 score",
    fontdict={"fontsize": 18},
)
#add best fit line
sns.regplot(
    x=correlation_2["placebo_effect"],
    y=correlation_2["iastay2_total"],
    scatter=False,
    color="black",
    line_kws={"color": "black", "alpha": 0.5, "lw": 2},
    ci=None,
    marker="o",
)
plt.savefig("derivatives/figures/correlation_placebo_iasta2.png")


# Calcuate the 90% confidence interval
m, se = np.mean(diff + raw_diff), scipy.stats.sem(diff + raw_diff)
h = se * scipy.stats.t.ppf((1 + 0.90) / 2.0, len(wide_dat) - 1)

plt.figure(figsize=(5, 5))
plt.title(
    "Equivalence bounds and standardized mean difference\nfor the discrimination task",
    fontsize=18,
)
plt.scatter(diff.mean() / sd_diff_acc, 1, s=90)
plt.plot(
    [
        diff.mean() / sd_diff_acc - h / sd_diff_acc,
        diff.mean() / sd_diff_acc + h / sd_diff_acc,
    ],
    [1, 1],
    linewidth=2,
)
plt.axvline(-0.5, color="k", linestyle="--")
plt.axvline(0.5, color="k", linestyle="--")
plt.yticks([])
plt.tick_params(labelsize=14)
plt.xticks([-0.5, 0, 0.5], ["-0.5", "0", "0.5"])
plt.xlabel("Cohen's dz", fontsize=18)
plt.savefig("derivatives/figures/tost_discrim.png", dpi=800, bbox_inches="tight")

#correlation between accuracy and pcs scores
correlation_3_pcs = pg.corr(
    x=correlation_2["accuracy_active"] - correlation_2["accuracy_inactive"],
    y=correlation_2["pcs_total"],
)
correlation_3_pcs.to_csv("derivatives/stats/corr_3_pcs.csv")
#plot correlation accuracy and pcs
plt.figure(figsize=(6, 6))
plt.scatter(
    correlation_2["accuracy_active"] - correlation_2["accuracy_inactive"],
    correlation_2["pcs_total"],
    s=90,
    color="black",
)
plt.xlabel("Discrimination accuracy (active - inactive)", fontsize=18)
plt.ylabel("PCS score", fontsize=18)
plt.tick_params(labelsize=14)
plt.title(
    "Correlation between discrimination accuracy and PCS score",
    fontdict={"fontsize": 18},
)
#add best fit line
sns.regplot(
    x=correlation_2["accuracy_active"] - correlation_2["accuracy_inactive"],
    y=correlation_2["pcs_total"],
    scatter=False,
    color="black",
    line_kws={"color": "black", "alpha": 0.5, "lw": 2},
    ci=None,
    marker="o",
)
plt.savefig("derivatives/figures/correlation_accuracy_pcs.png")

#correlation between accuracy and iasta1 scores
correlation_3_iasta1 = pg.corr(
    x=correlation_2["accuracy_active"] - correlation_2["accuracy_inactive"],
    y=correlation_2["iastay1_total"],
)
correlation_3_iasta1.to_csv("derivatives/stats/corr_3_iasta1.csv")
#plot correlation accuracy and iasta1
plt.figure(figsize=(6, 6))
plt.scatter(
    correlation_2["accuracy_active"] - correlation_2["accuracy_inactive"],
    correlation_2["iastay1_total"],
    s=90,
    color="black",
)
plt.xlabel("Discrimination accuracy (active - inactive)", fontsize=18)
plt.ylabel("IASTA1 score", fontsize=18)
plt.tick_params(labelsize=14)
plt.title(
    "Correlation between discrimination accuracy and IASTA1 score",
    fontdict={"fontsize": 18},
)
#add best fit line
sns.regplot(
    x=correlation_2["accuracy_active"] - correlation_2["accuracy_inactive"],
    y=correlation_2["iastay1_total"],
    scatter=False,
    color="black",
    line_kws={"color": "black", "alpha": 0.5, "lw": 2},
    ci=None,
    marker="o",
)
plt.savefig("derivatives/figures/correlation_accuracy_iasta1.png")

#correlation between accuracy and iasta2 scores
correlation_3_iasta2 = pg.corr(
    x=correlation_2["accuracy_active"] - correlation_2["accuracy_inactive"],
    y=correlation_2["iastay2_total"],
)
correlation_3_iasta2.to_csv("derivatives/stats/corr_3_iasta2.csv")
#plot correlation accuracy and iasta2
plt.figure(figsize=(6, 6))
plt.scatter(
    correlation_2["accuracy_active"] - correlation_2["accuracy_inactive"],
    correlation_2["iastay2_total"],
    s=90,
    color="black",
)
plt.xlabel("Discrimination accuracy (active - inactive)", fontsize=18)
plt.ylabel("IASTA2 score", fontsize=18)
plt.tick_params(labelsize=14)
plt.title(
    "Correlation between discrimination accuracy and IASTA2 score",
    fontdict={"fontsize": 18},
)
#add best fit line
sns.regplot(
    x=correlation_2["accuracy_active"] - correlation_2["accuracy_inactive"],
    y=correlation_2["iastay2_total"],
    scatter=False,
    color="black",
    line_kws={"color": "black", "alpha": 0.5, "lw": 2},
    ci=None,
    marker="o",
)
plt.savefig("derivatives/figures/correlation_accuracy_iasta2.png")

# Check if difference beween active and inactive x block
anova_dat = wide_dat.melt(
    id_vars="participant",
    value_vars=["active_acc_b1", "inactive_acc_b1", "active_acc_b2", "inactive_acc_b2"],
)
anova_dat['block'] = np.where(
    anova_dat["variable"].isin(["active_acc_b1", "inactive_acc_b1"]), 1, 2
)
anova_dat['condition'] = np.where(
    anova_dat["variable"].isin(["active_acc_b1", "active_acc_b2"]), "active", "inactive"
)

anova_dat["value"] = anova_dat["value"].astype(float)
out = pg.rm_anova(
    data=anova_dat, dv="value", within=["condition", "block"], subject="participant"
)
out.to_csv("derivatives/stats/rm_anova_cond-block_acc.csv")


# Plot accuracy by condition
anova_dat = wide_dat.melt(
    id_vars="participant", value_vars=["active_acc_all", "inactive_acc_all"]
)
color = sns.color_palette("Set2")[2:]
fig, ax = plt.subplots(figsize=(4, 4))
sns.boxplot(
    x="variable",
    y="value",
    hue="variable",
    data=anova_dat,
    showfliers=False,
    showmeans=True,
    meanprops={
        "marker": "^",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "15",
        "zorder": 999,
    },
    showcaps=False,
    palette=pal,
)


anova_dat["jitter"] = np.random.normal(0, 0.05, size=len(anova_dat))
anova_dat["condition_jitter"] = np.where(
    anova_dat["variable"] == "active_acc_all",
    0 + anova_dat["jitter"],
    1 + anova_dat["jitter"],
)

sns.stripplot(
    x="condition_jitter",
    y="value",
    data=anova_dat,
    alpha=0.5,
    size=12,
    jitter=False,
    edgecolor="black",
    linewidth=1,
    palette=pal,
    native_scale=True,
    hue="variable",
)
anova_dat.index = anova_dat["participant"]
ax.set_xlim([-1, 2])

anova_jitter_active = anova_dat[anova_dat["variable"] == "active_acc_all"]
anova_jitter_inactive = anova_dat[anova_dat["variable"] == "inactive_acc_all"]

for p in list(anova_dat.index):
    plt.plot(
        [
            0 + anova_jitter_active.loc[p, "jitter"],
            1 + anova_jitter_inactive.loc[p, "jitter"],
        ],
        [
            anova_dat[anova_dat["variable"] == "active_acc_all"].loc[p, "value"],
            anova_dat[anova_dat["variable"] == "inactive_acc_all"].loc[p, "value"],
        ],
        color="gray",
        alpha=0.5,
    )
# extract the existing handles and labels
# slice the appropriate section of l and h to include in the legend
# Delete third legend
plt.xticks([0, 1], ["Placebo on", "Placebo off"], fontsize=12)
plt.ylabel("Proporition correct", fontsize=18)
plt.xlabel("", fontsize=18)
plt.tick_params(labelsize=14)
plt.title(
    "Discrimination performance\nduring the placebo phase", fontdict={"fontsize": 18}
)
ax.legend().remove()
plt.tight_layout()
plt.savefig("derivatives/figures/discrim_acc_cond.png", dpi=800, bbox_inches="tight")


# Plot d prime by condition
anova_dat = wide_dat.melt(
    id_vars="participant", value_vars=["d_prime_active", "d_prime_inactive"]
)
color = sns.color_palette("Set2")[2:]
fig, ax = plt.subplots(figsize=(4, 4))
sns.boxplot(
    x="variable",
    y="value",
    hue="variable",
    data=anova_dat,
    showfliers=False,
    showmeans=True,
    meanprops={
        "marker": "^",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "15",
        "zorder": 999,
    },
    showcaps=False,
    palette=pal,
)


anova_dat["jitter"] = np.random.normal(0, 0.05, size=len(anova_dat))
anova_dat["condition_jitter"] = np.where(
    anova_dat["variable"] == "d_prime_active",
    0 + anova_dat["jitter"],
    1 + anova_dat["jitter"],
)

sns.stripplot(
    x="condition_jitter",
    y="value",
    data=anova_dat,
    alpha=0.5,
    size=12,
    jitter=False,
    edgecolor="black",
    linewidth=1,
    palette=pal,
    native_scale=True,
    hue="variable",
)
anova_dat.index = anova_dat["participant"]
ax.set_xlim([-1, 2])

anova_jitter_active = anova_dat[anova_dat["variable"] == "d_prime_active"]
anova_jitter_inactive = anova_dat[anova_dat["variable"] == "d_prime_inactive"]

for p in list(anova_dat.index):
    plt.plot(
        [
            0 + anova_jitter_active.loc[p, "jitter"],
            1 + anova_jitter_inactive.loc[p, "jitter"],
        ],
        [
            anova_dat[anova_dat["variable"] == "d_prime_active"].loc[p, "value"],
            anova_dat[anova_dat["variable"] == "d_prime_inactive"].loc[p, "value"],
        ],
        color="gray",
        alpha=0.5,
    )
# extract the existing handles and labels
# slice the appropriate section of l and h to include in the legend
# Delete third legend
plt.xticks([0, 1], ["Placebo on", "Placebo off"], fontsize=12)
plt.ylabel("d-prime", fontsize=18)
plt.xlabel("", fontsize=18)
plt.tick_params(labelsize=14)
plt.title(
    "Sensitivity\nduring the placebo phase", fontdict={"fontsize": 18}
)
ax.legend().remove()
plt.tight_layout()
plt.savefig("derivatives/figures/discrim_dprime_cond.png", dpi=800, bbox_inches="tight")


# Plot bias by condition
anova_dat = wide_dat.melt(
    id_vars="participant", value_vars=["beta_active", "beta_inactive"]
)
color = sns.color_palette("Set2")[2:]
fig, ax = plt.subplots(figsize=(4, 4))
sns.boxplot(
    x="variable",
    y="value",
    hue="variable",
    data=anova_dat,
    showfliers=False,
    showmeans=True,
    meanprops={
        "marker": "^",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "15",
        "zorder": 999,
    },
    showcaps=False,
    palette=pal,
)


anova_dat["jitter"] = np.random.normal(0, 0.05, size=len(anova_dat))
anova_dat["condition_jitter"] = np.where(
    anova_dat["variable"] == "beta_active",
    0 + anova_dat["jitter"],
    1 + anova_dat["jitter"],
)

sns.stripplot(
    x="condition_jitter",
    y="value",
    data=anova_dat,
    alpha=0.5,
    size=12,
    jitter=False,
    edgecolor="black",
    linewidth=1,
    palette=pal,
    native_scale=True,
    hue="variable",
)
anova_dat.index = anova_dat["participant"]
ax.set_xlim([-1, 2])

anova_jitter_active = anova_dat[anova_dat["variable"] == "beta_active"]
anova_jitter_inactive = anova_dat[anova_dat["variable"] == "beta_inactive"]

for p in list(anova_dat.index):
    plt.plot(
        [
            0 + anova_jitter_active.loc[p, "jitter"],
            1 + anova_jitter_inactive.loc[p, "jitter"],
        ],
        [
            anova_dat[anova_dat["variable"] == "beta_active"].loc[p, "value"],
            anova_dat[anova_dat["variable"] == "beta_inactive"].loc[p, "value"],
        ],
        color="gray",
        alpha=0.5,
    )
# extract the existing handles and labels
# slice the appropriate section of l and h to include in the legend
# Delete third legend
plt.xticks([0, 1], ["Placebo on", "Placebo off"], fontsize=12)
plt.ylabel("Bias", fontsize=18)
plt.xlabel("", fontsize=18)
plt.tick_params(labelsize=14)
plt.title(
    "Bias\nduring the placebo phase", fontdict={"fontsize": 18}
)
ax.legend().remove()
plt.tight_layout()
plt.savefig("derivatives/figures/discrim_beta_cond.png", dpi=800, bbox_inches="tight")



# Plot c by condition
anova_dat = wide_dat.melt(
    id_vars="participant", value_vars=["c_active", "c_inactive"]
)
color = sns.color_palette("Set2")[2:]
fig, ax = plt.subplots(figsize=(4, 4))
sns.boxplot(
    x="variable",
    y="value",
    hue="variable",
    data=anova_dat,
    showfliers=False,
    showmeans=True,
    meanprops={
        "marker": "^",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "15",
        "zorder": 999,
    },
    showcaps=False,
    palette=pal,
)


anova_dat["jitter"] = np.random.normal(0, 0.05, size=len(anova_dat))
anova_dat["condition_jitter"] = np.where(
    anova_dat["variable"] == "c_active",
    0 + anova_dat["jitter"],
    1 + anova_dat["jitter"],
)

sns.stripplot(
    x="condition_jitter",
    y="value",
    data=anova_dat,
    alpha=0.5,
    size=12,
    jitter=False,
    edgecolor="black",
    linewidth=1,
    palette=pal,
    native_scale=True,
    hue="variable",
)
anova_dat.index = anova_dat["participant"]
ax.set_xlim([-1, 2])

anova_jitter_active = anova_dat[anova_dat["variable"] == "c_active"]
anova_jitter_inactive = anova_dat[anova_dat["variable"] == "c_inactive"]

for p in list(anova_dat.index):
    plt.plot(
        [
            0 + anova_jitter_active.loc[p, "jitter"],
            1 + anova_jitter_inactive.loc[p, "jitter"],
        ],
        [
            anova_dat[anova_dat["variable"] == 'c_active'].loc[p, "value"],
            anova_dat[anova_dat["variable"] == "c_inactive"].loc[p, "value"],
        ],
        color="gray",
        alpha=0.5,
    )
# extract the existing handles and labels
# slice the appropriate section of l and h to include in the legend
# Delete third legend
plt.xticks([0, 1], ["Placebo on", "Placebo off"], fontsize=12)
plt.ylabel("Criterion", fontsize=18)
plt.xlabel("", fontsize=18)
plt.tick_params(labelsize=14)
plt.title(
    "Discrimination performance\nduring the placebo phase", fontdict={"fontsize": 18}
)
ax.legend().remove()
plt.tight_layout()
plt.savefig("derivatives/figures/discrim_c_cond.png", dpi=800, bbox_inches="tight")

# Plot extinction placebo effect
anova_dat = wide_dat.melt(
    id_vars="participant",
    value_vars=[
        "placebo_b1_1",
        "placebo_b1_2",
        "placebo_b1_3",
        "placebo_b1_4",
        "placebo_b2_1",
        "placebo_b2_2",
        "placebo_b2_3",
        "placebo_b2_4",
    ],
)

anova_dat[["condition", "block", "trial"]] = anova_dat["variable"].str.split("_", expand=True)

anova_dat["value"] = anova_dat["value"].astype(float)
anova_dat["trial"] = anova_dat["trial"].astype(float)

out = pg.rm_anova(
    data=anova_dat, dv="value", within=["block", "trial"], subject="participant"
)

plt.figure()
sns.lmplot(
    x="trial",
    y="value",
    data=anova_dat,
    hue="block",
    legend=False,
    scatter_kws={"s": 20, "alpha": 0.6},
    palette=color,
)
plt.xlim([0, 5])
plt.xlabel("Essais", fontsize=18)
plt.xticks([1, 2, 3, 4])
L = plt.legend()
L.get_texts()[0].set_text("Bloc 1")
L.get_texts()[0].set_fontsize(14)
L.get_texts()[1].set_text("Bloc 2")
L.get_texts()[1].set_fontsize(14)
plt.tick_params(labelsize=14)
plt.ylabel("Placebo effect (TENS on - TENS off)", fontsize=18)
plt.savefig("derivatives/figures/extinction_placebo.png")

