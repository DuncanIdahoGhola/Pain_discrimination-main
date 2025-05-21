# %%
import numpy as np
import pandas as pd
import os
from os.path import join as opj
import scipy
import matplotlib.pyplot as plt
from psychopy import data
import seaborn as sns
from statsmodels.stats.weightstats import ttost_paired
from scipy import stats
import pingouin as pg
import urllib.request
import matplotlib.font_manager as fm
from matplotlib.patches import Patch


data_dir = "C:\\Users\\labmp\\Documents\\GitHub\\Pain_discrimination\\Venv\\Data"
# Find participants
participants = [p for p in os.listdir(data_dir) if "sub-" in p]
participants.sort()
if not os.path.exists("derivatives"):
    os.mkdir("derivatives")


# Exclude participants with perfect discrimination
exclude_perfect = 0

# Exclude participants with less than X% placebo effect (0 = negative placebo, -100 keep everyone)
exclude_placebo = -100

exclude_custom = [
    "sub-001",
    "sub-002",
    "sub-003",
    "sub-005",
    "sub-006",
    "sub-008",
    "sub-009",
    "sub-033",
]

# Get nice font
urllib.request.urlretrieve(
    "https://github.com/gbif/analytics/raw/master/fonts/Arial%20Narrow.ttf",
    "arialnarrow.ttf",
)
fm.fontManager.addfont("arialnarrow.ttf")
prop = fm.FontProperties(fname="arialnarrow.ttf")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = prop.get_name()


# Loop participants
all_active, all_thresholds, all_plateaus = [], [], []
all_eval_frames, all_discrim_task = [], []
all_discrim_task_long = []

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
deriv_dir = "C:\\Users\\labmp\\Documents\\GitHub\\Pain_discrimination\\Venv\\derivatives"

for p in participants:
    part_path = opj(data_dir, p)
    deriv_path = opj(deriv_dir, p)
    if not os.path.exists(deriv_path):
        os.mkdir(deriv_path)

    # ###################################################
    # # QUEST
    # ###################################################
    quest_dir = opj(part_path, "QUEST")
    # Find main file
    quest_file = [f for f in os.listdir(quest_dir) if ".csv" in f and "trial" not in f]

    if len(quest_file) > 1:  # Make sure there is just one
        quest_file = [quest_file[-1]]
    # Make sure there is just one
    for idx, f in enumerate(quest_file):
        quest = pd.read_csv(os.path.join(part_path, "QUEST", quest_file[idx]))
        plateau = quest["temp_plateau"].values[0]
        quest = quest[quest["pic_response"] != "None"]
        trials_num = quest["trials.thisRepN"].dropna().values

        # Set missing value for specific participant
        if "sub-010" in p:
            quest.loc[0, "pic_sent"] = None

        # Drop NaNs and extract values
        quest.loc[quest["pic_response"].isna(), "pic_sent"] = np.nan
        quest_intensities = quest["pic_sent"].dropna().values
        quest_detections = quest["pic_response"].dropna().values


        print(quest[["pic_sent", "pic_response"]])
        print(quest["pic_response"])

        if "sub-012" in p:
            thresh = 0.5 + plateau
        elif int(p[4:7]) < 7:
            thresh = quest["threshold"].dropna().values[0]
        else:
            thresh = quest["mean_6_reversals"].dropna().values[0] + plateau

        combinedInten = quest_intensities.astype(float)
        combinedResp = quest_detections.astype(float)

        fit = data.FitWeibull(
            combinedInten, combinedResp, expectedMin=0.5, guess=[0.2, 0.5]
        )
        smoothInt = np.arange(min(combinedInten), max(combinedInten), 0.001)
        smoothResp = fit.eval(smoothInt)

        plt.figure(figsize=(5, 5))
        plt.plot(smoothInt, smoothResp, "-")
        if len(quest_file) > 1:
            thresh = thresh + plateau
        # pylab.plot([thresh, thresh],[0,0.8],'--'); pylab.plot([0, thresh], [0.8,0.8],'--')
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
        for f in os.listdir(part_path)
        if "maintask" in f and "trial" not in f and ".csv" in f
    ]
    assert len(main_file) == 1  # Make sure there is just one

    main = pd.read_csv(os.path.join(part_path, main_file[0]))

    # Some files are saved with a different delimiter
    if len(main.columns) == 1:
        main = pd.read_csv(os.path.join(part_path, main_file[0]), sep=";")

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
    eval_task = eval_task[eval_task["ratingScale.response"].isna() == False]
    eval_task_active_noplacebo = eval_task[
        (eval_task["condition"] == "active")
        & (eval_task["loop_eval_discri.thisRepN"].isna())
    ].reset_index(drop=True)
    eval_task_inactive_noplacebo = eval_task[
        (eval_task["condition"] == "inactive")
        & (eval_task["loop_eval_discri.thisRepN"].isna())
    ].reset_index(drop=True)

    eval_task_active_placebo = eval_task[
        (eval_task["condition"] == "active")
        & (~eval_task["loop_eval_discri.thisRepN"].isna())
    ].reset_index(drop=True)
    eval_task_inactive_placebo = eval_task[
        (eval_task["condition"] == "inactive")
        & ~(eval_task["loop_eval_discri.thisRepN"].isna())
    ].reset_index(drop=True)

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

    placebo_diff = np.mean(
        eval_task_inactive_placebo["ratingScale.response"].values
        - eval_task_active_placebo["ratingScale.response"].values
    )
    placebo_diff_perc = (
        placebo_diff
        / np.mean(eval_task_inactive_placebo["ratingScale.response"].values)
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

    thresh = main["temp_pic_set"].values[0]

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

    out_frame = pd.DataFrame(
        dict(
            ratings=list(eval_task_active["ratingScale.response"].values)
            + list(eval_task_inactive["ratingScale.response"].values),
            condition=["active"] * len(eval_task_active)
            + ["inactive"] * len(eval_task_inactive),
            trial=list(np.arange(len(eval_task_active))) * 2,
            participant=p[:7],
        )
    )
    all_eval_frames.append(out_frame)

    # Plot temperature
    temp_files = [
        f
        for f in os.listdir(part_path)
        if "maintask" in f and "temp_trial_eval" in f and ".csv" in f
    ]
    plt.figure(figsize=(5, 5))
    for trial in temp_files:
        temp = pd.read_csv(opj(part_path, trial))
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
    for presence, response in zip(
        discrim_task["pic_presence"].values, discrim_task["pic_response"].values
    ):
        if presence == "pic-present" and response == 1:
            accurate.append(1)
        elif presence == "pic-absent" and response == 0:
            accurate.append(1)
        elif presence == "pic-present" and response == 0:
            accurate.append(0)
        elif presence == "pic-absent" and response == 1:
            accurate.append(0)

    discrim_task["accuracy"] = accurate

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
        for f in os.listdir(part_path)
        if "maintask" in f and "temp_trial_pic" in f and ".csv" in f
    ]
    plt.figure(figsize=(5, 5))
    for trial in temp_files:
        temp = pd.read_csv(opj(part_path, trial))
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

all_eval_frame = pd.concat(all_eval_frames)
all_discrim_task = pd.concat(all_discrim_task)
all_discrim_task_long = pd.concat(all_discrim_task_long)
wide_dat["participant"] = wide_dat.index.values
wide_dat["participant"] = wide_dat.index.values
wide_dat["part_id"], _ = wide_dat["participant"].str.split("_", 2).str
wide_dat["part_id"] = wide_dat["participant"].str.split("_", n=1).str[1]


if exclude_perfect:
    wide_dat_perf = (
        list(wide_dat[wide_dat["acc_all"] == 1]["participant"]) + exclude_custom
    )
    wide_dat = wide_dat[~wide_dat["participant"].isin(wide_dat_perf)]
    all_eval_frame = all_eval_frame[~all_eval_frame["participant"].isin(wide_dat_perf)]
    all_discrim_task = all_discrim_task[
        ~all_discrim_task["participant"].isin(wide_dat_perf)
    ]


n_exclude_placebo = np.sum(wide_dat["perc_placebo_all"] < exclude_placebo)
wide_dat_placebo = (
    list(wide_dat[wide_dat["perc_placebo_all"] < exclude_placebo]["part_id"])
    + exclude_custom
)
wide_dat = wide_dat[~wide_dat["part_id"].isin(wide_dat_placebo)]
all_eval_frame = all_eval_frame[~all_eval_frame["participant"].isin(wide_dat_placebo)]
all_discrim_task = all_discrim_task[
    ~all_discrim_task["participant"].isin(wide_dat_placebo)
]

fig, ax = plt.subplots(figsize=(7, 4))
sns.pointplot(
    x="trial",
    y="ratings",
    hue="condition",
    data=all_eval_frame,
    errorbar="se",
    alpha=0.9,
    axes=ax,
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
plt.savefig(opj("derivatives", "group_eval_task.png"), dpi=800, bbox_inches="tight")


# Same but with discrimination interspersed
all_discrim_task_long_avg = (
    all_discrim_task_long.groupby(["participant", "condition", "actual_trial"])
    .mean()
    .reset_index()
)

all_discrim_task_long_avg["actual_trial"] = (
    all_discrim_task_long_avg["actual_trial"].astype(int) - 0.8
)
all_eval_frame["trial"].astype(int)

all_eval_frame_avg = (
    all_eval_frame.groupby(["participant", "condition", "trial"]).mean().reset_index()
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
plt.savefig(opj("derivatives", "group_discrim_task.png"), dpi=800, bbox_inches="tight")


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
plt.savefig(opj("derivatives", "group_eval_task.png"), dpi=800)


fig, ax = plt.subplots(figsize=(4, 4))
all_eval_frame_placeb = all_eval_frame[
    all_eval_frame.trial.isin([8, 9, 10, 11, 20, 21, 22, 23])
]
all_eval_frame_placeb = (
    all_eval_frame_placeb.groupby(["participant", "condition"]).mean().reset_index()
)
all_eval_frame_placeb.index = all_eval_frame_placeb["participant"]
sns.boxplot(
    x="condition",
    y="ratings",
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
plt.xticks([0, 1], ["Placebo on", "Placebo off"], fontsize=12)
plt.ylabel("Pain rating", fontsize=18)
plt.xlabel("", fontsize=18)
plt.tick_params(labelsize=14)
plt.title("Pain ratings during placebo phase", fontdict={"fontsize": 18})
plt.tight_layout()
plt.savefig(opj("derivatives", "mean_placebo_effect.png"), dpi=800)


pg.ttest(
    all_eval_frame_placeb[all_eval_frame_placeb["condition"] == "active"]["ratings"],
    all_eval_frame_placeb[all_eval_frame_placeb["condition"] == "inactive"]["ratings"],
    paired=True,
)

val = (
    all_eval_frame_placeb[all_eval_frame_placeb["condition"] == "active"]["ratings"]
    - all_eval_frame_placeb[all_eval_frame_placeb["condition"] == "inactive"]["ratings"]
)
cohen_dz = val.mean() / val.std()
# Evaluation task
fig, ax = plt.subplots()
sns.pointplot(
    x="condition",
    y="accuracy",
    data=all_discrim_task,
    color="gray",
    alpha=0.5,
    zorder=0,
    legend=False,
    axes=ax,
)
sns.swarmplot(
    x="condition",
    y="accuracy",
    data=all_discrim_task,
    hue="condition",
    size=8,
    zorder=10,
    palette="Set2",
    axes=ax,
)

plt.axhline(0.5, color="k", linestyle="--", alpha=0.5)
plt.axhline(1, color="k", linestyle="--", alpha=0.5)
plt.ylim([0.4, 1.1])
plt.ylabel("Proportion de réponses correctes", fontsize=18)

plt.xlabel("Condition", fontsize=18)
plt.tick_params(labelsize=16)
l, h = ax.get_legend_handles_labels()
ax.legend(l, ["Actif", "Inactif"], fontsize=14, title="", loc="lower left")


plt.savefig(opj("derivatives", "group_discrim_task.png"))


plt.figure()
ax = sns.stripplot(y=wide_dat["perc_placebo_all"], jitter=True)
ax.axhline(10, color="k", linestyle="--")

##########################################################
# Get sociodemo
##########################################################

wide_dat["part_id"], _ = wide_dat["participant"].str.split("_", 2).str

wide_dat.index = wide_dat["part_id"]

socio = pd.read_csv("sociodemo.csv")

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


##########################################################
# Statistics
##########################################################


wide_dat.to_csv("derivatives/summary_allpart_wide.csv")

# Exclude participants if necessary
# Descriptives
wide_dat["mean_age"] = wide_dat["age"].mean()
wide_dat["sd_age"] = wide_dat["age"].std()
wide_dat["min_age"] = wide_dat["age"].min()
wide_dat["max_age"] = wide_dat["age"].max()


# T-test for placebo effect across all participants
t, p_val = stats.ttest_rel(
    wide_dat["average_placebo_eval_active"], wide_dat["average_placebo_eval_inactive"]
)

wide_dat["ttest_dep_placebo_eval_t"] = t
wide_dat["ttest_dep_placebo_eval_p"] = p_val
wide_dat["ttest_dep_placebo_eval_df"] = (
    wide_dat["average_placebo_eval_active"].shape[0] - 1
)

# TOST for placebo on discrimination
cohen_dz_bounds = 0.46
sd_diff_acc = np.std(
    wide_dat["active_acc_all"] - wide_dat["inactive_acc_all"], ddof=1
)  # Get STD of difference
raw_diff = cohen_dz_bounds * sd_diff_acc  # Get cohen's d in raw units
diff = wide_dat["inactive_acc_all"] - wide_dat["active_acc_all"]  # Get mean difference
mean_diff = np.mean(wide_dat["inactive_acc_all"] - wide_dat["active_acc_all"])


stats.pearsonr(wide_dat["active_acc_all"], wide_dat["inactive_acc_all"])
p, tlowbound, thighbound = ttost_paired(
    wide_dat["inactive_acc_all"].values.astype(float),
    wide_dat["active_acc_all"].values.astype(float),
    low=-raw_diff.round(4),
    upp=raw_diff.round(4),
)


stats.pearsonr(wide_dat["active_acc_all"], wide_dat["inactive_acc_all"])
p, tlowbound, thighbound = ttost_paired(
    wide_dat["inactive_acc_all"].values.astype(float),
    wide_dat["active_acc_all"].values.astype(float),
    low=-raw_diff.round(4),
    upp=raw_diff.round(4),
)


pg.ttest(
    wide_dat["inactive_acc_all"].astype(float),
    wide_dat["active_acc_all"].astype(float),
    paired=True,
)

wide_dat[["inactive_acc_all", "active_acc_all"]]

# Manual correction for missed trials
wide_dat.loc["sub-018", "inactive_acc_all"]
wide_dat.loc["sub-018", "active_acc_all"]

# calculate confidence interval u

stats.norm.interval(
    0.95, loc=mean_diff - raw_diff, scale=sd_diff_acc / np.sqrt(len(wide_dat))
)

diff_z = diff.mean() / sd_diff_acc

m, se = np.mean(diff + raw_diff), scipy.stats.sem(diff + raw_diff)
h = se * scipy.stats.t.ppf((1 + 0.90) / 2.0, len(wide_dat) - 1)

np.mean(wide_dat["inactive_acc_all"])
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
plt.axvline(-0.46, color="k", linestyle="--")
plt.axvline(0.46, color="k", linestyle="--")
plt.yticks([])
plt.tick_params(labelsize=14)
plt.xticks([-0.5, 0, 0.5], ["-0.5", "0", "0.5"])
plt.xlabel("Cohen's dz", fontsize=18)
plt.savefig("derivatives/tost_discrim.png", dpi=800, bbox_inches="tight")


np.std(wide_dat["inactive_acc_all"])
np.std(wide_dat["active_acc_all"])
len

plt.figure()
plt.scatter(np.mean(wide_dat["inactive_acc_all"] - wide_dat["active_acc_all"]), 1)
plt.axvline(-raw_diff, color="k", linestyle="--")
plt.axvline(raw_diff, color="k", linestyle="--")


# pg.tost

# wide_dat['tost_t_active_higher'], wide_dat['tost_p_active_higher'], wide_dat['tost_df'] = t1
# wide_dat['tost_t_active_lower'], wide_dat['tost_p_active_lowe'], wide_dat['tost_df'] = t2

# Check if difference beween active and inactive x block
anova_dat = wide_dat.melt(
    id_vars="participant",
    value_vars=["active_acc_b1", "inactive_acc_b1", "active_acc_b2", "inactive_acc_b2"],
)

anova_dat["condition"], _, anova_dat["block"] = (
    anova_dat["variable"].str.split("_", 2).str
)
anova_dat["value"] = anova_dat["value"].astype(float)

out = pg.rm_anova(
    data=anova_dat, dv="value", within=["condition", "block"], subject="participant"
)

out.to_csv("derivatives/rm_anova_cond*block_acc.csv")

wide_dat["rm_anova_cond*block_acc_Fcond"] = out["F"].values[0]
wide_dat["rm_anova_cond*block_acc_Fblock"] = out["F"].values[1]
wide_dat["rm_anova_cond*block_acc_Finter"] = out["F"].values[2]

wide_dat["rm_anova_cond*block_acc_pcond"] = out["p-unc"].values[0]
wide_dat["rm_anova_cond*block_acc_pblock"] = out["p-unc"].values[1]
wide_dat["rm_anova_cond*block_acc_pinter"] = out["p-unc"].values[2]


anova_dat = wide_dat.melt(
    id_vars="participant", value_vars=["active_acc_all", "inactive_acc_all"]
)


color = sns.color_palette("Set2")[2:]
fig, ax = plt.subplots(figsize=(4, 4))
sns.boxplot(
    x="variable",
    y="value",
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

plt.savefig("derivatives/acc_cond_block.png", dpi=800, bbox_inches="tight")


wide_dat["diff_acc"] = wide_dat["active_acc_all"] - wide_dat["inactive_acc_all"]
wide_dat.diff_acc = wide_dat.diff_acc.astype(float)
wide_dat.perc_placebo_all = wide_dat.perc_placebo_all.astype(float)
plt.figure()
sns.regplot(x="perc_placebo_all", y="diff_acc", data=wide_dat)
scipy.stats.pearsonr(wide_dat["perc_placebo_all"], wide_dat["diff_acc"])

# Extinction placebo effect
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


_, anova_dat["block"], anova_dat["trial"] = anova_dat["variable"].str.split("_", 2).str

out = pg.rm_anova(
    data=anova_dat, dv="value", within=["block", "trial"], subject="participant"
)

anova_dat["value"] = anova_dat["value"].astype(float)
anova_dat["trial"] = anova_dat["trial"].astype(float)

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
plt.ylabel("Effet placebo (Inactif - actif)", fontsize=18)
plt.savefig("derivatives/extinction_placebo.png")


from matplotlib.transforms import Affine2D

# BF 10 is calculated in JAMOVI
plt.figure(figsize=(5, 5))
y = np.array([1 - 0.235, 0.235])

wedges, labels = plt.pie(y, colors=["w", "darkred"], wedgeprops={"edgecolor": "k"})
plt.title("BF10 for difference\nbetween conditions = 0.235", fontsize=26)


starting_angle = 130
rotation = Affine2D().rotate(np.radians(starting_angle))

for wedge, label in zip(wedges, labels):
    label.set_position(rotation.transform(label.get_position()))
    if label._x > 0:
        label.set_horizontalalignment("left")
    else:
        label.set_horizontalalignment("right")

    wedge._path = wedge._path.transformed(rotation)

plt.tight_layout()
plt.savefig("derivatives/bf10.png", dpi=800, bbox_inches="tight")


# Save all results
wide_dat.to_csv("derivatives/summary_wide.csv")
