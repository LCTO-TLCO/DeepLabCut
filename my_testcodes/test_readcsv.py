# coding:utf-8

##############################
# TODO
##############################


import numpy as np
import pandas as pd
from scipy.spatial import distance
import itertools
import matplotlib.pyplot as plt


def read_csvfile(file_path=r"data/210617-no2DLC_resnet101_guitest6-25shuffle1_50000.csv"):
    # make header
    dfcolumns = pd.read_csv(file_path, skiprows=1, nrows=2, header=None)
    header = dfcolumns.iloc[0] + "[" + dfcolumns.iloc[1] + "]"
    # make df
    df = pd.read_csv(file_path,
                     header=None,
                     skiprows=3,
                     usecols=list(range(len(header))),
                     names=header,
                     index_col=0)
    return df


class Position:
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name

    def distance(self, other):
        return distance.euclidean((self.x, self.y), (other.x, other.y))


class Part:
    def __init__(self, df, name):
        self.x_df = df[f"{name}[x]"]
        self.y_df = df[f"{name}[y]"]
        self.likelihood_df = df[f"{name}[likelihood]"]
        self.positions = []
        for x, y in zip(self.x_df, self.y_df):
            self.positions.append(Position(x, y, name))

    def distance(self, others_part):
        distance_log = []
        for pos, others in zip(self.positions, others_part.positions):
            distance_log.append(pos.distance(others))
        return pd.DataFrame(distance_log, columns=["distance"])


class Scene:
    def __init__(self, df):
        self.lefthand = Part(df[["lefthand[x]", "lefthand[y]", "lefthand[likelihood]"]], "lefthand")
        self.righthand = Part(df[["righthand[x]", "righthand[y]", "righthand[likelihood]"]], "righthand")
        self.leftfoot = Part(df[["leftfoot[x]", "leftfoot[y]", "leftfoot[likelihood]"]], "leftfoot")
        self.rightfoot = Part(df[["rightfoot[x]", "rightfoot[y]", "rightfoot[likelihood]"]], "rightfoot")
        self.mouth = Part(df[["mouth[x]", "mouth[y]", "mouth[likelihood]"]], "mouth")

    def calc_distance(self, parts: list):
        """
        パーツの指定が二つの時のみ動作を補償
        """
        parts_dict = {"lefthand": self.lefthand, "leftfoot": self.leftfoot, "righthand": self.righthand,
                      "rightfoot": self.rightfoot, "mouth": self.mouth}
        return parts_dict[parts[0]].distance(parts_dict[parts[1]])


def _count_continuous(df):
    """
    食事をしているフレーム数の開始・終了を返す
    Parameters
    ----------
    df

    Returns
    -------

    """
    current_tail = 0
    frame_threshold = 10
    true_index = df[df["eating_flag"]].index
    false_index = df[~df["eating_flag"]].index
    count_df = pd.DataFrame(columns=["start", "end"])
    while current_tail < (max(true_index)):
        start_index = true_index[true_index > current_tail][0]
        end_index = false_index[false_index > start_index][0]
        if df["eating_flag"].iloc[start_index:start_index + frame_threshold].sum() < frame_threshold:
            # 三点の距離が短い状況が長く続いていない場合
            # 分析上のノイズとして無視する
            current_tail = end_index
            continue
        while len(false_index[
                      (false_index >= end_index) & (false_index <= end_index + frame_threshold)]) < frame_threshold:
            end_index = false_index[false_index > end_index + 1][0]
        count_df = count_df.append({"start": start_index, "end": end_index}, ignore_index=True)
        current_tail = end_index
    return count_df


def calc_eat_duration(scene, threshold_frames=30, threshold_distance=40):
    target_parts = ["lefthand", "righthand", "mouth"]
    distance_df_dict = {}
    distance_name = []
    # calc distance
    for pair in itertools.combinations(target_parts, 2):
        distance_df_dict[pair] = scene.calc_distance(pair)
        distance_name.append("-".join(pair))
    df_list = list(distance_df_dict.values())
    distance_df = pd.concat(df_list, axis="columns", names=distance_name)
    distance_df.columns = distance_name
    # calc duration
    ## set eating flag
    tmp_df = distance_df.sum(axis=1)
    distance_df["eating_flag"] = distance_df.sum(axis=1) < threshold_distance
    eating_frames = _count_continuous(distance_df)
    eating_frames = eating_frames[(eating_frames.end - eating_frames.start) > threshold_frames]
    eating_frames = eating_frames.assign(duration=(eating_frames.end - eating_frames.start))
    # plot
    eating_frames.duration.hist()
    plt.savefig("fig/eating_duration_hist.png")
    plt.show()
    plt.close()
    return eating_frames


def calc_eat_interval(eat_duration_df):
    if eat_duration_df.size <= 1:
        return False
    eat_interval = pd.DataFrame(columns=["interval"])
    for i in eat_duration_df.index[:-1]:
        eat_interval = eat_interval.append(
            {"interval": (eat_duration_df.iloc[i + 1].start - eat_duration_df.iloc[i].end)}, ignore_index=True)
    eat_interval.interval.hist()
    plt.savefig("fig/eat_interval_hist.png")
    plt.show()
    plt.close()
    return eat_interval


def export_eat_duration_and_interval():
    df = read_csvfile(
        "data/210617-no2DLC_resnet101_guitest6-25shuffle1_50000.csv")
    scene = Scene(df)
    duration_touch = calc_eat_duration(scene, threshold_frames=100)
    duration_eat = calc_eat_duration(scene)
    interval_eat = calc_eat_interval(duration_eat)
    interval_touch = calc_eat_interval(duration_eat)
    return duration_eat, interval_eat, duration_touch, interval_touch


def export_diff_wild_penk():
    pass


if __name__ == "__main__":
    export_eat_duration_and_interval()
