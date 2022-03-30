# coding:utf-8

##############################
# TODO
##############################
import os.path

import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy as sp
import itertools
import matplotlib.pyplot as plt
# import test_datas
import pathlib


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
        if len(false_index[false_index > start_index]) == 0:
            current_tail = max(max(true_index, false_index, key=max))
            continue
        end_index = false_index[false_index > start_index][0]
        if df["eating_flag"].iloc[start_index:start_index + frame_threshold].sum() < frame_threshold:
            # 三点の距離が短い状況が長く続いていない場合
            # 分析上のノイズとして無視する
            current_tail = end_index
            continue
        while len(false_index[
                      (false_index >= end_index) & (false_index <= end_index + frame_threshold)]) < frame_threshold:
            end_index = false_index[false_index > end_index + 1]
            if end_index.empty:
                break
            else:
                end_index = end_index[0]
        if not isinstance(end_index, np.int64):
            break
        count_df = count_df.append({"start": start_index, "end": end_index}, ignore_index=True)
        current_tail = end_index
    return count_df


def calc_eat_duration(scene, mouse_data, threshold_frames=30, threshold_distance=150):
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
    eating_frames = eating_frames.assign(duration=(eating_frames.end - eating_frames.start)).reset_index()
    os.makedirs(f"fig/{mouse_data['type']}", exist_ok=True)
    # plot
    eating_frames.duration.hist()
    plt.title(f"{mouse_data['no']} type:{mouse_data['type']} eat duration")
    plt.xlabel("frames")
    plt.ylabel("frequency")
    plt.savefig(f"fig/{mouse_data['type']}/{mouse_data['no']}_{mouse_data['type']}_eat_duration_hist.png")
    plt.show()
    plt.close()
    return eating_frames


def calc_eat_interval(eat_duration_df, mouse_data):
    if eat_duration_df.size <= 1:
        return pd.DataFrame(columns=["interval"])
    eat_interval = pd.DataFrame(columns=["interval"])
    for i in eat_duration_df.index[:-1]:
        eat_interval = eat_interval.append(
            {"interval": (eat_duration_df.iloc[i + 1].start - eat_duration_df.iloc[i].end)}, ignore_index=True)
    os.makedirs(f"fig/{mouse_data['type']}", exist_ok=True)
    eat_interval.interval.hist()
    plt.title(f"{mouse_data['no']} type:{mouse_data['type']} eat interval")
    plt.xlabel("frames")
    plt.ylabel("frequency")
    plt.savefig(f"fig/{mouse_data['type']}/{mouse_data['no']}_{mouse_data['type']}_eat_interval_hist.png")
    plt.show()
    plt.close()
    return eat_interval


def export_eat_duration_and_interval(files, params={}):
    df_duration_eat = pd.DataFrame()
    df_interval_eat = pd.DataFrame()
    df_duration_touch = pd.DataFrame()
    df_interval_touch = pd.DataFrame()
    eat_frames = params.get("eat_frames", 30)
    touch_frames = params.get("touch_frames", 10)
    eat_distances = params.get("eat_distances", 100)
    touch_distances = params.get("touch_distances", 100)
    paths = read_items(files)
    for no, path in paths.items():
        df = read_csvfile(path)
        scene = Scene(df)
        mouse_data = {"no": no}
        # 30Hz
        duration_touch = calc_eat_duration(scene, {"no": no, "type": "touch"}, threshold_frames=touch_frames,
                                           threshold_distance=touch_distances)
        duration_eat = calc_eat_duration(scene, {"no": no, "type": "eat"}, threshold_frames=eat_frames,
                                         threshold_distance=eat_distances)
        interval_touch = calc_eat_interval(duration_touch, {"no": no, "type": "touch"})
        interval_eat = calc_eat_interval(duration_eat, {"no": no, "type": "eat"})
        # add to df
        if not duration_eat.empty:
            df_duration_eat[no] = duration_eat.duration
        if not interval_eat.empty:
            df_interval_eat[no] = interval_eat.interval
        if not duration_touch.empty:
            df_duration_touch[no] = duration_touch.duration
        if not interval_touch.empty:
            df_interval_touch[no] = interval_touch.interval
    return df_duration_eat, df_interval_eat, df_duration_touch, df_interval_touch


def export_diff_wild_penk(wild_list, penk_list):
    # TODO
    #   ヒストグラムの算出
    #   数値はフレーム数
    #   グループごとの平均を出す
    #   セッション/グループごとの平均
    #
    # 平均
    col = ["duration_eat", "interval_eat", "duration_touch", "interval_touch"]
    wild = dict(zip(col, wild_list))
    penk = dict(zip(col, penk_list))
    # wild_list[0].mean(axis=0)
    # penk_list[0].mean(axis=0)
    for data_type in col:
        wild_current_data = wild[data_type]
        penk_current_data = penk[data_type]
    # TODO 二群を検定する手法を考える
    ## 食べた回数に全体に対して検定
    ##
    os.makedirs(f"fig/summary", exist_ok=True)


def read_items(path):
    ret_val = {}
    if isinstance(path, str):
        path_obj = pathlib.Path(path)
    print(os.getcwd())
    print(path_obj)
    if path_obj.is_dir():
        print("is dir")
        files = list(path_obj.glob('**/*.csv'))
        ret_val = dict(zip([file.stem for file in files], [str(file) for file in files]))
    elif path_obj.is_file():
        print("is a file")
        ret_val = {path_obj.stem: str(path_obj)}
    else:
        print("is dict")

    return ret_val


def read_px(file):
    # TODO
    #   60 or 30
    #   1080p 720p
    #   実時間/距離合計で
    #   eat : 1s/100px in 720p
    #   touch : 0.33s/100px in 720p
    pass


if __name__ == "__main__":
    wild_file = "analyze/data/WT"
    penk_file = "analyze/data/penk"
    wild_dfs = export_eat_duration_and_interval(wild_file)
    penk_dfs = export_eat_duration_and_interval(penk_file, {"eat_distances": 50, "touch_distances": 50})
    export_diff_wild_penk(wild_dfs, penk_dfs)
