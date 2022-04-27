# coding:utf-8

##############################
# TODO
##############################
import os.path
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy as sp
import scipy.stats as st
import itertools
import matplotlib.pyplot as plt
import math
import glob
import os
import pathlib


class FileIo:
    @staticmethod
    def read_csvfile(
            file_path=r"data/210617-no2DLC_resnet101_guitest6-25shuffle1_50000.csv",
    ):
        # make header
        dfcolumns = pd.read_csv(file_path, skiprows=1, nrows=2, header=None)
        header = dfcolumns.iloc[0] + "[" + dfcolumns.iloc[1] + "]"
        # make df
        df = pd.read_csv(
            file_path,
            header=None,
            skiprows=3,
            usecols=list(range(len(header))),
            names=header,
            index_col=0,
        )
        return df

    @staticmethod
    def read_items(path):
        """
        ディレクトリだったらファイル名(拡張子なし）をkey中身をvalueとしたdictで返す
        ファイル単体なら名前を付加してdictで返す
        """
        ret_val = {}
        if isinstance(path, str):
            path_obj = pathlib.Path(path)
        if path_obj.is_dir():
            print("is dir")
            files = list(path_obj.glob("**/*.csv"))
            ret_val = dict(
                zip([file.stem for file in files], [str(file) for file in files])
            )
        elif path_obj.is_file():
            print("is a file")
            ret_val = {path_obj.stem: str(path_obj)}
        else:
            print("is dict")

        return ret_val

    @staticmethod
    def read_dircsv(dir_path):
        filenames = glob.glob(f"{os.path.join(dir_path, '*.csv')}")
        return dict(zip(list(range(len(filenames))), filenames))

    @staticmethod
    def read_fileselect_gui():
        pass


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


class Scene(object):
    def __init__(self, df):
        self.lefthand = Part(
            df[["lefthand[x]", "lefthand[y]", "lefthand[likelihood]"]], "lefthand"
        )
        self.righthand = Part(
            df[["righthand[x]", "righthand[y]", "righthand[likelihood]"]], "righthand"
        )
        self.leftfoot = Part(
            df[["leftfoot[x]", "leftfoot[y]", "leftfoot[likelihood]"]], "leftfoot"
        )
        self.rightfoot = Part(
            df[["rightfoot[x]", "rightfoot[y]", "rightfoot[likelihood]"]], "rightfoot"
        )
        self.mouth = Part(df[["mouth[x]", "mouth[y]", "mouth[likelihood]"]], "mouth")

    def calc_distance(self, parts: list):
        """
        パーツの指定が二つの時のみ動作を補償
        """
        parts_dict = {
            "lefthand": self.lefthand,
            "leftfoot": self.leftfoot,
            "righthand": self.righthand,
            "rightfoot": self.rightfoot,
            "mouth": self.mouth,
        }
        return parts_dict[parts[0]].distance(parts_dict[parts[1]])


# 基底クラス
class Data(object):
    def __init__(self):
        # self.eat_interval = None
        # self.eat_duration = None
        # self.touch_interval = None
        # self.touch_duration = None
        # self.raw_data = None
        # self.mouse_id = None
        # self.num_of_mice = None
        # self.group_name = None
        # self.plot_name = None
        self.parallel_waiting_jobs = []

    def get_dataframe(self):
        return self.raw_data

    def plot_eat_duration(self, mouse_data):
        if mouse_data["type"] == "eat":
            data = self.duration_eat
        elif mouse_data["type"] == "touch":
            data = self.duration_touch
        eating_frames_hist = (
                pd.cut(
                    data.duration,
                    bins=np.arange(0, math.ceil(max(data.duration)), 0.5),
                ).value_counts(sort=False)
                / self.num_of_mice
        )
        eating_frames_hist.index = eating_frames_hist.index.categories.left
        plt.bar(
            eating_frames_hist.index,
            eating_frames_hist,
            width=0.5,
            align="edge",
            linewidth=0.5,
            edgecolor="black",
        )
        plt.axvline(
            data.duration.mean(),
            color="r",
            linestyle="--",
            linewidth=1,
            label="Mean: {:.2f}".format(data.duration.mean()),
        )
        plt.axvline(
            data.duration.median(),
            color="b",
            linestyle="-.",
            linewidth=1,
            label="Median: {:.2f}".format(data.duration.median()),
        )
        plt.title(f"{self.mouse_id} type:{mouse_data['type']} eat duration")
        plt.xlabel("Time[s]")
        plt.ylabel("frequency")
        plt.legend(bbox_to_anchor=(1.05, 0.0), loc="lower left")
        plt.tight_layout()
        os.makedirs(f"fig/{self.group_name}/eat_duration", exist_ok=True)
        plt.savefig(
            # f"fig/{self.group_name}/eat_duration/{mouse_data['group']}-{mouse_data['no']}_{mouse_data['type']}_eat_duration_hist.png"
            f"fig/{self.group_name}/eat_duration/{self.group_name}_{mouse_data['type']}_duration_hist.png"
        )
        # plt.show()
        plt.close()

    def plot_eat_interval(self, mouse_data):
        # mouse_data = settings
        # plot
        if mouse_data["type"] == "eat":
            data = self.interval_eat
        elif mouse_data["type"] == "touch":
            data = self.interval_touch
        eat_interval_hist = (
                pd.cut(
                    data.interval,
                    bins=np.arange(0, math.ceil(max(data.interval)), 0.5),
                ).value_counts(sort=False)
                / self.num_of_mice
        )
        eat_interval_hist.index = eat_interval_hist.index.categories.left
        plt.bar(
            eat_interval_hist.index,
            eat_interval_hist,
            width=0.5,
            align="edge",
            linewidth=0.5,
            edgecolor="black",
        )
        plt.axvline(
            data.interval.mean(),
            color="r",
            linestyle="--",
            linewidth=1,
            label="Mean: {:.2f}".format(data.interval.mean()),
        )
        plt.axvline(
            data.interval.median(),
            color="b",
            linestyle="-.",
            linewidth=1,
            label="Median: {:.2f}".format(data.interval.median()),
        )
        plt.title(
            f"{self.group_name}:type:{mouse_data['type']} eat interval"
        )
        plt.xlabel("Time[s]")
        plt.ylabel("frequency")
        plt.legend(bbox_to_anchor=(1.05, 0.0), loc="lower left")
        plt.tight_layout()
        os.makedirs(f"fig/{self.group_name}/eat_interval", exist_ok=True)
        plt.savefig(
            # f"fig/{self.group_name}/eat_interval/{self.group_name}-{mouse_data['no']}_{mouse_data['type']}_eat_interval_hist.png"
            f"fig/{self.group_name}/eat_interval/{self.plot_name}_{mouse_data['type']}_interval_hist.png"
        )
        # plt.show()
        plt.close()

    def plot_all(self):
        self.plot_eat_duration({"type": "eat"})
        self.plot_eat_duration({"type": "touch"})
        self.plot_eat_interval({"type": "eat"})
        self.plot_eat_interval({"type": "touch"})


# 単体
class MouseData(Data):
    # TODO
    #   delayedとか書いてある部分のジェネレータを返す？
    #   外側でParallel実行する？
    #   マウス単体のデータを管理する
    #       素のデータ
    #       各duration, interval
    #
    def __init__(self, **args):
        # params no path
        super().__init__()
        self._preprocess(**args)
        self.plot_all()
        # return (joblib.delayed(self._preprocess)(*args))

    def _preprocess(self, **args):
        params = args["params"]
        no = args["no"]
        path = args["files"]
        self.group_name = args["group_name"]
        self.mouse_id = no
        self.num_of_mice = 1
        self.plot_name = f"{self.group_name}-{self.mouse_id}"
        df = FileIo.read_csvfile(path)
        scene = Scene(df)
        self.raw_data = scene
        eat_frames = params.get("eat_frames", 30)
        touch_frames = params.get("touch_frames", 10)
        eat_distances = params.get("eat_distances", 100)
        touch_distances = params.get("touch_distances", 100)
        self.duration_touch = self.calc_eat_duration(
            scene,
            {"no": no, "type": "touch"},
            threshold_frames=touch_frames,
            threshold_distance=touch_distances,
        )
        self.duration_eat = self.calc_eat_duration(
            scene,
            {"no": no, "type": "eat"},
            threshold_frames=eat_frames,
            threshold_distance=eat_distances,
        )
        self.interval_touch = self.calc_eat_interval(
            self.duration_touch, {"no": no, "type": "touch"}
        )
        self.interval_eat = self.calc_eat_interval(
            self.duration_eat, {"no": no, "type": "eat"}
        )

    @staticmethod
    def calc_eat_duration(
            scene, mouse_data, threshold_frames=30, threshold_distance=150, num_of_mice=1
    ):
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
        distance_df["eating_flag"] = distance_df.sum(axis=1) < threshold_distance
        eating_frames = calc._count_continuous(distance_df)
        eating_frames = eating_frames[
            (eating_frames.end - eating_frames.start) > threshold_frames
            ]
        eating_frames = eating_frames.assign(
            duration=(eating_frames.end - eating_frames.start) / 30
        ).reset_index()
        eating_frames = eating_frames.assign(mouse_id=mouse_data["no"])
        if eating_frames.empty:
            return eating_frames
        return eating_frames

    @staticmethod
    def calc_eat_interval(eat_duration_df, mouse_data, num_of_mice=1):
        if eat_duration_df.size <= 1:
            return pd.DataFrame(columns=["interval"])
        eat_interval = pd.DataFrame(columns=["interval"])
        for i in eat_duration_df.index[:-1]:
            eat_interval = eat_interval.append(
                {
                    "interval": (eat_duration_df.iloc[i + 1].start - eat_duration_df.iloc[i].end) / 30},
                ignore_index=True,
            )
        eat_interval = eat_interval.assign(mouse_id=mouse_data["no"])
        if eat_interval.empty:
            return eat_interval
        return eat_interval

    @staticmethod
    def export_analysed_data(data_list, group_name):
        (
            df_duration_eat,
            df_interval_eat,
            df_duration_touch,
            df_interval_touch,
            # all_stats,
        ) = data_list
        os.makedirs(f"data/export/{group_name}", exist_ok=True)
        df_duration_eat.to_csv(f"data/export/{group_name}/individual_data_duration_eat.csv")
        df_interval_eat.to_csv(f"data/export/{group_name}/individual_data_interval_eat.csv")
        df_duration_touch.to_csv(
            f"data/export/{group_name}/individual_data_duration_touch.csv"
        )
        df_interval_touch.to_csv(
            f"data/export/{group_name}/individual_data_interval_touch.csv"
        )
        # all_stats.to_csv(f"data/export/{group_name}/all_stats.csv")

    @staticmethod
    def plot_cumlative_durations_and_intervals(data_list, params=None):
        if params is None:
            params = {}
        if params.get("group", False):
            group_name = params["group"]
        else:
            group_name = "nogroup"
        for data_type, data in zip(
                ["eat duration", "eat interval", "touch duration", "touch interval"], data_list
        ):
            raw_df = data.filter(["duration", "interval"])
            if not os.path.exists(
                    f"fig/{group_name}/duration_or_interval_cumlative/{params.get('mouse_id', 'allmice')}"
            ):
                os.makedirs(
                    f"fig/{group_name}/duration_or_interval_cumlative/{params.get('mouse_id', 'allmice')}"
                )
            if params.get("mouse_id", False):
                raw_df = raw_df[data.mouse_id.isin([params["mouse_id"]])]
            if raw_df.empty or all(raw_df.count() == 0):
                continue
            # duration or interval
            cum_df = st.cumfreq(
                raw_df, numbins=np.arange(0, math.ceil(raw_df.max()), 0.5).size
            )
            x = cum_df.lowerlimit + np.linspace(
                0, cum_df.binsize * cum_df.cumcount.size, cum_df.cumcount.size
            )
            plt.bar(x, cum_df.cumcount)
            plt.title(
                f"{group_name} {params.get('mouse_id', 'allmice')} cumlative {data_type}"
            )
            plt.xlabel("Time[s]")
            plt.ylabel("frequency")
            plt.tight_layout()
            plt.savefig(
                f"fig/{group_name}/duration_or_interval_cumlative/{params.get('mouse_id', 'allmice')}/{group_name}_{params.get('mouse_id', 'allmice')}_{data_type}_cumlative_histogram.png"
            )
            # plt.show()
            plt.close()


# 複数
class MouseGroup(Data):
    # TODO
    #   マウスグループのデータを管理する
    #   グループのMouseData全て
    #   内部でデータフレームを結合するための仕組みが必要？
    #   グループごと分析部分を実装
    #   ファイルリストからMouseDataを回す

    def __init__(self, files, params):
        super().__init__()
        self.group_name = params.get("group", "nogroup")
        self.mice_data = {}
        self.mouse_id = []
        self._preprocess(files, params)
        self.export_analysed_data()
        self.plot_all()

    def _preprocess(self, files, params):
        if params is None:
            params = {}
        df_duration_eat = pd.DataFrame(columns=["mouse_id", "duration"])
        df_interval_eat = pd.DataFrame(columns=["mouse_id", "interval"])
        df_duration_touch = pd.DataFrame(columns=["mouse_id", "duration"])
        df_interval_touch = pd.DataFrame(columns=["mouse_id", "interval"])
        paths = FileIo.read_items(files)
        self.num_of_mice = len(paths)
        self.plot_name = f"{self.group_name}-all"

        # joblib?
        # self.parallel_waiting_jobs = [joblib.delayed(MouseData)(params, path, no) for path, no in paths.items()]
        # self.run_jobs_parallel(self.parallel_waiting_jobs)

        # or normal
        for no, path in paths.items():
            mice_data = MouseData(files=path, params=params, group_name=self.group_name, no=no)
            self.mice_data = {**self.mice_data, no: mice_data}
            self.mouse_id.append(no)
            # add to df
            if not mice_data.duration_eat.empty:
                df_duration_eat = df_duration_eat.append(mice_data.duration_eat[["duration"]].assign(mouse_id=no))
            if not mice_data.interval_eat.empty:
                df_interval_eat = df_interval_eat.append(mice_data.interval_eat[["interval"]].assign(mouse_id=no))
            if not mice_data.duration_touch.empty:
                df_duration_touch = df_duration_touch.append(mice_data.duration_touch[["duration"]].assign(mouse_id=no))
            if not mice_data.interval_touch.empty:
                df_interval_touch = df_interval_touch.append(mice_data.interval_touch[["interval"]].assign(mouse_id=no))

        self.duration_eat = df_duration_eat.copy()
        self.interval_eat = df_interval_eat.copy()
        self.duration_touch = df_duration_touch.copy()
        self.interval_touch = df_interval_touch.copy()

    def plot_all(self):
        self.plot_eat_duration({"type": "eat"})
        self.plot_eat_duration({"type": "touch"})
        self.plot_eat_interval({"type": "eat"})
        self.plot_eat_interval({"type": "touch"})

    def export_analysed_data(self):
        # TODO
        #   export csv
        #   mean
        #   median
        #   mode
        os.makedirs(f"data/export/{self.group_name}", exist_ok=True)

        self.duration_eat.to_csv(f"data/export/{self.group_name}/individual_data_duration_eat.csv")
        self.interval_eat.to_csv(f"data/export/{self.group_name}/individual_data_interval_eat.csv")
        self.duration_touch.to_csv(
            f"data/export/{self.group_name}/individual_data_duration_touch.csv"
        )
        self.interval_touch.to_csv(
            f"data/export/{self.group_name}/individual_data_interval_touch.csv"
        )
        # all_stats.to_csv(f"data/export/{self.group_name}/all_stats.csv")

    # def plot_eat_duration(self, mouse_data):
    #     data = mouse_data["type"]
    #     eating_frames_hist = (
    #             pd.cut(
    #                 data.duration,
    #                 bins=np.arange(0, math.ceil(max(data.duration)), 0.5),
    #             ).value_counts(sort=False)
    #             / len(self.mice_data)
    #     )
    #     eating_frames_hist.index = eating_frames_hist.index.categories.left
    #     plt.bar(
    #         eating_frames_hist.index,
    #         eating_frames_hist,
    #         width=0.5,
    #         align="edge",
    #         linewidth=0.5,
    #         edgecolor="black",
    #     )
    #     plt.axvline(
    #         data.duration.mean(),
    #         color="r",
    #         linestyle="--",
    #         linewidth=1,
    #         label="Mean: {:.2f}".format(data.duration.mean()),
    #     )
    #     plt.axvline(
    #         data.duration.median(),
    #         color="b",
    #         linestyle="-.",
    #         linewidth=1,
    #         label="Median: {:.2f}".format(data.duration.median()),
    #     )
    #     plt.title(f"{mouse_data['no']} type:{mouse_data['type']} eat duration")
    #     plt.xlabel("Time[s]")
    #     plt.ylabel("frequency")
    #     plt.legend(bbox_to_anchor=(1.05, 0.0), loc="lower left")
    #     plt.tight_layout()
    #     os.makedirs(f"fig/{mouse_data['group']}/eat_duration", exist_ok=True)
    #     plt.savefig(
    #         f"fig/{mouse_data['group']}/eat_duration/{mouse_data['group']}-{mouse_data['no']}_{mouse_data['type']}_eat_duration_hist.png"
    #     )
    #     # plt.show()
    #     plt.close()
    #
    # def plot_eat_interval(self, mouse_data):
    #     # mouse_data = settings
    #     # plot
    #     eat_interval_hist = (
    #             pd.cut(
    #                 data.interval,
    #                 bins=np.arange(0, math.ceil(max(data.interval)), 0.5),
    #             ).value_counts(sort=False)
    #             / len(self.mice_data)
    #     )
    #     eat_interval_hist.index = eat_interval_hist.index.categories.left
    #     plt.bar(
    #         eat_interval_hist.index,
    #         eat_interval_hist,
    #         width=0.5,
    #         align="edge",
    #         linewidth=0.5,
    #         edgecolor="black",
    #     )
    #     plt.axvline(
    #         data.interval.mean(),
    #         color="r",
    #         linestyle="--",
    #         linewidth=1,
    #         label="Mean: {:.2f}".format(data.interval.mean()),
    #     )
    #     plt.axvline(
    #         data.interval.median(),
    #         color="b",
    #         linestyle="-.",
    #         linewidth=1,
    #         label="Median: {:.2f}".format(data.interval.median()),
    #     )
    #     plt.title(
    #         f"{self.group_name}:type:{mouse_data['type']} eat interval"
    #     )
    #     plt.xlabel("Time[s]")
    #     plt.ylabel("frequency")
    #     plt.legend(bbox_to_anchor=(1.05, 0.0), loc="lower left")
    #     plt.tight_layout()
    #     os.makedirs(f"fig/{self.group_name}/eat_interval", exist_ok=True)
    #     plt.savefig(
    #         f"fig/{self.group_name}/eat_interval/{self.group_name}-{mouse_data['no']}_{mouse_data['type']}_eat_interval_hist.png"
    #     )
    #     # plt.show()
    #     plt.close()


# 分析
class GroupCompare:
    def __init__(self, groups):
        self.groups = groups
        self.settings = None
        self.plot_all()

    def export_diff_wild_penk(self):
        # TODO
        #   ヒストグラムの算出
        #   数値はフレーム数
        #   グループごとの平均を出す
        #   セッション/グループごとの平均

        # 平均
        col = ["duration eat", "interval eat", "duration touch", "interval touch"]
        groupA_dict = dict(zip(col, [getattr(self.groups[0], name.replace(' ', '_')) for name in col]))
        groupB_dict = dict(zip(col, [getattr(self.groups[1], name.replace(' ', '_')) for name in col]))
        groupA_name = self.groups[0].group_name
        groupB_name = self.groups[1].group_name
        groups = "-".join([groupA_name, groupB_name])
        os.makedirs(f"fig/{groups}/summary", exist_ok=True)
        for data_type in col:
            # groupA_dict.duration_eat
            groupA_current_data = groupA_dict[data_type]
            groupB_current_data = groupB_dict[data_type]
            ## in wild group
            groupA_st_result = st.kruskal(
                *[groupA_current_data.query('mouse_id == @data').filter(["duration", "interval"]).values.flatten() for
                  data in
                  groupA_current_data.mouse_id.unique()], nan_policy="omit")
            ## in penk group
            groupB_st_result = st.kruskal(
                *[groupB_current_data.query('mouse_id == @data').filter(["duration", "interval"]).values.flatten() for
                  data in
                  groupB_current_data.mouse_id.unique()], nan_policy="omit")
            ## both groups
            both_st_result = st.mannwhitneyu(groupA_current_data.filter(["duration", "interval"]).values.flatten(),
                                             groupB_current_data.filter(["duration", "interval"]).values.flatten())

            # TODO
            #  export stats
            os.makedirs(f"analyze/stats/{groups}", exist_ok=True)

            # plot
            fig, ax = plt.subplots(1, 2)
            ax[0].boxplot([groupA_current_data.filter(["duration", "interval"]).values.flatten(),
                           groupB_current_data.filter(["duration", "interval"]).values.flatten()],
                          labels=["wild", "penk"],
                          sym="+")
            ax[0].set_xlabel("type")
            ax[0].set_ylabel("sec")
            ax[0].set_title(f"{data_type}")
            # 外れ値描画なし
            ax[1].boxplot([groupA_current_data.filter(["duration", "interval"]).values.flatten(),
                           groupB_current_data.filter(["duration", "interval"]).values.flatten()],
                          # whis=[5, 95],
                          labels=["wild", "penk"],
                          sym="")
            ax[1].set_xlabel("type")
            ax[1].set_ylabel("sec")
            ax[1].set_title(f"{data_type}")
            plt.savefig(f"fig/{groups}/summary/{data_type.replace(' ', '_')}_vsboxplot.png")
            plt.close(fig)

    def plot_compare_graph_durations_and_intervals(self, density=False):
        """
        @params density: default False
                                プロット対象データを個体の確率の平均にする
                                Falseの場合すべてのマウスのデータを一緒くたにし純粋な個数を個体数で割って平均を算出
        """
        data_types = ["duration eat", "interval eat", "duration touch", "interval touch"]

        groups = "-".join([self.groups[i].group_name for i in range(len(self.groups))])

        for data_type in data_types:
            if not os.path.exists(f"fig/{groups}/compare_histogram"):
                os.makedirs(f"fig/{groups}/compare_histogram")
            fig, ax = plt.subplots()
            range_max = max(
                [
                    getattr(data, data_type.replace(" ", "_"))
                        .filter(["duration", "interval"])
                        .values.max()
                    for data in self.groups
                ]
            )
            ranges = (0, math.ceil(range_max))
            bins = int(math.ceil(range_max) / 0.5)
            multiplier = 0
            while bins > 40:
                bins = int(bins / 2)
                multiplier += 1
            idx = np.arange(*ranges, 0.5 * pow(2, multiplier))
            for data_list in self.groups:
                raw_df = (
                    getattr(data_list, data_type.replace(" ", "_"))
                        .filter(["duration", "interval"])
                        .values
                )
                if not raw_df.size:
                    continue
                # duration or interval
                if density:
                    hist_dfs = []
                    for mouse in getattr(data_list, data_type.replace(" ", "_")).mouse_id.unique():
                        data = getattr(data_list, data_type.replace(" ", "_"))
                        data = data[data.mouse_id.isin([mouse])].filter(
                            ["duration", "interval"]
                        )
                        hist_dfs.append(
                            np.histogram(data, range=ranges, bins=bins, density=True)[0]
                        )
                    hist_df = np.mean(hist_dfs, axis=0)
                else:
                    n = getattr(data_list, data_type.replace(" ", "_")).mouse_id.unique().size
                    hist_df = np.histogram(raw_df, range=ranges, bins=bins)
                    hist_df = hist_df[0] / n
                if idx.size != hist_df.size:
                    # odd
                    idx = idx[:-1]
                ax.bar(
                    idx,
                    hist_df,
                    label=data_list.group_name,
                    align="edge",
                    alpha=0.3,
                    ec="black",
                    width=0.5 * pow(2, multiplier),
                )
                # ax.hist(raw_df, label=param["group"], alpha=0.3, range=ranges, bins=bins, ec='black')

            ax.set_title(
                f"{groups} {data_type} {'normed graph' if density else 'histogram'}"
            )
            ax.set_xlabel("Time[s]")
            ax.set_ylabel("frequency")
            ax.legend(loc="lower right")
            # ax.legend(bbox_to_anchor=(1.05, 0.0), loc='lower left')
            fig.tight_layout()
            plt.savefig(
                f"fig/{groups}/compare_histogram/{groups}_{data_type.replace(' ', '_')}_compare_{'normed graph' if density else 'histogram'}.png"
            )
            plt.show()
            plt.close()

    def plot_compare_graph_cumlative_durations_and_intervals(self):
        data_types = ["duration eat", "interval eat", "duration touch", "interval touch"]
        groups = "-".join([self.groups[i].group_name for i in range(len(self.groups))])

        if not os.path.exists(f"fig/{groups}/compare_cumlative"):
            os.makedirs(f"fig/{groups}/compare_cumlative")
        for data_type in data_types:
            fig, ax = plt.subplots()
            range_max = max(
                [
                    getattr(data, data_type.replace(" ", "_"))
                        .filter(["duration", "interval"])
                        .values.max()
                    for data in self.groups
                ]
            )
            # TODO
            ## durationに何が入っている前提だったか
            ## 基本的に
            ranges = (0, math.ceil(range_max))
            bins = int(math.ceil(range_max) / 0.5)
            while bins > 40:
                bins = int(bins / 2)
            for data_list in self.groups:
                raw_df = (
                    getattr(data_list, data_type.replace(" ", "_"))
                        .filter(["duration", "interval"])
                        .values
                )
                if not raw_df.size:
                    continue
                # duration or interval
                ax.hist(
                    raw_df,
                    density=True,
                    cumulative=True,
                    label=data_list.group_name,
                    alpha=0.3,
                    range=ranges,
                    bins=bins,
                    ec="black",
                )

            ax.set_title(f"{groups} cumlative {data_type}")
            ax.set_xlabel("Time[s]")
            ax.set_ylabel("frequency")
            ax.legend(loc="lower right")
            # ax.legend(bbox_to_anchor=(1.05, 0.0), loc='lower left')
            fig.tight_layout()
            plt.savefig(
                f"fig/{groups}/compare_cumlative/{groups}_{data_type.replace(' ', '_')}_cumlative_histogram.png"
            )
            # plt.show()
            plt.close()

    def plot_all(self):
        self.plot_compare_graph_cumlative_durations_and_intervals()
        self.plot_compare_graph_durations_and_intervals()
        self.plot_compare_graph_durations_and_intervals(density=True)
        self.export_diff_wild_penk()


class calc:

    @staticmethod
    def calc_eat_duration(
            scene, mouse_data, threshold_frames=30, threshold_distance=150, num_of_mice=1
    ):
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
        distance_df["eating_flag"] = distance_df.sum(axis=1) < threshold_distance
        eating_frames = calc._count_continuous(distance_df)
        eating_frames = eating_frames[
            (eating_frames.end - eating_frames.start) > threshold_frames
            ]
        eating_frames = eating_frames.assign(
            duration=(eating_frames.end - eating_frames.start) / 30
        ).reset_index()
        eating_frames = eating_frames.assign(mouse_id=mouse_data["no"])
        if eating_frames.empty:
            return eating_frames
        # plot
        eating_frames_hist = (
                pd.cut(
                    eating_frames.duration,
                    bins=np.arange(0, math.ceil(max(eating_frames.duration)), 0.5),
                ).value_counts(sort=False)
                / num_of_mice
        )
        eating_frames_hist.index = eating_frames_hist.index.categories.left
        plt.bar(
            eating_frames_hist.index,
            eating_frames_hist,
            width=0.5,
            align="edge",
            linewidth=0.5,
            edgecolor="black",
        )
        plt.axvline(
            eating_frames.duration.mean(),
            color="r",
            linestyle="--",
            linewidth=1,
            label="Mean: {:.2f}".format(eating_frames.duration.mean()),
        )
        plt.axvline(
            eating_frames.duration.median(),
            color="b",
            linestyle="-.",
            linewidth=1,
            label="Median: {:.2f}".format(eating_frames.duration.median()),
        )
        plt.title(f"{mouse_data['no']} type:{mouse_data['type']} eat duration")
        plt.xlabel("Time[s]")
        plt.ylabel("frequency")
        plt.legend(bbox_to_anchor=(1.05, 0.0), loc="lower left")
        plt.tight_layout()
        os.makedirs(f"fig/{mouse_data['group']}/eat_duration", exist_ok=True)
        plt.savefig(
            f"fig/{mouse_data['group']}/eat_duration/{mouse_data['group']}-{mouse_data['no']}_{mouse_data['type']}_eat_duration_hist.png"
        )
        # plt.show()
        plt.close()
        return eating_frames

    @staticmethod
    def calc_eat_interval(eat_duration_df, mouse_data, num_of_mice=1):
        if eat_duration_df.size <= 1:
            return pd.DataFrame(columns=["interval"])
        eat_interval = pd.DataFrame(columns=["interval"])
        for i in eat_duration_df.index[:-1]:
            eat_interval = eat_interval.append(
                {
                    "interval": (
                                        eat_duration_df.iloc[i + 1].start - eat_duration_df.iloc[i].end
                                )
                                / 30
                },
                ignore_index=True,
            )
        eat_interval = eat_interval.assign(mouse_id=mouse_data["no"])
        if eat_interval.empty:
            return eat_interval
        # plot
        eat_interval_hist = (
                pd.cut(
                    eat_interval.interval,
                    bins=np.arange(0, math.ceil(max(eat_interval.interval)), 0.5),
                ).value_counts(sort=False)
                / num_of_mice
        )
        eat_interval_hist.index = eat_interval_hist.index.categories.left
        plt.bar(
            eat_interval_hist.index,
            eat_interval_hist,
            width=0.5,
            align="edge",
            linewidth=0.5,
            edgecolor="black",
        )
        plt.axvline(
            eat_interval.interval.mean(),
            color="r",
            linestyle="--",
            linewidth=1,
            label="Mean: {:.2f}".format(eat_interval.interval.mean()),
        )
        plt.axvline(
            eat_interval.interval.median(),
            color="b",
            linestyle="-.",
            linewidth=1,
            label="Median: {:.2f}".format(eat_interval.interval.median()),
        )
        plt.title(
            f"{mouse_data['group']}:{mouse_data['no']} type:{mouse_data['type']} eat interval"
        )
        plt.xlabel("Time[s]")
        plt.ylabel("frequency")
        plt.legend(bbox_to_anchor=(1.05, 0.0), loc="lower left")
        plt.tight_layout()
        os.makedirs(f"fig/{mouse_data['group']}/eat_interval", exist_ok=True)
        plt.savefig(
            f"fig/{mouse_data['group']}/eat_interval/{mouse_data['group']}-{mouse_data['no']}_{mouse_data['type']}_eat_interval_hist.png"
        )
        # plt.show()
        plt.close()
        return eat_interval

    @staticmethod
    def export_eat_duration_and_interval(files, params=None):
        if params is None:
            params = {}
        df_duration_eat = pd.DataFrame(columns=["mouse_id", "duration"])
        df_interval_eat = pd.DataFrame(columns=["mouse_id", "interval"])
        df_duration_touch = pd.DataFrame(columns=["mouse_id", "duration"])
        df_interval_touch = pd.DataFrame(columns=["mouse_id", "interval"])
        eat_frames = params.get("eat_frames", 30)
        touch_frames = params.get("touch_frames", 10)
        eat_distances = params.get("eat_distances", 100)
        touch_distances = params.get("touch_distances", 100)
        group = params.get("group", "nogroup")
        paths = FileIo.read_items(files)
        for no, path in paths.items():
            df = FileIo.read_csvfile(path)
            scene = Scene(df)
            mouse_data = {"no": no}
            # 30Hz
            duration_touch = calc_eat_duration(
                scene,
                {"no": no, "type": "touch", "group": group},
                threshold_frames=touch_frames,
                threshold_distance=touch_distances,
            )
            duration_eat = calc_eat_duration(
                scene,
                {"no": no, "type": "eat", "group": group},
                threshold_frames=eat_frames,
                threshold_distance=eat_distances,
            )
            interval_touch = calc_eat_interval(
                duration_touch, {"no": no, "type": "touch", "group": group}
            )
            interval_eat = calc_eat_interval(
                duration_eat, {"no": no, "type": "eat", "group": group}
            )
            # add to df
            if not duration_eat.empty:
                df_duration_eat = df_duration_eat.append(duration_eat[["duration"]].assign(mouse_id=no))
            if not interval_eat.empty:
                df_interval_eat = df_interval_eat.append(interval_eat[["interval"]].assign(mouse_id=no))
            if not duration_touch.empty:
                df_duration_touch = df_duration_touch.append(duration_touch[["duration"]].assign(mouse_id=no))
            if not interval_touch.empty:
                df_interval_touch = df_interval_touch.append(interval_touch[["interval"]].assign(mouse_id=no))
        return df_duration_eat, df_interval_eat, df_duration_touch, df_interval_touch

    @staticmethod
    def export_diff_wild_penk(dfs_list, params_list):
        # TODO
        #   ヒストグラムの算出
        #   数値はフレーム数
        #   グループごとの平均を出す
        #   セッション/グループごとの平均

        # 平均
        col = ["duration_eat", "interval_eat", "duration_touch", "interval_touch"]
        wild = dict(zip(col, dfs_list[0]))
        penk = dict(zip(col, dfs_list[1]))
        groupA_name = params_list[0].get("group", "")
        groupB_name = params_list[1].get("group", "")
        groups = "-".join([groupA_name, groupB_name])
        os.makedirs(f"fig/{groups}/summary", exist_ok=True)
        for data_type in col:
            wild_current_data = wild[data_type]
            penk_current_data = penk[data_type]
            ## in wild group
            wild_st_result = st.kruskal(
                *[wild_current_data.query('mouse_id == @data').filter(["duration", "interval"]).values.flatten() for
                  data in
                  wild_current_data.mouse_id.unique()], nan_policy="omit")
            ## in penk group
            penk_st_result = st.kruskal(
                *[penk_current_data.query('mouse_id == @data').filter(["duration", "interval"]).values.flatten() for
                  data in
                  penk_current_data.mouse_id.unique()], nan_policy="omit")
            ## both groups
            both_st_result = st.mannwhitneyu(wild_current_data.filter(["duration", "interval"]).values.flatten(),
                                             penk_current_data.filter(["duration", "interval"]).values.flatten())

            # plot
            fig, ax = plt.subplots(1, 2)
            ax[0].boxplot([wild_current_data.filter(["duration", "interval"]).values.flatten(),
                           penk_current_data.filter(["duration", "interval"]).values.flatten()],
                          labels=["wild", "penk"],
                          sym="+")
            ax[0].set_xlabel("type")
            ax[0].set_ylabel("sec")
            ax[0].set_title(f"{data_type}")
            # 外れ値描画なし
            ax[1].boxplot([wild_current_data.filter(["duration", "interval"]).values.flatten(),
                           penk_current_data.filter(["duration", "interval"]).values.flatten()],
                          # whis=[5, 95],
                          labels=["wild", "penk"],
                          sym="")
            ax[1].set_xlabel("type")
            ax[1].set_ylabel("sec")
            ax[1].set_title(f"{data_type}")
            plt.savefig(f"fig/{groups}/summary/{data_type}_vsboxplot.png")
            plt.close(fig)

    @staticmethod
    def export_analysed_data(data_list, group_name):
        (
            df_duration_eat,
            df_interval_eat,
            df_duration_touch,
            df_interval_touch,
            # all_stats,
        ) = data_list
        os.makedirs(f"data/export/{group_name}", exist_ok=True)
        df_duration_eat.to_csv(f"data/export/{group_name}/individual_data_duration_eat.csv")
        df_interval_eat.to_csv(f"data/export/{group_name}/individual_data_interval_eat.csv")
        df_duration_touch.to_csv(
            f"data/export/{group_name}/individual_data_duration_touch.csv"
        )
        df_interval_touch.to_csv(
            f"data/export/{group_name}/individual_data_interval_touch.csv"
        )
        # all_stats.to_csv(f"data/export/{group_name}/all_stats.csv")

    @staticmethod
    def read_px(file):
        # TODO
        #   設定について
        ## 1080p の場合
        #   "eat_frames", 30
        #   "eat_distances", 100
        #   "touch_frames", 10
        #   "touch_distances", 100
        ## 720p の場合
        #   "eat_frames", 30 / 1.5 = 20
        #   "eat_distances", 100 / 1.5 = 66.6
        #   "touch_frames", 10 / 1.5 = 6.6
        #   "touch_distances", 100 / 1.5 = 66.6
        #   実時間/距離合計で
        #   eat : 1s/100px in 720p
        #   touch : 0.33s/100px in 720p
        pass

    @staticmethod
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
        count_df = pd.DataFrame(columns=["start", "end"], dtype=float)
        while current_tail < (max(true_index)):
            start_index = true_index[true_index > current_tail][0]
            end_index = false_index[false_index > start_index]
            if end_index.empty:
                end_index = true_index[-1]
            else:
                end_index = end_index[0]
            if (
                    df["eating_flag"].iloc[start_index: start_index + frame_threshold].sum()
                    < frame_threshold
            ):
                # 三点の距離が短い状況が長く続いていない場合
                # 分析上のノイズとして無視する
                current_tail = end_index
                continue
            while (
                    len(
                        false_index[
                            (false_index >= end_index)
                            & (false_index <= end_index + frame_threshold)
                        ]
                    )
                    < frame_threshold
            ):
                end_index = false_index[false_index > end_index + 1]
                if end_index.empty:
                    break
                else:
                    end_index = end_index[0]
            if not isinstance(end_index, np.int64):
                break
            count_df = count_df.append(
                {"start": start_index, "end": end_index}, ignore_index=True
            )
            current_tail = end_index
        return count_df


class plot:

    @staticmethod
    def plot_cumlative_durations_and_intervals(data_list, params=None):
        if params is None:
            params = {}
        if params.get("group", False):
            group_name = params["group"]
        else:
            group_name = "nogroup"
        for data_type, data in zip(
                ["eat duration", "eat interval", "touch duration", "touch interval"], data_list
        ):
            raw_df = data.filter(["duration", "interval"])
            if not os.path.exists(
                    f"fig/{group_name}/duration_or_interval_cumlative/{params.get('mouse_id', 'allmice')}"
            ):
                os.makedirs(
                    f"fig/{group_name}/duration_or_interval_cumlative/{params.get('mouse_id', 'allmice')}"
                )
            if params.get("mouse_id", False):
                raw_df = raw_df[data.mouse_id.isin([params["mouse_id"]])]
            if raw_df.empty or all(raw_df.count() == 0):
                continue
            # duration or interval
            cum_df = st.cumfreq(
                raw_df, numbins=np.arange(0, math.ceil(raw_df.max()), 0.5).size
            )
            x = cum_df.lowerlimit + np.linspace(
                0, cum_df.binsize * cum_df.cumcount.size, cum_df.cumcount.size
            )
            plt.bar(x, cum_df.cumcount)
            plt.title(
                f"{group_name} {params.get('mouse_id', 'allmice')} cumlative {data_type}"
            )
            plt.xlabel("Time[s]")
            plt.ylabel("frequency")
            plt.tight_layout()
            plt.savefig(
                f"fig/{group_name}/duration_or_interval_cumlative/{params.get('mouse_id', 'allmice')}/{group_name}_{params.get('mouse_id', 'allmice')}_{data_type}_cumlative_histogram.png"
            )
            # plt.show()
            plt.close()

    @staticmethod
    def plot_compare_graph_durations_and_intervals(data_lists, params, density=False):
        """
        @params density: default False
                                プロット対象データを個体の確率の平均にする
                                Falseの場合すべてのマウスのデータを一緒くたにし純粋な個数を個体数で割って平均を算出
        """
        data_types = ["eat duration", "eat interval", "touch duration", "touch interval"]
        groups = "-".join([param.get("group", "noname_group") for param in params])

        for data_type in data_types:
            if not os.path.exists(f"fig/{groups}/compare_histogram"):
                os.makedirs(f"fig/{groups}/compare_histogram")
            fig, ax = plt.subplots()
            range_max = max(
                [
                    data[data_types.index(data_type)]
                        .filter(["duration", "interval"])
                        .values.max()
                    for data in data_lists
                ]
            )
            ranges = (0, math.ceil(range_max))
            bins = int(math.ceil(range_max) / 0.5)
            multiplier = 0
            while bins > 40:
                bins = int(bins / 2)
                multiplier += 1
            idx = np.arange(*ranges, 0.5 * pow(2, multiplier))
            for data_list, param in zip(data_lists, params):
                raw_df = (
                    data_list[data_types.index(data_type)]
                        .filter(["duration", "interval"])
                        .values
                )
                if not raw_df.size:
                    continue
                # duration or interval
                if density:
                    hist_dfs = []
                    for mouse in data_list[data_types.index(data_type)].mouse_id.unique():
                        data = data_list[data_types.index(data_type)]
                        data = data[data.mouse_id.isin([mouse])].filter(
                            ["duration", "interval"]
                        )
                        hist_dfs.append(
                            np.histogram(data, range=ranges, bins=bins, density=True)[0]
                        )
                    hist_df = np.mean(hist_dfs, axis=0)
                else:
                    n = data_list[data_types.index(data_type)].mouse_id.unique().size
                    hist_df = np.histogram(raw_df, range=ranges, bins=bins)
                    hist_df = hist_df[0] / n
                if idx.size != hist_df.size:
                    # odd
                    idx = idx[:-1]
                ax.bar(
                    idx,
                    hist_df,
                    label=param["group"],
                    align="edge",
                    alpha=0.3,
                    ec="black",
                    width=0.5 * pow(2, multiplier),
                )
                # ax.hist(raw_df, label=param["group"], alpha=0.3, range=ranges, bins=bins, ec='black')

            ax.set_title(
                f"{groups} {data_type} {'normed graph' if density else 'histogram'}"
            )
            ax.set_xlabel("Time[s]")
            ax.set_ylabel("frequency")
            ax.legend(loc="lower right")
            # ax.legend(bbox_to_anchor=(1.05, 0.0), loc='lower left')
            fig.tight_layout()
            plt.savefig(
                f"fig/{groups}/compare_histogram/{groups}_{data_type}_compare_{'normed graph' if density else 'histogram'}.png"
            )
            plt.show()
            plt.close()

    @staticmethod
    def plot_compare_graph_cumlative_durations_and_intervals(data_lists, params):
        data_types = ["eat duration", "eat interval", "touch duration", "touch interval"]
        groups = "-".join([param.get("group", "noname_group") for param in params])
        if not os.path.exists(f"fig/{groups}/compare_cumlative"):
            os.makedirs(f"fig/{groups}/compare_cumlative")
        for data_type in data_types:
            fig, ax = plt.subplots()
            range_max = max(
                [
                    data[data_types.index(data_type)]
                        .filter(["duration", "interval"])
                        .values.max()
                    for data in data_lists
                ]
            )
            # TODO
            ## durationに何が入っている前提だったか
            ## 基本的に
            ranges = (0, math.ceil(range_max))
            bins = int(math.ceil(range_max) / 0.5)
            while bins > 40:
                bins = int(bins / 2)
            for data_list, param in zip(data_lists, params):
                raw_df = (
                    data_list[data_types.index(data_type)]
                        .filter(["duration", "interval"])
                        .values
                )
                if not raw_df.size:
                    continue
                # duration or interval
                ax.hist(
                    raw_df,
                    density=True,
                    cumulative=True,
                    label=param["group"],
                    alpha=0.3,
                    range=ranges,
                    bins=bins,
                    ec="black",
                )

            ax.set_title(f"{groups} cumlative {data_type}")
            ax.set_xlabel("Time[s]")
            ax.set_ylabel("frequency")
            ax.legend(loc="lower right")
            # ax.legend(bbox_to_anchor=(1.05, 0.0), loc='lower left')
            fig.tight_layout()
            plt.savefig(
                f"fig/{groups}/compare_cumlative/{groups}_{data_type}_cumlative_histogram.png"
            )
            # plt.show()
            plt.close()


def run_jobs_parallel(jobs):
    joblib.Parallel(n_jobs=-2, verbose=2)(jobs)


if __name__ == "__main__":
    args = sys.argv
    groupA_name = "pelet50_wild"
    groupB_name = "pelet50_penk"
    wild_file = "analyze/data/WT"
    penk_file = "analyze/data/penk"
    wild_file = "analyze/data/WT_test"
    penk_file = "analyze/data/penk_test"
    wild_files = FileIo.read_items(wild_file)
    penk_files = FileIo.read_items(penk_file)
    # wild_dfs = export_eat_duration_and_interval(wild_file)
    # penk_dfs = export_eat_duration_and_interval(penk_file, {"eat_distances": 50, "touch_distances": 50})
    # export_diff_wild_penk(wild_dfs, penk_dfs)
    if False:
        # if len(args) <= 1:
        # 一例として距離が60pxくらい
        wild_dfs, penk_dfs = joblib.Parallel(n_jobs=-2, verbose=2)(
            [
                joblib.delayed(export_eat_duration_and_interval)(
                    file,
                    {
                        "group": f"{type_name}",
                        "eat_frames": 60,
                        "touch_frames": 20,
                        "eat_distances": 50,
                        "touch_distances": 50,
                    },
                )
                for file, type_name in zip([wild_file, penk_file], [groupA_name, groupB_name])
            ]
        )
        # wild_dfs = export_eat_duration_and_interval(
        #     wild_file,
        #     {"group": groupA_name, "eat_distances": 100, "touch_distances": 100},
        # )
        # penk_dfs = export_eat_duration_and_interval(
        #     penk_file,
        #     {"group": groupB_name, "eat_distances": 100, "touch_distances": 100},
        # )

        # plot_cumlative_durations_and_intervals(wild_dfs, {"group": groupA_name})
        # plot_cumlative_durations_and_intervals(penk_dfs, {"group": groupB_name})
        joblib.Parallel(n_jobs=-2, verbose=2)(
            [
                joblib.delayed(plot_cumlative_durations_and_intervals)(
                    wild_dfs, {"group": groupA_name, "mouse_id": no}
                )
                for no in wild_files.keys()
            ]
            + [
                joblib.delayed(plot_cumlative_durations_and_intervals)(
                    penk_dfs, {"group": groupB_name, "mouse_id": no}
                )
                for no in penk_files.keys()
            ]
            + [
                joblib.delayed(plot_compare_graph_cumlative_durations_and_intervals)(
                    [wild_dfs, penk_dfs],
                    [{"group": groupA_name}, {"group": groupB_name}],
                )
            ]
            + [
                joblib.delayed(plot_compare_graph_durations_and_intervals)(
                    [wild_dfs, penk_dfs],
                    [{"group": groupA_name}, {"group": groupB_name}],
                )
            ]
            + [
                joblib.delayed(plot_compare_graph_durations_and_intervals)(
                    [wild_dfs, penk_dfs],
                    [{"group": groupA_name}, {"group": groupB_name}],
                    density=True,
                )
            ]
            + [joblib.delayed(export_diff_wild_penk)([wild_dfs, penk_dfs],
                                                     [{"group": groupA_name}, {"group": groupB_name}])]
            + [joblib.delayed(export_analysed_data)(data, name) for data, name in
               zip([wild_dfs, penk_dfs], [groupA_name, groupB_name])]
            # + [joblib.delayed()()]
        )

        # joblib.Parallel(n_jobs=-2, verbose=2)(
        #     [
        #         joblib.delayed(plot_cumlative_durations_and_intervals)(
        #             penk_dfs, {"group": groupB_name, "mouse_id": no}
        #         )
        #         for no in penk_files.keys()
        #     ]
        # )
        # for no in wild_files.keys():
        #     plot_cumlative_durations_and_intervals(
        #         wild_dfs, {"group": groupA_name, "mouse_id": no}
        #     )
        # for no in penk_files.keys():
        #     plot_cumlative_durations_and_intervals(
        #         penk_dfs, {"group": groupB_name, "mouse_id": no}
        #     )
        #
        # plot_compare_graph_cumlative_durations_and_intervals(
        #     [wild_dfs, penk_dfs], [{"group": groupA_name}, {"group": groupB_name}]
        # )
        # plot_compare_graph_durations_and_intervals(
        #     [wild_dfs, penk_dfs], [{"group": groupA_name}, {"group": groupB_name}]
        # )
        # plot_compare_graph_durations_and_intervals(
        #     [wild_dfs, penk_dfs],
        #     [{"group": groupA_name}, {"group": groupB_name}],
        #     density=True,
        # )
        # export_diff_wild_penk([wild_dfs, penk_dfs], [{"group": groupA_name}, {"group": groupB_name}], )
        # export_analysed_data(wild_dfs, groupA_name)
        # export_analysed_data(penk_dfs, groupB_name)
    if False:
        dfs = []
        for groups_dir in args[1:]:
            filepath_dict = FileIo.read_dircsv(groups_dir)
            dfs.append(
                export_eat_duration_and_interval(
                    filepath_dict, {"group": args[1:].index(groups_dir)}
                )
            )
    if True:
        # Class debug
        # TODO
        #   グループ名指定を柔軟にする
        #
        wild_data = MouseGroup(wild_file, {
            "group": f"{groupA_name}",
            "eat_frames": 60,
            "touch_frames": 20,
            "eat_distances": 50,
            "touch_distances": 50,
        })
        penk_data = MouseGroup(penk_file, {
            "group": f"{groupB_name}",
            "eat_frames": 60,
            "touch_frames": 20,
            "eat_distances": 50,
            "touch_distances": 50,
        })
        compare_groups = GroupCompare([wild_data, penk_data])
    # 設定について
    ## 1080p の場合
    ## "eat_frames", 30
    ## "eat_distances", 100
    ## "touch_frames", 10
    ## "touch_distances", 100

    ## 720p の場合
    ## "eat_frames", 30 / 1.5 = 20
    ## "eat_distances", 100 / 1.5 = 66.6
    ## "touch_frames", 10 / 1.5 = 6.6
    ## "touch_distances", 100 / 1.5 = 66.6
