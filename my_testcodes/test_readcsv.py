# coding:utf-8

##############################
# TODO
## 食事間のインターバルを検出
##############################


import numpy as np
import pandas as pd
from scipy.spatial import distance
import itertools
import matplotlib.pyplot as plt


def read_csvfile(file_path=r"D:\DeeplabCut\movie\210617_no2\210617-no2DLC_resnet101_guitest6-25shuffle1_50000.csv"):
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
    eating_duration = eating_frames.end - eating_frames.start
    # plot
    eating_duration.hist()
    plt.savefig("fig/eating_duration_hist.png")
    plt.show()
    plt.close()
    return eating_duration


############################## reference ##############################

def calc_INI(mice) -> pd.DataFrame:
    """
    最後のノーズポークから次のノーズポークまで(INI=inter nosepoke interval)を算出する
    No TC, TC100, TC70, 及びそれ以降のタスクを対象とする(指定はしない)
    マウス・タスク毎の平均データと生データlistを返す
    INIの単位は秒
    生データは呼び出し側で、マウス・タスク毎にグラフ化(INIの逆数をplot, 見づらければ10回移動平均)
    タスク中にINIのデータが存在しない場合の例外処理を用意する
    :param mice: mouse_idのリスト
    :return: DataFrame, columns=['mouse_id', 'task', 'INI_list'];　unique key=(mouse_id, task)
    """

    df_summary = pd.DataFrame(columns=['mouse_id', 'task', 'INI_list'])

    for mouse_id in mice:
        data = get_data(mouse_id).assign(session_id=_get_session_id(mouse_id))
        tasks_in_log = data.task.unique().tolist()

        for task in tasks_in_log:
            if task in ["T0"]:
                continue
            current_data = data[data.task == task]

            # rewardがある場合　magazine nose poked
            # failureの場合 nose poke
            # omissionの場合 task called
            def _calc_session_INI(session_id):
                nonlocal current_data
                current_session = current_data[current_data.session_id.isin([session_id])]
                next_session = current_data[current_data.session_id.isin([session_id + 1])]
                # 次がない場合
                if next_session.empty:
                    return np.nan
                # 対象のセッションのcorrect/incorrect/omissionによる処理分け
                if not current_session[current_session.event_type.isin(["time over"])].empty:
                    current_session = current_session[current_session.event_type.isin(["task called"])]
                elif not current_session[current_session.event_type.isin(["reward"])].empty:
                    current_session = current_session[current_session.event_type.isin(["nose poke"])]
                #                    current_session = current_session[current_session.event_type.isin(["magazine nose poked"])]
                elif not current_session[current_session.event_type.isin(["failure"])].empty:
                    current_session = current_session[current_session.event_type.isin(["nose poke"])]
                elif not current_session[current_session.event_type.isin(["nose poke"])].empty:
                    current_session = current_session[current_session.event_type.isin(["nose poke"])]
                # next session
                if next_session[next_session.event_type.isin(["task called"])].empty:
                    # task callがないタスクの時
                    # これに引っかからないのはT0のみ
                    next_session = next_session[next_session.event_type.isin(["nose poke"])]
                else:
                    next_session = next_session[next_session.event_type.isin(["nose poke"])]
                #                    next_session = next_session[next_session.event_type.isin(["task called"])]
                # 次のトライアルがstartで止まってる場合
                if next_session.empty:
                    return np.nan
                INI = next_session.timestamps.values - current_session.timestamps.values
                return np.timedelta64(INI[0], "s")

            tmp_INI = np.array(list(map(_calc_session_INI, current_data.session_id.unique().tolist())))
            df_summary = df_summary.append(
                pd.DataFrame({"mouse_id": mouse_id, "task": task,
                              "INI_list": tmp_INI[~pd.isna(tmp_INI)].astype(np.timedelta64) / np.timedelta64(1, "s")}),
                ignore_index=True)
    return df_summary


def calc_IMI(mice) -> pd.DataFrame:
    """
    最後のマガジンノーズポーク(reward後 or task call)から次のマガジンノーズポークまで(IMI=inter magazine nosepoke interval)を算出する
    T0 (reward), No TC, TC100, TC70, 及びそれ以降のタスクを対象とする (T0のみ明示的に含めるようにコーディング)
    マウス・タスク毎の平均データと生データlistを返す
    INI, IMIの単位は秒
    生データは呼び出し側で、マウス・タスク毎にグラフ化(IMIの逆数をplot, 見づらければ10回移動平均)
    タスク中にIMIのデータが存在しない場合の例外処理を用意する
    :param mice: mouse_idのリスト
    :return: DataFrame, columns=['mouse_id', 'task', 'mean_IMI', 'IMI_list'];　unique key=(mouse_id, task)
    """
    df_summary = pd.DataFrame(columns=['mouse_id', 'task', 'INI_list'])
    for mouse_id in mice:
        data = get_data(mouse_id).assign(session_id=_get_session_id(mouse_id))
        tasks_in_log = data.task.unique().tolist()
        for task in tasks_in_log:
            current_data = data[data.task == task]

            def _calc_session_IMI(session_id):
                nonlocal current_data
                current_session = current_data[current_data.session_id.isin([session_id])]
                next_session = current_data[current_data.session_id.isin([session_id + 1])]
                if task == "T0":
                    current_session = current_session[current_session.event_type.isin(["reward"])]
                    next_session = next_session[next_session.event_type.isin(["reward"])]
                    if next_session.empty:
                        return np.nan
                else:
                    # 最後のトライアルなら計算しない
                    if next_session.empty:
                        return np.nan
                    # current session
                    if not current_session[current_session.event_type.isin(["task called"])].empty:
                        current_session = current_session[current_session.event_type.isin(["task called"])]
                    elif not current_session[current_session.event_type.isin(["magazine nose poked"])].empty:
                        current_session = current_session[
                            current_session.event_type.isin(["magazine nose poked"])]  # for No TC
                    #                     elif not current_session[current_session.event_type.isin(["time over"])].empty:
                    #                         current_session = current_session[current_session.event_type.isin(["task called"])]
                    #                     elif not current_session[current_session.event_type.isin(["reward"])].empty:
                    #                         current_session = current_session[current_session.event_type.isin(["task called"])]
                    # #                        current_session = current_session[current_session.event_type.isin(["magazine nose poked"])]
                    #                     elif not current_session[current_session.event_type.isin(["failure"])].empty:
                    #                         current_session = current_session[current_session.event_type.isin(["task called"])]
                    # 該当するデータがなければ計算しない
                    if current_session.empty:
                        return np.nan
                    # next session
                    if next_session[next_session.event_type.isin(["task called"])].empty:
                        # task callがないタスクの時
                        next_session = next_session[next_session.event_type.isin(["magazine nose poked"])]
                        if next_session.empty:
                            return np.nan
                    else:
                        next_session = next_session[next_session.event_type.isin(["task called"])]
                IMI = next_session.timestamps.values - current_session.timestamps.values
                return np.timedelta64(IMI[0], "s")

            tmp_IMI = np.array(list(map(_calc_session_IMI, current_data.session_id.unique().tolist())))
            df_summary = df_summary.append(
                pd.DataFrame({"mouse_id": mouse_id, "task": task,
                              "IMI_list": tmp_IMI[~pd.isna(tmp_IMI)].astype(np.timedelta64) / np.timedelta64(1, "s")}),
                ignore_index=True)
    return df_summary


def calc_IMI_after(mice, sort_type) -> pd.DataFrame:
    """
    最後のマガジンノーズポーク(reward後 or task call)から次のマガジンノーズポークまで(IMI=inter magazine nosepoke interval)を算出する
    T0 (reward), No TC, TC100, TC70, 及びそれ以降のタスクを対象とする (T0のみ明示的に含めるようにコーディング)
    マウス・タスク毎の平均データと生データlistを返す
    INI, IMIの単位は秒
    生データは呼び出し側で、マウス・タスク毎にグラフ化(IMIの逆数をplot, 見づらければ10回移動平均)
    タスク中にIMIのデータが存在しない場合の例外処理を用意する
    :param mice: mouse_idのリスト
    :return: DataFrame, columns=['mouse_id', 'task', 'mean_IMI', 'IMI_list'];　unique key=(mouse_id, task)
    """
    df_summary = pd.DataFrame(columns=['mouse_id', 'task', 'INI_list'])
    for mouse_id in mice:
        data = get_data(mouse_id).assign(session_id=_get_session_id(mouse_id))
        tasks_in_log = data.task.unique().tolist()
        for task in tasks_in_log:
            print(f"mouse_id={mouse_id}, task={task}")
            current_data = data[data.task == task]

            def _calc_session_IMI(session_id):
                nonlocal current_data
                current_session = current_data[current_data.session_id.isin([session_id])]
                next_session = current_data[current_data.session_id.isin([session_id + 1])]
                if task == "T0":
                    current_session = current_session[current_session.event_type.isin(["reward"])]
                    next_session = next_session[next_session.event_type.isin(["reward"])]
                    if next_session.empty:
                        return np.nan
                else:
                    # 該当するデータがなければ計算しない
                    if current_session.empty:
                        return np.nan
                    # 最後のトライアルなら計算しない
                    if next_session.empty:
                        return np.nan
                    if sort_type == 'win':
                        if not current_session[current_session.event_type.isin(["reward"])].empty:
                            current_session = current_session[current_session.event_type.isin(["reward"])]
                        else:
                            return np.nan
                    elif sort_type == 'lose':
                        if not current_session[current_session.event_type.isin(["failure"])].empty:
                            current_session = current_session[current_session.event_type.isin(["failure"])]
                        else:
                            return np.nan
                    else:
                        print(f'not defined sort type: {sort_type}')
                        return np.nan
                    # next session
                    if not next_session[next_session.event_type.isin(["task called"])].empty:
                        next_session = next_session[next_session.event_type.isin(["task called"])]
                    else:
                        return np.nan
                IMI = next_session.timestamps.values - current_session.timestamps.values
                return np.timedelta64(IMI[0], "s")

            tmp_IMI = np.array(list(map(_calc_session_IMI, current_data.session_id.unique().tolist())))
            df_summary = df_summary.append(
                pd.DataFrame({"mouse_id": mouse_id, "task": task,
                              "IMI_list": tmp_IMI[~pd.isna(tmp_IMI)].astype(np.timedelta64) / np.timedelta64(1, "s")}),
                ignore_index=True)
    return df_summary


def do_calc_INI_IMI(mouse_group_name, time_window=1):
    mouse_group_dict = get_mouse_group_dict()
    choice_mouse_group_dict = mouse_group_dict[mouse_group_name]
    mice = choice_mouse_group_dict["mice"]
    IMI_tasks = choice_mouse_group_dict["IMI_tasks_section"]
    INI_tasks = choice_mouse_group_dict["INI_tasks_section"]

    # INI CSV出力
    df_INI = calc_INI(mice)
    # TODO: タスク毎にcsv出力 (データはmouse_id, mean_INI)
    #   df_INI_task.to_csv(f"./data/inter/INI_{mouse_group_name}_{task}.csv")
    for task in INI_tasks:
        df_INI_task = df_INI[df_INI.task.isin([task])]
        df_INI_task.to_csv(f"./data/inter/INI_{mouse_group_name}_{task}.csv")

        # INI グラフ出力
        # TODO df_INIに含まれるマウス・タスク毎のINI_listを基に、その生データをplot保存(eps, png)
        #   横軸:trial, 縦軸:INIの逆数の移動平均(time_window)
        #   f"./fig/inter/INI_{mouse_group_name}_no{mouse_id}_{task}.png"
        for mouse_id in df_INI_task.mouse_id.unique().tolist():
            INI_data = df_INI_task[df_INI_task.mouse_id.isin([mouse_id])]
            #            data = INI_data.INI_list.rdiv(1)
            data = INI_data.INI_list
            moving_average = data.rolling(window=time_window).mean()
            plt.plot(range(len(moving_average)), moving_average)
            plt.title(f"INI mouseid:{mouse_id} task:{task}")
            plt.savefig(f"./fig/inter/INI_{mouse_group_name}_no{mouse_id}_{task}.png")
            plt.show()

    # IMI CSV出力
    df_IMI = calc_IMI(mice)
    print(df_IMI)
    # TODO: タスク毎にcsv出力 (データはmouse_id, mean_INI)
    #    df_IMI_task.to_csv(f"./data/inter/IMI_{mouse_group_name}_{task}.csv")

    # IMI グラフ出力
    # TODO df_IMIに含まれるマウス・タスク毎のIMI_listを基に、その生データをplot保存(eps, png)
    #   横軸:trial, 縦軸:INIの逆数の移動平均(time_window)
    #   f"./fig/inter/IMI_{mouse_group_name}_no{mouse_id}_{task}.png"
    for task in IMI_tasks:
        df_IMI_task = df_IMI[df_IMI.task.isin([task])]
        df_IMI_task.to_csv(f"./data/inter/IMI_{mouse_group_name}_{task}.csv")

        for mouse_id in df_IMI_task.mouse_id.unique().tolist():
            IMI_data = df_IMI_task[df_IMI_task.mouse_id.isin([mouse_id])]
            #            data = IMI_data.IMI_list.rdiv(1)
            data = IMI_data.IMI_list
            moving_average = data.rolling(window=time_window).mean()
            plt.plot(range(len(moving_average)), moving_average)
            plt.title(f"IMI mouseid:{mouse_id} task:{task}")
            plt.savefig(f"./fig/inter/IMI_{mouse_group_name}_no{mouse_id}_{task}.png")
            plt.show()


def do_hist_IMI(mouse_group_names):
    mouse_group_dict = get_mouse_group_dict()

    reset_plot_style()

    df_IMI = []

    for i, mouse_group_name in enumerate(mouse_group_names):
        choice_mouse_group_dict = mouse_group_dict[mouse_group_name]
        mice = choice_mouse_group_dict["mice"]
        IMI_tasks = choice_mouse_group_dict["IMI_tasks_section"]

        # IMI CSV出力
        df_IMI_tmp = calc_IMI(mice)
        df_IMI_tmp['group_name'] = mouse_group_name
        if i == 0:  # TODO 宝田君 ここカッコよい感じにできますか？
            df_IMI = df_IMI_tmp
        else:
            df_IMI = pd.concat([df_IMI, df_IMI_tmp])

    print(df_IMI)

    for task in IMI_tasks:  # 適当過ぎ 要修正
        df_IMI_task = df_IMI[df_IMI.task.isin([task])]
        list_group = list()
        for mouse_group_name in mouse_group_names:
            list_group.append(df_IMI_task[df_IMI_task.group_name == mouse_group_name].IMI_list)
        plt.hist(list_group, bins=10, range=(0, 100), stacked=False, density=True, label=mouse_group_names)
        plt.title(f'{task}')
        plt.legend()
        plt.xlabel('Inter magazine-nosepoke interval (sec)')
        plt.ylabel('Density')
        plt.savefig(f"./fig/inter/IMI_{'-'.join(mouse_group_names)}_{task}.png", bbox_inches='tight', pad_inches=0.05)
        plt.show()


def do_hist_IMI_after(mouse_group_names, sort_type='win', upper_limit=150, hist_bins=15, hist_range=(0, 150)):
    mouse_group_dict = get_mouse_group_dict()

    reset_plot_style()

    df_summary = pd.DataFrame(columns=['mouse_group_name', 'mouse_id', 'task',
                                       'interval_median_all', 'interval_median_part',
                                       'interval_mean_all', 'interval_mean_part',
                                       ])
    df_IMI = []

    task_all_list = list()
    for i, mouse_group_name in enumerate(mouse_group_names):
        choice_mouse_group_dict = mouse_group_dict[mouse_group_name]
        mice = choice_mouse_group_dict["mice"]
        IMI_tasks = choice_mouse_group_dict["IMI_tasks_section"]
        task_all_list.extend(IMI_tasks)

        # IMI CSV出力
        df_IMI_tmp = calc_IMI_after(mice, sort_type)
        df_IMI_tmp['group_name'] = mouse_group_name
        if i == 0:
            df_IMI = df_IMI_tmp
        else:
            df_IMI = pd.concat([df_IMI, df_IMI_tmp])

        # TODO df_summary :
        #  interval_median_all　= 各マウス各タスク中の全部のIMIの中央値
        #  interval_mean_all　= 各マウス各タスク中の全部のIMIの平均値
        #  interval_median_part　= 各マウス各タスク中のIMIのうち、upper_limit未満の中央値
        #  interval_mean_part　= 各マウス各タスク中のIMIのうち、upper_limit未満の平均値
        #  各タスクはIMI_tasksで指定したもの
        #  各タスク毎とは別に、IMI_tasksに含まれるすべてのタスクの各マウス毎の上記4つの指標を1行追加
        for mouse_id in mice:
            current_mouse_IMI_df = df_IMI[df_IMI.mouse_id.isin([mouse_id])]
            if current_mouse_IMI_df.empty:
                continue
            for task in IMI_tasks:
                current_IMI = current_mouse_IMI_df[current_mouse_IMI_df.task.isin([task])]
                if current_IMI.empty:
                    continue
                current_IMI = current_IMI.IMI_list
                interval_median_all = current_IMI.median()
                interval_mean_all = current_IMI.mean()
                interval_median_part = current_IMI[current_IMI < upper_limit].median()
                interval_mean_part = current_IMI[current_IMI < upper_limit].mean()
                df_summary = df_summary.append(
                    {'mouse_group_name': mouse_group_name, 'mouse_id': mouse_id, 'task': task,
                     'interval_median_all': interval_median_all, 'interval_median_part': interval_median_part,
                     'interval_mean_all': interval_mean_all, 'interval_mean_part': interval_mean_part},
                    ignore_index=True)
            # all task
            current_mouse_IMI = current_mouse_IMI_df.IMI_list
            interval_median_all = current_mouse_IMI.median()
            interval_mean_all = current_mouse_IMI.mean()
            interval_median_part = current_mouse_IMI[current_mouse_IMI < upper_limit].median()
            interval_mean_part = current_mouse_IMI[current_mouse_IMI < upper_limit].mean()
            df_summary = df_summary.append(
                {'mouse_group_name': mouse_group_name, 'mouse_id': mouse_id, 'task': "alltask",
                 'interval_median_all': interval_median_all, 'interval_median_part': interval_median_part,
                 'interval_mean_all': interval_mean_all, 'interval_mean_part': interval_mean_part},
                ignore_index=True)

    print(df_summary)
    df_summary.to_csv(f"./data/interval/interval_{'-'.join(mouse_group_names)}_{sort_type}.csv")

    for task in list(dict.fromkeys(task_all_list)):
        df_IMI_task = df_IMI[df_IMI.task.isin([task])]
        list_group = list()
        for mouse_group_name in mouse_group_names:
            #            list_group.append(df_IMI_task[df_IMI_task.group_name == mouse_group_name].IMI_list)
            list_group.append(df_IMI_task.query('group_name==@mouse_group_name and IMI_list<@upper_limit').IMI_list)
        plt.hist(list_group, bins=hist_bins, range=hist_range, stacked=False, density=True, label=mouse_group_names)
        for mouse_group_name in mouse_group_names:
            plt.vlines(np.average(
                df_summary.query('mouse_group_name==@mouse_group_name and task==@task')['interval_median_part']),
                ymin=0,
                ymax=plt.gca().get_ylim()[1], colors='k')
        plt.title(f'{task}')
        plt.legend()
        plt.xlabel(f'Interval to retry after {sort_type} (sec)')
        plt.ylabel('Density')
        plt.savefig(f"./fig/interval/IMI_{sort_type}_{'-'.join(mouse_group_names)}_{task}.png", bbox_inches='tight',
                    pad_inches=0.05)
        plt.show(block=False)
        plt.clf()
        plt.close()

    for i, mouse_group_name in enumerate(mouse_group_names):
        choice_mouse_group_dict = mouse_group_dict[mouse_group_name]
        mice = choice_mouse_group_dict["mice"]
        IMI_tasks = choice_mouse_group_dict["IMI_tasks_section"]

        for task in IMI_tasks:
            df_IMI_task = df_IMI[df_IMI.task.isin([task])]
            for mouse_id in mice:
                list_group = list()
                #                list_group.append(df_IMI_task[df_IMI_task.mouse_id == mouse_id].IMI_list)
                list_group.append(df_IMI_task.query('mouse_id==@mouse_id and IMI_list<@upper_limit').IMI_list)
                plt.hist(list_group, bins=hist_bins, range=hist_range, stacked=False, density=True,
                         label=f"no.{mouse_id}")
                plt.vlines(df_summary.query('mouse_id==@mouse_id and task==@task')['interval_median_part'], ymin=0,
                           ymax=plt.gca().get_ylim()[1], colors='k')
                plt.title(f'{mouse_group_name} no.{mouse_id} - {task} - {sort_type}')
                plt.legend()
                plt.xlabel(f'Interval to retry after {sort_type} (sec)')
                plt.ylabel('Density')
                plt.savefig(f"./fig/interval_each/IMI_{sort_type}_mouse_{mouse_group_name}_{task}_{mouse_id}.png",
                            bbox_inches='tight',
                            pad_inches=0.05)
                plt.show(block=False)
                plt.clf()
                plt.close()


## pass
def get_data(mouse_id):
    pass


def _get_session_id(mouse_id):
    pass


def get_mouse_group_dict():
    pass


def reset_plot_style():
    pass


############################## end reference ##############################


def main():
    df = read_csvfile(
        "/Users/ur/PycharmProjects/DeepLabCut_test/my_testcodes/210617-no2DLC_resnet101_guitest6-25shuffle1_50000.csv")
    scene = Scene(df)
    calc_eat_duration(scene)


if __name__ == "__main__":
    main()
