# coding:utf-8

##############################
# TODO
## 食事中の時間を抜き出す
## 食事間のインターバルを検出
##############################


import numpy as np
import pandas as pd
from scipy.spatial import distance


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
            pos.distance(others)


class Scene:
    def __init__(self, df):
        self.lefthand = Part(df[["lefthand[x]", "lefthand[y]", "lefthand[likelihood]"]], "lefthand")
        self.righthand = Part(df[["righthand[x]", "righthand[y]", "righthand[likelihood]"]], "righthand")
        self.leftfoot = Part(df[["leftfoot[x]", "leftfoot[y]", "leftfoot[likelihood]"]], "leftfoot")
        self.rightfoot = Part(df[["rightfoot[x]", "rightfoot[y]", "rightfoot[likelihood]"]], "rightfoot")
        self.mouth = Part(df[["mouth[x]", "mouth[y]", "mouth[likelihood]"]], "mouth")


def mogmog():
    pass


def main():
    df = read_csvfile()
