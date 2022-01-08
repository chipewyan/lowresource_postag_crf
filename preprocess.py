import csv
import math
import numpy as np
# from numpy.lib.shape_base import split
import pandas as pd
from sklearn.model_selection import train_test_split

# input_file = "../UD_Tatar-NMCTT/tt_nmctt-ud-test.conllu"
input_file = input("Specify the input conllu file (.conllu):\n") # 
assert input_file.endswith(".conllu"), "Invalid file name"
# output_file = "tatar_posdata.csv"
output_file = input("Specify the output file name (.csv):\n") # 
assert output_file.endswith(".csv"), "Invalid file name" 
header = ["ID", "WORD", "POS"]
body = []

def readfile(input_file: str):
    with open(input_file) as f:
        lines = f.readlines()
    for l in lines:
        if not l.startswith("#"):
            elems = l.split("\t")
            if len(elems) == 1:
                body.append(["", "", ""])
            elif "-" in elems[0]:
                continue
            else:
                id = (elems[0])
                word = elems[1]
                pos = elems[3]
                body.append([id, word, pos])

def writefile(output_file: str):
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(body)

ratio = 0.8

def split_data(df: pd.DataFrame, ratio: int) -> int:
    split_point = round(len(df) * ratio)
    split_point = find_sent_split(split_point)
    return split_point

def find_sent_split(split_point: int) -> int:
    for i in range(len(df)):
        if pd.isna(df.iloc[split_point - i, 0]):
            split_point -= i
            break
        if pd.isna(df.iloc[split_point + i, 0]):
            split_point += i
            break
    return split_point

if __name__ == "__main__":
    readfile(input_file)
    writefile(output_file)
    df = pd.read_csv(output_file)
    # print(df.head(20))
    split_point = split_data(df, ratio)
    train = df[:split_point]
    test = df[split_point:]
    train.to_csv(output_file[:-4] + "-train.csv")
    print("Training data created!")
    test.to_csv(output_file[:-4] + "-test.csv")
    print("Test data created!")