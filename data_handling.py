import pandas as pd
from pathlib import Path
import soundfile as sf
import librosa
import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    train_data1, train_data2, val_data, test_data = read_data_from_csv()
    save = False
    if save:
        save_features(train_data1["files"], train_data1["labels"], "train1")
        save_features(train_data2["files"], train_data2["labels"], "train2")
        save_features(val_data["files"], val_data["labels"], "val")
        save_features(test_data["files"], test_data["labels"], "test")


def read_data_from_csv():
    """
    Reads data from csv files and constructs a data structure containing 
    file paths and labels for each dataset
    """

    train_df1 = pd.read_csv("bsdk10k-splits/bsd10k-train.csv")
    val_df = pd.read_csv("bsdk10k-splits/bsd10k-val.csv")
    test_df = pd.read_csv("bsdk10k-splits/bsd10k-test.csv")

    # file paths for 10k dataset audios
    audio_path = Path("audio")
    train_files1 = [f"{audio_path}/{sound_id}.wav" for sound_id in train_df1["sound_id"]]
    val_files = [f"{audio_path}/{sound_id}.wav" for sound_id in val_df["sound_id"]]
    test_files = [f"{audio_path}/{sound_id}.wav" for sound_id in test_df["sound_id"]]

    # labels for 10k dataset audios
    train_labels1 = train_df1["class_idx"].to_list()
    val_labels = val_df["class_idx"].to_list()
    test_labels = test_df["class_idx"].to_list()

    # load 35k audio
    audio_path2 = Path("BSD35k-CS-draft/audio")
    train_df2 = pd.read_csv("BSD35k-CS-draft/metadata/BSD35k-CS-draft-metadata.csv")

    # map 35k labels to the same format as 10k labels
    class_to_id = dict(zip(train_df1["class"], train_df1["class_idx"]))
    train_df2["class_idx"] = train_df2["class"].map(class_to_id)
    train_df2 = train_df2.dropna(subset=["class_idx"])

    train_files2 = [f"{audio_path2}/{sound_id}.wav" for sound_id in train_df2["sound_id"]]
    train_labels2 = train_df2["class_idx"].to_list()

    # concatenate train data
    train_files = train_files1 + train_files2
    train_labels = train_labels1 + train_labels2

    # split train data into half for easier data storing
    middle_idx = int(len(train_labels)//2)
    
    train_files1 = train_files[:middle_idx]
    train_labels1 = train_labels[:middle_idx]

    train_files2 = train_files[middle_idx:]
    train_labels2 = train_labels[middle_idx:]

    train_data1 ={
        "files": train_files1,
        "labels": train_labels1}
    
    train_data2 ={
        "files": train_files2,
        "labels": train_labels2}
    
    val_data ={
        "files": val_files,
        "labels": val_labels}
    
    test_data ={
        "files": test_files,
        "labels": test_labels}
    
    return train_data1, train_data2, val_data, test_data


def extract_mel(file):
    y, sr = sf.read(file)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024, n_mels=128)
    return S


def save_features(files, labels, split):
    """
    Extracts features and saves them along with labels into a pkl file.
    
    :param files: list of file paths
    :param labels: list of labels
    :param split: data split (str)
    """
    counter = 0
    features = []
    for file in files:
        counter += 1
        features.append(extract_mel(file))
        if counter % 1000 == 0:
            print(counter)
            
    data = {
        "features": features,
        "labels": labels}
    
    with open(f"features_and_labels_{split}.pkl", "wb") as f:
        pickle.dump(data, f)


def read_data_from_pkl():
    with open("features_and_labels_train1.pkl", "rb") as f:
        train_data1 = pickle.load(f)

    with open("features_and_labels_train2.pkl", "rb") as f:
        train_data2 = pickle.load(f)

    with open("features_and_labels_val.pkl", "rb") as f:
        val_data = pickle.load(f)

    with open("features_and_labels_test.pkl", "rb") as f:
        test_data = pickle.load(f)

    return train_data1, train_data2, val_data, test_data

if __name__ == "__main__":
    main()