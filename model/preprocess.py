import numpy as np
import pandas as pd
from mne_bids import BIDSPath,read_raw_bids
from sklearn.model_selection import train_test_split

# 参数
target_sampling_rate = 125  # 目标采样率
window_size = 5  # 窗口大小（秒）
overlap = 2.5  # 重叠时间（秒）
n_samples_per_window = int(window_size * target_sampling_rate)  # 每个窗口的数据点数
labels = ["ver_eyem", "hor_eyem", "blink", "hor_headm", "ver_headm", "tongue",
          "chew", "swallow", "eyebrow", "blink_hor_headm", "blink_ver_headm",
          "blink_eyebrow", "tongue_eyebrow", "swallow_eyebrow"]
# 创建映射字典
label_mapping = {label: i + 2 for i, label in enumerate(labels)}

bids_root = "/home/mnt_disk1/Motion_Artifact/Motion_Artifact/derivatives/preprocessed_BIDS/"
label_root = "/home/mnt_disk1/Motion_Artifact/Motion_Artifact/Manual_Labels/"
save_root = "/home/mnt_disk1/model_result/"
# 计算滑动窗口的起始位置
def get_window_indices(n_samples, step_size):
    return np.arange(0, n_samples - n_samples_per_window + 1, step_size)

# 读取 EEG 数据并预处理
def load_eeg_and_labels(bids_root, subject_id, subject_run, label_root):
    # 读取 BIDS EEG 数据
    task = 'artifact'  # 任务名称
    datatype = 'eeg'
    bids_path = BIDSPath(subject=subject_id, task=task, run=subject_run,
                         root=bids_root, datatype=datatype)
    raw = read_raw_bids(bids_path)
    raw.resample(target_sampling_rate)  # 降采样至 125 Hz
    data = raw.get_data()
    n_samples = data.shape[1]  # 总样本数
    n_channels = data.shape[0]
    step_size = int((window_size - overlap) * target_sampling_rate)  # 滑动步长

    # 读取标签数据
    label_path = label_root + '/sub' +subject_id + '_run0' + str(subject_run) + ".csv"
    df = pd.read_csv(label_path)

    # 处理标签数据
    label_matrix = np.zeros((n_channels, n_samples))  # 初始化标签矩阵
    for _, row in df.iterrows():
        ch_name, start, end, label = row
        if (label != 'close_base') and (label != 'open_base'):
            if (ch_name in raw.ch_names) :
                ch_idx = raw.ch_names.index(ch_name)
                start_idx = int(start * target_sampling_rate)
                end_idx = int(end * target_sampling_rate)
                label_matrix[ch_idx, start_idx:end_idx] = label_mapping[label]  # 标记对应时间段
            elif ch_name == 'ALL' :
                start_idx = int(start * target_sampling_rate)
                end_idx = int(end * target_sampling_rate)
                label_matrix[:, start_idx:end_idx] = label_mapping[label]  # 标记对应时间段
        elif (label == 'close_base') :
            start_idx = int(start * target_sampling_rate)
            end_idx = int(end * target_sampling_rate)
            label_matrix[:, start_idx:end_idx] = 1  # 标记对应时间段

    # 生成窗口数据
    windows = []
    labels = []
    for start_idx in get_window_indices(n_samples, step_size):
        end_idx = start_idx + n_samples_per_window
        selected_window = data[:, start_idx:end_idx]
        selected_label = label_matrix[:, start_idx:end_idx]
        if np.any(selected_label != 0):
            windows.append(selected_window)  # 存入 data 矩阵
            labels.append(selected_label)  # 存入 label 矩阵

    return np.array(windows), np.array(labels)

#构造数据集
def randomize(all_data, all_labels):
    all_data = all_data.astype(np.float32)
    all_labels = all_labels.astype(np.float32)

    data_train, data_test, label_train, label_test = train_test_split(
        all_data, all_labels, test_size=0.2, random_state=42, shuffle=True)
    # 二次分割：训练集分为训练和验证集（80%训练，20%验证）
    data_train, data_val, label_train, label_val = train_test_split(
        data_train, label_train, test_size=0.2, random_state=42, shuffle=True)

    print(data_train.shape, data_test.shape, data_val.shape, label_train.shape,
          label_test.shape, label_val.shape)

    return data_train, data_test, data_val, label_train, label_test, label_val

if __name__ == "__main__":
    all_data, all_labels = [], []
    for id in range(1, 31):
        subject_id = str(id)
        for run in range(1,7):
            data, labels = load_eeg_and_labels(bids_root, subject_id, run, label_root)
            all_data.append(data)
            all_labels.append(labels)
    # 转换为 NumPy 数组
    all_data = np.concatenate(all_data, axis=0)  # 形状: (总窗口数, 22, 375)
    all_labels = np.concatenate(all_labels, axis=0)  # 形状: (总窗口数, 22, 375)
    np.save(save_root + "all_data.npy", all_data)
    np.save(save_root + "all_labels.npy", all_labels)
    data_train, data_test, data_val, label_train, label_test, label_val = randomize(all_data, all_labels)
    np.save(save_root + "data_train.npy", data_train)
    np.save(save_root + "data_test.npy", data_test)
    np.save(save_root + "data_val.npy", data_val)
    np.save(save_root + "label_train.npy", label_train)
    np.save(save_root + "label_test.npy", label_test)
    np.save(save_root + "label_val.npy", label_val)
