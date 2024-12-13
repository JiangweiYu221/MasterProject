import os

subject = "zhy"
for i in range(1,7):
    run = subject + str(i)
    path = "D:\qq文件\交接代码\伪迹数据\已采数据/" + subject + "/" + run
    if os.path.exists(path) and os.path.isdir(path):
        new_raw = "data_raw.bdf"
        old_raw = "data.bdf"
        old_2 = "data2.bdf"
        new_2 = "data.bdf"
        old_path_raw = os.path.join(path, old_raw)
        old_path_2 = os.path.join(path, old_2)
        new_path_raw = os.path.join(path, new_raw)
        new_path_2 = os.path.join(path, new_2)
        os.rename(old_path_raw, new_path_raw)
        os.rename(old_path_2, new_path_2)
