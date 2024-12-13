import mne
import numpy as np
import matplotlib.pyplot as plt
import os

# path = "D:\qq文件\交接代码\伪迹数据\mne_edf/ft3_data.edf"
# raw = mne.io.read_raw_edf(path, preload=True)
# raw.plot()
# plt.show(block=True)#展示原数据

subject = "zhy"
for i in range(1,7):
    run = subject + str(i)
    path = "D:\qq文件\交接代码\伪迹数据\已采数据/" + subject + "/" + run
    if os.path.exists(path) and os.path.isdir(path):
        file_path = os.path.join(path, "data.bdf")
        raw = mne.io.read_raw_bdf(file_path, preload=True)
        # raw.plot()
        # plt.show()
        # psd = raw.compute_psd()
        # psd.plot()
        # plt.show(block=True)
        freqs= np.arange(50, 251, 50)
        raw = raw.notch_filter(freqs=freqs)
        raw = raw.filter(h_freq=150, l_freq=0.5)
        # psd = raw.compute_psd()
        # psd.plot()
        # plt.show(block=True)
        L_anode = ["Fp1","F7","T7","P7","Fp1","F3","C3","P3","Fz","Cz","Fp2","F4","C4","P4","Fp2","F8","T8","P8"]
        L_cathode = ["F7","T7","P7","O1","F3","C3","P3","O1","Cz","Pz","F4","C4","P4","O2","F8","T8","P8","O2"]
        T_anode = ["F7","Fp1","F7","F3","Fz","F4","T7","C3","Cz","C4","P7","P3","Pz","P4","O1","O2"]
        T_cathode = ["Fp1","Fp2","F3","Fz","F4","F8","C3","Cz","C4","T8","P3","Pz","P4","P8","O2","P8"]

        L_pick = ['Fp1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fz-Cz', 'Cz-Pz', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp2-F8', 'F8-T8', 'T8-P8', 'P8-O2']
        T_pick = ['F7-Fp1', 'Fp1-Fp2', 'F7-F3', 'F3-Fz', 'Fz-F4', 'F4-F8', 'T7-C3', 'C3-Cz', 'Cz-C4', 'C4-T8', 'P7-P3', 'P3-Pz', 'Pz-P4', 'P4-P8', 'O1-O2', 'O2-P8']
        L_pick.extend(T_pick)
        L_raw_bip_ref = mne.set_bipolar_reference(raw, anode=L_anode, cathode=L_cathode,drop_refs=False)
        LT_raw_bip_ref = mne.set_bipolar_reference(L_raw_bip_ref, anode=T_anode, cathode=T_cathode)
        bip = LT_raw_bip_ref.pick(L_pick)
        export_path = "D:\qq文件\交接代码\伪迹数据\mne_set/" + run + "_data.set"
        bip.export(export_path, overwrite=True)





