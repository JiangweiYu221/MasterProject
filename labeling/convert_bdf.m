set_path = 'D:\qq文件\交接代码\伪迹数据\mne_set';
out_root = 'D:\qq文件\交接代码\伪迹数据\已采数据\';
% 查找以 zhw 开头的文件
subject = 'zhy';
filePattern = fullfile(set_path, [subject,'*']); % 通配符 * 匹配以 zhw 开头的文件
fileList = dir(filePattern);

for i = 1:length(fileList)
    filename = fileList(i).name;
    path = fileList(i).folder;
    % filepath = append(fileList(i).folder, '\', filename);
    EEG = pop_loadset(filename, path);
    out_path = append(out_root, subject, '\', subject, num2str(i), '\data2.bdf');
    pop_writeeeg( EEG, out_path, 'TYPE', 'BDF' );
end