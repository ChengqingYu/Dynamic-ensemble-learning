%% 加载数据
clear;clc;
load data_all_wind.mat
%% 数据处理
dataset = getAllDataset(data_2,5,1,1500,300,200);
save('dataset.mat','dataset');

%% 统计信息
info = statistic(dataset.series)