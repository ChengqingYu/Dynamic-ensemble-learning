%% ��������
clear;clc;
load data_all_wind.mat
%% ���ݴ���
dataset = getAllDataset(data_2,5,1,1500,300,200);
save('dataset.mat','dataset');

%% ͳ����Ϣ
info = statistic(dataset.series)