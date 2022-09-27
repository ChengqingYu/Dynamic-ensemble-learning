clc;clear;close all;

%% 载入数据
load data_all_wind;

%% 数据处理
series   = data_7;  %现在在数据7上获得了不错的结果
inNum    = 20;
outNum   = 1;
bias     = 1;           %最小值为1
trainNum = 1500;
validNum = 300;
testNum  = 200;
dataType = '';
dataset = getAllDataset(series,inNum,outNum,bias,trainNum,validNum,testNum,dataType);

save('dataset.mat','dataset');

%% 统计信息
load dataset
statistic(dataset.series)

%% 绘图
fileName   = '';
root       = ['C:\Users\liuhui116\Desktop\铁路风工程\图片\',fileName,'\'];
fileType   = '-dpng';
resolution = '-r600';

figureName = 'Wind speed time series';
figure('Name',figureName);
tValue=[dataset.trainY,dataset.validY,dataset.testY];
x = 0:length(tValue)-1;
plot(x(1:trainNum),tValue(1:trainNum), ...
    x(trainNum+1:trainNum+validNum),tValue(trainNum+1:trainNum+validNum), ...
    x(trainNum+validNum+1:end),tValue(trainNum+validNum+1:end), ...
    'LineWidth',     1);
legend('Training Set','Validation Set','Testing Set', ...
    'Orientation',   'vertical', ...
    'FontSize',      11, ...
	'FontName',      'Times New Roman');
xlabel('Times (3-min)', ...
    'FontSize',      12, ...
	'FontName',      'Times New Roman', ...
    'FontWeight',    'bold');
ylabel('Wind speed (m/s)', ...
    'FontSize',      12, ...
	'FontName',      'Times New Roman', ...
    'FontWeight',    'bold');
grid on;
% set(gca,'FontName','Times New Roman','FontSize',20);
set(gcf,'visible','off');
set(gcf,'Position',[200,200,700,350]);
print(gcf,[root,figureName],fileType,resolution);

