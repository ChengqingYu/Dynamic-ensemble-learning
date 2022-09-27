%% GMDH基预测器

%% 清空环境变量
clc;clear;close all;

%% 训练测试数据
load dataset.mat
trainXn=dataset.trainXn;
trainYn=dataset.trainYn;
testXn=dataset.testXn;
testY=dataset.testY;
outputps=dataset.outputps; 

%% GMDH网络构建
params.MaxLayerNeurons = 25;   % Maximum Number of Neurons in a Layer
params.MaxLayers = 5;          % Maximum Number of Layers
params.alpha = 0;              % Selection Pressure
params.pTrain = 0.8;           % Train Ratio

%% GMDH网络训练
net = GMDH(params, trainXn, trainYn);

%% GMDH网络预测
testYn_out = ApplyGMDH(net, testXn);
testOutput = mapminmax('reverse',testYn_out,outputps);

%% 误差分析
tValue=testY;
pValue=testOutput;
disp('GMDH');
myerror = myError(tValue,pValue,'show');

%% 输出预测结果
xlsName = '.\myResults\results.xlsx';
xlswrite(xlsName,tValue',1,'A');
xlswrite(xlsName,pValue',1,'C');

%% 绘图
close all;
fileName = '';
root = ['C:\Users\Flywi\Desktop\交通流预测\myFigure\',fileName,'\'];
fileType = '-djpeg';
resolution = '-r600';
xTest = 1:dataset.testNum;

% Plot predicted wind speed on testing set
figureName = 'GMDH预测结果';
figure('Name',figureName);
plot(xTest,tValue,xTest,pValue, ...
    'LineWidth',     1);
xlabel('时间 (小时)', ...
    'FontSize',      12, ...
    'FontWeight',    'bold');
ylabel('交通量 （辆 / 时）', ...
    'FontSize',      12, ...
    'FontWeight',    'bold');
legend('观测值','预测值', ...
    'FontSize',      11, ...
    'FontWeight',    'normal');
grid on;
set(gcf,'Position',[200,200,700,350]);
print(gcf,[root,figureName],fileType,resolution);

% Plot scatter plot
figureName = 'GMDH预测散点图';
figure('Name',figureName);
plot(tValue,pValue(),'.', ...
    'MarkerSize',    10);
hold on;
plot(min(tValue):0.001:max(tValue),min(tValue):0.001:max(tValue), ...
    'LineWidth',     1);
axis([min(tValue),max(tValue),min(tValue),max(tValue)]);
axis square;
xlabel('观测交通量 （辆/时）', ...
    'FontSize',     12, ...
    'FontWeight',   'bold');
ylabel('预测交通量 （辆/时）)', ...
    'FontSize',     12, ...
    'FontWeight',   'bold');
text(min(tValue)+500,max(tValue)-500,['PCC = ',num2str(myerror.pcc)], ...
    'FontSize',     12, ....
    'FontWeight',   'normal');
set(gcf,'Position',[200,200,350,350]);
print(gcf,[root,figureName],fileType,resolution);

%%
figureName = 'GMDH综合预测结果';
figure('Name',figureName);
PlotResults(tValue, pValue, 'Test Data');
set(gcf,'Position',[200,200,800,600]);
print(gcf,[root,figureName],fileType,resolution);

%% Save variables
% Environments
fullpath = mfilename('fullpath');
[path,name] = fileparts(fullpath);
save(['.\myStatus\',name,'_',datestr(datetime,'yyyy-mm-dd_HH-MM-SS')]);

