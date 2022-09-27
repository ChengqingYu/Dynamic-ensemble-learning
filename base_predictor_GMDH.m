%% GMDH��Ԥ����

%% ��ջ�������
clc;clear;close all;

%% ѵ����������
load dataset.mat
trainXn=dataset.trainXn;
trainYn=dataset.trainYn;
testXn=dataset.testXn;
testY=dataset.testY;
outputps=dataset.outputps; 

%% GMDH���繹��
params.MaxLayerNeurons = 25;   % Maximum Number of Neurons in a Layer
params.MaxLayers = 5;          % Maximum Number of Layers
params.alpha = 0;              % Selection Pressure
params.pTrain = 0.8;           % Train Ratio

%% GMDH����ѵ��
net = GMDH(params, trainXn, trainYn);

%% GMDH����Ԥ��
testYn_out = ApplyGMDH(net, testXn);
testOutput = mapminmax('reverse',testYn_out,outputps);

%% ������
tValue=testY;
pValue=testOutput;
disp('GMDH');
myerror = myError(tValue,pValue,'show');

%% ���Ԥ����
xlsName = '.\myResults\results.xlsx';
xlswrite(xlsName,tValue',1,'A');
xlswrite(xlsName,pValue',1,'C');

%% ��ͼ
close all;
fileName = '';
root = ['C:\Users\Flywi\Desktop\��ͨ��Ԥ��\myFigure\',fileName,'\'];
fileType = '-djpeg';
resolution = '-r600';
xTest = 1:dataset.testNum;

% Plot predicted wind speed on testing set
figureName = 'GMDHԤ����';
figure('Name',figureName);
plot(xTest,tValue,xTest,pValue, ...
    'LineWidth',     1);
xlabel('ʱ�� (Сʱ)', ...
    'FontSize',      12, ...
    'FontWeight',    'bold');
ylabel('��ͨ�� ���� / ʱ��', ...
    'FontSize',      12, ...
    'FontWeight',    'bold');
legend('�۲�ֵ','Ԥ��ֵ', ...
    'FontSize',      11, ...
    'FontWeight',    'normal');
grid on;
set(gcf,'Position',[200,200,700,350]);
print(gcf,[root,figureName],fileType,resolution);

% Plot scatter plot
figureName = 'GMDHԤ��ɢ��ͼ';
figure('Name',figureName);
plot(tValue,pValue(),'.', ...
    'MarkerSize',    10);
hold on;
plot(min(tValue):0.001:max(tValue),min(tValue):0.001:max(tValue), ...
    'LineWidth',     1);
axis([min(tValue),max(tValue),min(tValue),max(tValue)]);
axis square;
xlabel('�۲⽻ͨ�� ����/ʱ��', ...
    'FontSize',     12, ...
    'FontWeight',   'bold');
ylabel('Ԥ�⽻ͨ�� ����/ʱ��)', ...
    'FontSize',     12, ...
    'FontWeight',   'bold');
text(min(tValue)+500,max(tValue)-500,['PCC = ',num2str(myerror.pcc)], ...
    'FontSize',     12, ....
    'FontWeight',   'normal');
set(gcf,'Position',[200,200,350,350]);
print(gcf,[root,figureName],fileType,resolution);

%%
figureName = 'GMDH�ۺ�Ԥ����';
figure('Name',figureName);
PlotResults(tValue, pValue, 'Test Data');
set(gcf,'Position',[200,200,800,600]);
print(gcf,[root,figureName],fileType,resolution);

%% Save variables
% Environments
fullpath = mfilename('fullpath');
[path,name] = fileparts(fullpath);
save(['.\myStatus\',name,'_',datestr(datetime,'yyyy-mm-dd_HH-MM-SS')]);

