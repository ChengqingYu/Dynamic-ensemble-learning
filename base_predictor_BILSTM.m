%% BILSTM基预测器

%% 清空环境变量
clc;clear;close all;

%% 训练测试数据
load dataset.mat

trainXn=dataset.trainXn;
trainYn=dataset.trainYn;
testXn=dataset.validXn;
testY=dataset.validY;

outputps=dataset.outputps; 

%% BILSTM网络构建
inputSize = size(trainXn,1);
layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(100,'OutputMode','sequence')
    bilstmLayer(100,'OutputMode','last')
    fullyConnectedLayer(100)
    tanhLayer()
    fullyConnectedLayer(1)
    tanhLayer()
    regressionLayer()
    ];
options = trainingOptions('adam',...
    'ExecutionEnvironment',  'cpu',...
    'GradientThreshold',     1,...
    'MaxEpochs',             100,...
    'MiniBatchSize',         16,...
    'InitialLearnRate',      0.01, ...
    'LearnRateSchedule',     'piecewise', ...
    'LearnRateDropPeriod',   25, ...
    'LearnRateDropFactor',   0.3, ...
    'SequenceLength',        'longest',...
    'Shuffle',               'never',...
    'Plots',                 'training-progress',...
    'Verbose',               0,...
    'L2Regularization',      0.001);

%% BILSTM网络训练
net = trainNetwork(con2seq(trainXn)',trainYn',layers,options);
    
%% BILSTM网络预测  
testYn_out = predict(net,con2seq(testXn)');
testY_out = mapminmax('reverse',testYn_out',outputps);

%% 误差分析
tValue=testY;
pValue=testY_out;
disp('BILSTM');
myerror = myError(tValue,pValue,'show');

%% 输出预测结果
xlsName = '.\myResults\results_on_valid.xlsx';
xlswrite(xlsName,tValue',1,'A');
xlswrite(xlsName,pValue',1,'B');

%% 绘图
figure
plot(pValue)
hold on
plot(tValue)
title('测试样本预测结果','fontsize',14)
xlabel('样本','fontsize',14)
ylabel('输出','fontsize',14)
legend('预测值','实际值')
grid on

%% Save variables
% Environments
fullpath = mfilename('fullpath'); 
[~,name]=fileparts(fullpath);
save(['.\myStatus\',name]);
clear fullpath path name;

% Key parameters
bilstm = net;
save('.\myModel\bilstm','bilstm');

