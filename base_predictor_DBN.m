%% DBN基预测器

%% 清空环境变量
clc;clear;close all;

%% 训练测试数据
load dataset

trainXn=dataset.trainXn';
trainYn=dataset.trainYn';
testXn=dataset.validXn';
testYn=dataset.validYn';
testY=dataset.validY;

outputps=dataset.outputps;

%% DBN网络构建
dbn.sizes                =  [100]; 
opts.numepochs           =  100;
opts.batchsize           =  1;
opts.momentum            =  0;
opts.alpha               =  0;
opts.plot                =  0;
dbn                      =  dbnsetup(dbn, trainXn, opts);
dbn                      =  dbntrain(dbn, trainXn, opts);
net                      =  dbnunfoldtonn(dbn, 1);
net.activation_function  =  'sigm';                 %'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
net.output               =  'linear';               %'sigm' (=logistic), 'softmax' and 'linear'
net.learningRate         =  0.001;

%% DBN网络训练
opts.numepochs           =  100;
opts.batchsize           =  1;
net = nntrain(net, trainXn, trainYn, opts);

%% DBN网络预测
[testErr,testYn_out] = nntest(net, testXn, testYn);
testOutput = mapminmax('reverse',testYn_out',outputps);

%% 误差分析
tValue=testY;
pValue=testOutput;
disp('DBN');
myerror = myError(tValue,pValue,'show_less');

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

%% 变量保存
% Environments
fullpath = mfilename('fullpath'); 
[~,name]=fileparts(fullpath);
save(['.\myStatus\',name]);
clear fullpath path name;

% Key parameters
dbn = net;
save('.\myModel\dbn','dbn');

