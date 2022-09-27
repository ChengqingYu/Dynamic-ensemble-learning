%% BPNN基预测器

%% 清空环境变量
clc;clear;

%% 训练测试数据
load dataset.mat

trainXn=dataset.trainXn;
trainYn=dataset.trainYn;
testXn=dataset.validXn;
testY=dataset.validY;

outputps=dataset.outputps;

%% BP网络构建
net=newff(trainXn,trainYn,[10]);
net.trainParam.epochs          =   500;          % 迭代次数
net.trainParam.lr              =   0.1;          % 学习率
net.trainParam.goal            =   1E-5;         % 目标
net.trainParam.showWindow      =   0;

%% BP网络训练
net=train(net,trainXn,trainYn);

%% BP网络预测
testYn_out=sim(net,testXn);
BPNNoutput=mapminmax('reverse',testYn_out,outputps);

%% 误差分析
tValue=testY;
pValue=BPNNoutput;
myerror = myError(tValue,pValue,'Show');

%% 输出预测结果
xlsName = '.\myResults\results_on_valid.xlsx';
xlswrite(xlsName,tValue',1,'A');
xlswrite(xlsName,pValue',1,'B');

%% 保存模型参数
bpnn = net;
save('.\myModel\bpnn','bpnn');