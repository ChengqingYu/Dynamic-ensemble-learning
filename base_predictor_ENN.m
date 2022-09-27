%% ENN基预测器

%% 清空环境变量
clc;clear;

%% 训练测试数据
load dataset.mat

trainXn=dataset.trainXn;
trainYn=dataset.trainYn;
testXn=dataset.validXn;
testY=dataset.validY;

outputps=dataset.outputps;

%% ENN网络构建
net = elmannet(1:2,10,'traingdx');
net.trainParam.show       =  1;         %设置显示级别
net.trainParam.epochs     =  100;      %设置最大迭代次数
net.trainParam.goal       =  1e-5;   %误差容限，阈值误差
net.trainParam.max_fail   =  6;         %最多验证失败次数
net.trainParam.showWindow = false; 
net.trainParam.showCommandLine = false; 
net = init(net);                        %对网络初始化

%% ENN网络训练
net = train(net,trainXn,trainYn);

%% ENN网络预测
testYn_out=sim(net,testXn);
ENNoutput=mapminmax('reverse',testYn_out,outputps);

%% 误差分析
tValue=testY;
pValue=ENNoutput;
disp('ENN');
myerror = myError(tValue,pValue,'show');

%% 输出预测结果
xlsName = '.\myResults\results_on_valid.xlsx';
xlswrite(xlsName,tValue',1,'A');
xlswrite(xlsName,pValue',1,'C');

%% 保存模型参数
enn = net;
save('.\myModel\enn','enn');