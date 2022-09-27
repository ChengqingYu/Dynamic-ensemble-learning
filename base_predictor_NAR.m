%% NAR基预测器

%% 清空环境变量
clc;clear;

%% 训练测试数据
load dataset.mat

trainXn=dataset.trainXn;
trainYn=dataset.trainYn;
testXn=dataset.validXn;
testY=dataset.validY;

outputps=dataset.outputps;

%% NAR网络构建
net=newff(minmax(trainXn),[20,1],{'logsig','purelin'},'trainlm');
net.trainParam.lr = 0.001; 
net.trainParam.mc = 0.04; 
net.trainParam.epochs = 500; 
net.trainParam.goal = 1e-5; 
net.trainParam.max_fail=6; 
net.trainParam.showWindow = false; 
net.trainParam.showCommandLine = false; 

%% NAR网络训练
[net,tr] = train(net,trainXn,trainYn);

%% NAR网络预测
testYn_out=sim(net,testXn);
BPNNoutput=mapminmax('reverse',testYn_out,outputps);

%% 误差分析
tValue=testY;
pValue=BPNNoutput;
myerror = myError(tValue,pValue);
myDisp({'MAE','MAPE','MSE','RMSE','SDE'},[myerror.mae,myerror.mape,myerror.mse,myerror.rmse,myerror.sde]);

%% 输出预测结果
xlsName = '.\myResults\results_on_valid.xlsx';
xlswrite(xlsName,pValue',1,'D');

%% 保存模型参数
nar = net;
save('.\myModel\nar','nar');

