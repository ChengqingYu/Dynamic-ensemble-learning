%% GRNN基预测器

%% 清空环境变量
clc;clear;

%% 训练测试数据
load dataset.mat

trainXn=dataset.trainXn;
trainYn=dataset.trainYn;
testXn=dataset.validXn;
testY=dataset.validY;

outputps=dataset.outputps;

%% GRNN网络构建 训练
net=newgrnn(trainXn,trainYn,0.3);

%% GRNN网络预测
testYn_out=sim(net,testXn);
testY_out=mapminmax('reverse',testYn_out,outputps);

%% 误差分析
tValue=testY;
pValue=testY_out;
x = 1:length(tValue);
myerror = myError(tValue,pValue);
myDisp({'MAE','MAPE','MSE','RMSE','SDE'},[myerror.mae,myerror.mape,myerror.mse,myerror.rmse,myerror.sde]);

%% 输出预测结果
xlsName = '.\myResults\results_on_valid.xlsx';
xlswrite(xlsName,pValue',1,'D');

%% 保存模型参数
grnn = net;
save('.\myModel\grnn','grnn');