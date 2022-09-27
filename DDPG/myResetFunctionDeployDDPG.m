function [InitialObservation,LoggedSignals] = myResetFunctionDeployDDPG(TestXn)

% 部署环境初始化函数
% TestXn               input   归一化验证集特征
% InitialObservation   output  初始状态_signature
% LoggedSignals        output  记录_signature

% Author: Ye Li
% Create date: 2020/10/12
% Modified date: 2020/10/19


LoggedSignals.Step  =  0;
LoggedSignals.count =  0;
LoggedSignals.totalerror = 0;
LoggedSignals.aveerror =  0;

InitialObservation  = TestXn(:,1);


end