function [InitialObservation,LoggedSignals] = myResetFunctionDeployDQN(EnvConstants)

% 部署环境初始化函数
% TestXn               input   归一化测试集特征
% InitialObservation   output  初始状态_signature
% LoggedSignals        output  记录_signature

% Author: Ye Li
% Create date: 2020/10/12
% Modified date: 2020/11/20


% Return initial environment state variables as logged signals.
LoggedSignals.Num = 1;
LoggedSignals.State = EnvConstants.Dataset.testXn(:,1);
InitialObservation = LoggedSignals.State;

end

