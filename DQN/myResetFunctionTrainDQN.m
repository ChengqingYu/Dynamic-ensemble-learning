function [InitialObservation,LoggedSignals] = myResetFunctionTrainDQN(EnvConstants)

% 测试环境初始化函数
% ValidXn              input   归一化验证集特征
% InitialObservation   output  初始状态_signature
% LoggedSignals        output  记录_signature

% Author: Ye Li
% Create date: 2020/10/12
% Modified date: 2020/10/22


% Return initial environment state variables as logged signals.
LoggedSignals.Num = 1;
LoggedSignals.State = EnvConstants.Dataset.validXn(:,1);
InitialObservation = LoggedSignals.State;

end

