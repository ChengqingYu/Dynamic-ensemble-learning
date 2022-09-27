function [NextObs,Reward,IsDone,LoggedSignals] = myStepFunctionTrainDQN(Action,LoggedSignals,EnvConstants)
% 	Action,...
% 	LoggedSignals,...
% 	MooSolutions,...
% 	CompromiseSolution,...
% 	BasePredValue,...
%     ValidXn,...
%     ValidY)

% 测试环境步进函数
% Action               input   动作_signature
% LoggedSignals        input   记录_signature
% MooSolutions         input   MO最优解集
% CompromiseSolution   input   折中解
% BasePredValue        input   基预测器在【验证】集的预测
% ValidXn              input   归一化验证集特征
% ValidY               input   验证集输出

% Observation          output  观测_signature
% Reward               output  奖励_signature
% IsDone               output  结束标记_signature
% LoggedSignals        output  记录_signature

% Author: Ye Li
% Create date: 2020/10/12
% Modified date: 2020/11/21


ensembleoutput_action      =  EnvConstants.MooSolutions(Action==1,:) * EnvConstants.BasePredValueValid(:,LoggedSignals.Num);
ensembleerror_action       =  abs(ensembleoutput_action - EnvConstants.Dataset.validY(LoggedSignals.Num));

ensembleoutput_invariant   =  EnvConstants.CompromiseSolution * EnvConstants.BasePredValueValid(:,LoggedSignals.Num);
ensembleerror_invariant    =  abs(ensembleoutput_invariant - EnvConstants.Dataset.validY(LoggedSignals.Num));

if ensembleerror_action < ensembleerror_invariant
    Reward = +EnvConstants.RewardValue;
else
    Reward = -EnvConstants.RewardValue;
end

if LoggedSignals.Num + 1 <= EnvConstants.Dataset.validNum
    LoggedSignals.Num = LoggedSignals.Num + 1;
    IsDone = false;
else
    LoggedSignals.Num = 1;
    IsDone = true;
end

LoggedSignals.State = EnvConstants.Dataset.validXn(:,LoggedSignals.Num);
NextObs{1}(:,1) = LoggedSignals.State;

end

