function [NextObs,Reward,IsDone,LoggedSignals] = myStepFunctionDeployDQN(Action,LoggedSignals,EnvConstants)
% 	Action,...
% 	LoggedSignals,...
% 	MooSolutions,...
% 	CompromiseSolution,...
% 	BasePredValue,...
%     TestXn,...
%     TestY)

% 部署环境步进函数
% Action               input   动作_signature
% LoggedSignals        input   记录_signature
% MooSolutions         input   MO最优解集
% CompromiseSolution   input   折中解
% BasePredValue        input   基预测器在【测试】集的预测
% testXn               input   归一化测试集特征
% TestY                input   测试集输出

% Observation          output  观测_signature
% Reward               output  奖励_signature
% IsDone               output  结束标记_signature
% LoggedSignals        output  记录_signature

% Author: Ye Li
% Create date: 2020/10/12
% Modified date: 2020/11/21


ensembleoutput_action      =  EnvConstants.MooSolutions(Action==1,:) * EnvConstants.BasePredValueTest(:,LoggedSignals.Num);
ensembleerror_action       =  abs(ensembleoutput_action - EnvConstants.Dataset.testY(LoggedSignals.Num));
%移动至外部计算
ensembleoutput_invariant   =  EnvConstants.CompromiseSolution * EnvConstants.BasePredValueTest(:,LoggedSignals.Num);
ensembleerror_invariant    =  abs(ensembleoutput_invariant - EnvConstants.Dataset.testY(LoggedSignals.Num));

if ensembleerror_action < ensembleerror_invariant
    Reward = +EnvConstants.RewardValue;
else
    Reward = -EnvConstants.RewardValue;
end

if LoggedSignals.Num + 1 <= EnvConstants.Dataset.testNum
    LoggedSignals.Num = LoggedSignals.Num + 1;
    IsDone = false;
else
    LoggedSignals.Num = 1;
    IsDone = true;
end

LoggedSignals.State = EnvConstants.Dataset.testXn(:,LoggedSignals.Num);
NextObs{1}(:,1) = LoggedSignals.State;

end

