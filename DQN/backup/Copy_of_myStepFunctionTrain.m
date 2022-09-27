function [Observation,Reward,IsDone,LoggedSignals] = myStepFunctionTrain(...
	Action,...
	LoggedSignals,...
	MooSolutions,...
	CompromiseSolution,...
	BasePredValue,...
    ValidXn,...
    ValidY)

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
% Modified date: 2020/10/22


LoggedSignals.Step         =  LoggedSignals.Step+1;

%% Reward
% DQN选择的权重
ensembleoutput_action      =  MooSolutions(Action==1,:) * BasePredValue(:,LoggedSignals.count);
ensembleerror_action       =  abs(ensembleoutput_action - ValidY(LoggedSignals.count));
% min MAE选择的权重
ensembleoutput_invariant   =  CompromiseSolution * BasePredValue(:,LoggedSignals.count);
ensembleerror_invariant    =  abs(ensembleoutput_invariant - ValidY(LoggedSignals.count));

Rewardvalue                =  0.5;

if ensembleerror_action < ensembleerror_invariant
    Reward = +Rewardvalue;
else
    Reward = -Rewardvalue;
end

%% IsDone
LoggedSignals.count=LoggedSignals.count+1;
if LoggedSignals.count == length(ValidY)+1 %验证集长度
    IsDone  =  1;
    LoggedSignals.count=1;
else
    IsDone  =  0;
end

%% Observation
Observation{1}(:,1)        =  ValidXn(:,LoggedSignals.count);

%绘图
% if Reward==Rewardvalue
%     plot(LoggedSignals.Step,find(Action==1),'o','MarkerSize',5,'color','r')
% else
%     plot(LoggedSignals.Step,find(Action==1),'x','MarkerSize',5,'color','b')
% end
% hold on
% pause(0.00001)

end

