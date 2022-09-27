function [Observation,Reward,IsDone,LoggedSignals] = myStepFunctionDeploy(...
	Action,...
	LoggedSignals,...
	MooSolutions,...
	CompromiseSolution,...
	BasePredValue,...
    TestXn,...
    TestY)

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
% Modified date: 2020/10/22


LoggedSignals.Step         =  LoggedSignals.Step+1;

%% Reward
% DDPG选择的权重
ensembleoutput_action      =  MooSolutions(Action==1,:)*BasePredValue(:,LoggedSignals.count);
ensembleerror_action       =  abs(ensembleoutput_action-TestY(LoggedSignals.count));
% min MAE选择的权重
ensembleoutput_invariant   =  CompromiseSolution* BasePredValue(:,LoggedSignals.count);
ensembleerror_invariant    =  abs(ensembleoutput_invariant-TestY(LoggedSignals.count));
Rewardvalue                =  0.5;
if ensembleerror_action < ensembleerror_invariant
    Reward = +Rewardvalue;
else
    Reward = -Rewardvalue;
end


%% IsDone
LoggedSignals.count=LoggedSignals.count+1;
if LoggedSignals.count == length(TestY)+1       %验证集长度
    IsDone  =  1;
    LoggedSignals.count=1;
else
    IsDone  =  0;
end

%% Observation
Observation{1}(:,1)       =  TestXn(:,LoggedSignals.count);

% %绘图
% if Reward==Rewardvalue
%     plot(LoggedSignals.Step,find(Action==1),'o','MarkerSize',5,'color','r')
% else
%     plot(LoggedSignals.Step,find(Action==1),'x','MarkerSize',5,'color','b')
% end
% hold on
% pause(0.00001)

end

