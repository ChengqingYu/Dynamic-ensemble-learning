%% DQN优化集成权重系数(动态)

%% Clear All
clear;clc;close all;


%% Load Dataset
results_on_valid=xlsread('results_on_valid.xlsx');
save('.\myResults\results_on_valid','results_on_valid');
pValue = results_on_valid(:,2:4)';
tValue = results_on_valid(:,1)';

load dataset.mat
validXn=dataset.validXn;
validYn=dataset.validYn;
testXn=dataset.testXn;
testY=dataset.testY;


%% NSGA-II
population_size              =  50;
total_number_of_generations  =  200;
mooResults=nsga_2(population_size,total_number_of_generations);


%% Compromise
pfError=zeros(1,mooResults.number_of_archive);
for i = 1:mooResults.number_of_archive
    w           =  mooResults.solutions(i,:);
    pValueSum   =  w * pValue;
    allError    =  myError(tValue,pValueSum);
    pfError(i)  =  allError.mae;
end
[~,index]=min(pfError);
compromiseSolution = mooResults.solutions(index,:);

%% Pareto Front
figure('Name','Pareto front of NSGA-II');
plot(mooResults.paretoFront(:,1),mooResults.paretoFront(:,2),'o');
xlabel('MSE');
ylabel('SDE');
hold on;
plot(mooResults.paretoFront(index,1),mooResults.paretoFront(index,2),'r*','MarkerSize',30);
legend('Pareto front','Selected solution');


%% DDPG
%% Environments
% Parameters
num_obs                      =  5;                            %状态数
num_action                   =  1;                            %动作数
% Observation
ObservationInfo              =  rlNumericSpec([num_obs 1]);   %创建连续状态空间所需的INFO（维度）
ObservationInfo.Name         =  'Historical values';          %INFO名称
ObservationInfo.Description  =  'x1, x2, x3, x4, x5';         %INFO描述
ObservationInfo.LowerLimit   =  -1;
ObservationInfo.UpperLimit   =  +1;
% Action
ActionInfo                   =  rlNumericSpec([1 1]);         %创建连续动作空间所需的INFO（元素）
ActionInfo.Name              =  'Combination weights';        %INFO名称
ActionInfo.Description       =  'w1, w2, w3, ..., w50';       %INFO描述
ActionInfo.LowerLimit        =  0;
ActionInfo.UpperLimit        =  1;


%% Training Environments
ResetHandleTrain             =  @() myResetFunctionTrainDDPG(validXn);                      %重置函数 0输入2输出
StepHandleTrain              =  @(Action,LoggedSignals) myStepFunctionTrainDDPG(...         %步进函数 2输入4输出
    Action,...
    LoggedSignals,...
    mooResults.solutions,...
    compromiseSolution,...
    pValue,...
    validXn,...
    tValue);
envTrain = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandleTrain,ResetHandleTrain);  %自定义RL环境


%% Deployment Environments
ResetHandleDeploy            =  @() myResetFunctionDeployDDPG(testXn);
StepHandleDeploy             =  @(Action,LoggedSignals) myStepFunctionDeployDDPG(...
    Action,...
    LoggedSignals,...
    mooResults.solutions,...
    compromiseSolution,...
    pValue,...
    testXn,...
    testY);
envDeploy = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandleDeploy,ResetHandleDeploy);


%% Policy and Representation
%% Critic Network
statePath = [                                                       %状态子网
    imageInputLayer([num_obs 1 1], 'Normalization', 'none', 'Name', 'state')                     %输入层（num_obs输入）
    fullyConnectedLayer(48,'Name','CriticObsFC2','BiasLearnRateFactor', 0, 'Bias', zeros(48,1))    %全连接层（无激活函数）
    tanhLayer('Name','tanh3')                                                                    %tanh层（激活函数）
    ];
actionPath = [                                                      %动作子网
    imageInputLayer([num_action 1 1], 'Normalization', 'none', 'Name', 'action')                  %输入层（num_model输入） 在哪里定义呀？
    fullyConnectedLayer(48,'Name','CriticActFC1','BiasLearnRateFactor', 0, 'Bias', zeros(48,1))    %全连接层（无激活函数）
    tanhLayer('Name','tanh2')                                                                    %tanh层（激活函数）
    ];
commonPath = [                                                      %公共子网
    concatenationLayer(1,2,'Name','concat')                                                      %连接层
    fullyConnectedLayer(20,'Name','StateValue1')                                                 %全连接层
    reluLayer('Name','Relu1')                                                                    %relu层
    fullyConnectedLayer(1,'Name','StateValue')                                                   %输出层 全连接层（输出1个值 Q值）
    ];
criticNetwork = layerGraph(statePath);                                                               %状态子网 初始化网络
criticNetwork = addLayers(criticNetwork, actionPath);                                                %动作子网 添加至网络
criticNetwork = addLayers(criticNetwork, commonPath);                                                %公共子网 添加至网络
criticNetwork = connectLayers(criticNetwork,'tanh3','concat/in1');                                   %连接状态子网输出 至 连接层in1
criticNetwork = connectLayers(criticNetwork,'tanh2','concat/in2');                                   %连接状态子网输出 至 连接层in2
% Configuration options
criticOptions = rlRepresentationOptions(...                   %相当于网络训练参数
    'Optimizer',           'adam',...
    'LearnRate',           1e-5,...
    'GradientThreshold',   1,...
    'UseDevice',           "cpu");
% Representation
critic = rlQValueRepresentation(...                           %依据net创建critic representation
    criticNetwork,...
    ObservationInfo,...
    ActionInfo,...
    'Observation',{'state'},...    %net状态输入层名称
    'Action',{'action'},...        %net动作输入层名称for critic
    criticOptions);


%% Actor Network
% actorNetwork = [
%     imageInputLayer([num_obs 1 1], 'Normalization', 'none', 'Name', 'state')         %输入层（num_obs输入）
%     fullyConnectedLayer(100,'Name','ActionFC1')                                       %全连接层（20）
%     reluLayer('Name','ActionRelu1')                                                  %relu层
%     fullyConnectedLayer(100,'Name','ActionFC2')                                       %全连接层（20）
%     reluLayer('Name','ActionRelu2')                                                  %relu层
%     fullyConnectedLayer(num_action, 'Name', 'action')                                %全连接层
%     tanhLayer('Name','ActorTanh')                                                    %tanh层
%     scalingLayer('Name','ActorScaling','Scale',0.5,'Bias',0.5)];                     %缩放偏执层（tanh->action） [-1 1]->[0 1]
actorNetwork = [
    imageInputLayer([num_obs 1 1], 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(48,'Name','ActionFC1')
    reluLayer('Name','ActionRelu1')
    fullyConnectedLayer(48,'Name','ActionFC2')
    reluLayer('Name','ActionRelu2')
    fullyConnectedLayer(1, 'Name', 'action')
    tanhLayer('Name','ActorTanh')
    scalingLayer('Name','ActorScaling','Scale',max(ActionInfo.UpperLimit))];
% Configuration options
actorOptions = rlRepresentationOptions(...                    %相当于网络训练参数
    'Optimizer',           'adam',...
    'LearnRate',           1e-6,...
    'GradientThreshold',   1,...
    'UseDevice',           "cpu");
% Representation
actor = rlDeterministicActorRepresentation(...                               %依据net创建actor representation【TODO】
    actorNetwork,...
    ObservationInfo,...
    ActionInfo,...
    'Observation',{'state'},...    %net状态输入层名称
    'Action',{'ActorScaling'},...  %net动作输出层名称for actor
    actorOptions);


%% Agent
% Configuration options
agentOptions = rlDDPGAgentOptions(...
    'DiscountFactor',     0.9, ...                         %折扣系数
    'TargetSmoothFactor', 1e-3, ...                        %DDPG软更新参数t
    'MiniBatchSize',      16);                             %minibatch数量
% agentOpts.NoiseOptions.Variance          = 0.003;
% agentOpts.NoiseOptions.VarianceDecayRate = 1e-5;
% agentOpts.NoiseOptions.SampleTime        = 1;
% DDPG agent
agent = rlDDPGAgent(actor,critic,agentOptions);               %DDPG agent


%% Train Agent
% Configuration options
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',                 500,...
    'MaxStepsPerEpisode',          dataset.validNum,...
    'ScoreAveragingWindowLength',  5,...
    'StopTrainingCriteria',        "AverageReward",...
    'StopTrainingValue',           1000,...
    'Verbose',                     true,....  %显示训练过程 command window
    'Plots',                       'none');   %显示训练过程 episode manager
trainingInfo = train(agent,envTrain,trainOpts); %在envTrain中依据trainOpts训练agent agent中已包含了critic和actor两部分
figure
plot(trainingInfo.EpisodeReward)   %各回合奖励
hold on
plot(trainingInfo.AverageReward)   %各回合最后一个window内的平均奖励


%% Simulate DDPG Agent
%% Simulation On envTrain
simOpts           =  rlSimulationOptions('MaxSteps',dataset.validNum);
experience_train  =  sim(envTrain,agent,simOpts);  %在环境env1中依据simOpts模拟训练好的agent 得到sim的经历
% [variantind1 n1]  =  find(squeeze(experience_train{step_count}.Action.CombinationWeights.Data(1,:,:))==1);

%% Simulation On envDeploy
simOpts           =  rlSimulationOptions('MaxSteps',dataset.testNum);
experience_deploy =  sim(envDeploy,agent,simOpts);
% [variantind2 n2]  =  find(squeeze(experience_test{step_count}.Action.CombinationWeights.Data(1,:,:))==1);

%%【TODO】 依据experience中的动作选择权重

%% 保存环境
%环境
fullpath = mfilename('fullpath'); 
[path,name]=fileparts(fullpath);
save(['.\myStatus\',name]);
%重要参数