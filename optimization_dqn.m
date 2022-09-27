%% DQN优化集成权重系数(动态)

%% Clear All
clear;clc;close all;


%% Load Dataset
% Data for getting pareto optimal solutions
results_on_valid = xlsread('results_on_valid.xlsx');
save('.\myResults\results_on_valid','results_on_valid');
pValue = results_on_valid(:,2:4)';
tValue = results_on_valid(:,1)';

% Data for train and deploy agent
load dataset
validXn = dataset.validXn;
validYn = dataset.validYn;
validY = dataset.validY;
testXn = dataset.testXn;
testY = dataset.testY;
outputps = dataset.outputps;

%%
load dbn;load bilstm;load lstm;load gru;
pValue1 = myPredict('dbn',dbn,testXn,outputps);
pValue2 = myPredict('lstm',lstm,testXn,outputps);
pValue3 = myPredict('gru',gru,testXn,outputps);
pValueTest = [pValue1;pValue2;pValue3];


%% NSGA-II
load mooResults;
load population_size;


%% Compromise
load nsga2_weights;
compromiseSolution = nsga2_weights;


%% Pareto Front
load compromiseParetoFront;
% figure('Name','Pareto front of NSGA-II');
% plot(mooResults.paretoFront(:,1),mooResults.paretoFront(:,2),'o');
% xlabel('MSE');
% ylabel('SDE');
% hold on;
% plot(compromiseParetoFront(1),compromiseParetoFront(2),'r*','MarkerSize',30);
% legend('Pareto front','Selected solution');


%% DQN
%% Environments
num_obs                      =  dataset.inputNum;             %状态数
num_action                   =  population_size;              %动作数

ObservationInfo              =  rlNumericSpec([num_obs 1]);   %创建连续状态空间所需的INFO（维度）
ObservationInfo.Name         =  'Historical values';          %INFO名称
ObservationInfo.Description  =  'x1, x2, x3, x4, x5';         %INFO描述

actions                      =  eye(num_action);
ActionInfo                   =  rlFiniteSetSpec(num2cell(actions,2));   %创建连续动作空间所需的INFO（元素）
ActionInfo.Name              =  'Combination weights';           %INFO名称
ActionInfo.Description       =  'w1, w2, w3, ..., w50';          %INFO描述
% Parameters
envConstants.Dataset            = dataset;
envConstants.MooSolutions       = mooResults.solutions;
envConstants.CompromiseSolution = compromiseSolution;
envConstants.BasePredValueValid = pValue;
envConstants.BasePredValueTest  = pValueTest;
envConstants.RewardValue        = 1.0;

% Training Environment
ResetHandleTrain             =  @() myResetFunctionTrainDQN(envConstants);
StepHandleTrain              =  @(Action,LoggedSignals) myStepFunctionTrainDQN(Action,LoggedSignals,envConstants);
envTrain = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandleTrain,ResetHandleTrain);

% Deployment Environment
ResetHandleDeploy            =  @() myResetFunctionDeployDQN(envConstants);
StepHandleDeploy             =  @(Action,LoggedSignals) myStepFunctionDeployDQN(Action,LoggedSignals,envConstants);
envDeploy = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandleDeploy,ResetHandleDeploy);


%% Policy and Representation
%% Critic Network
statePath = [
    imageInputLayer([num_obs 1 1], 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(48, 'Name', 'CriticStateFC1')
    reluLayer("Name","state_relu1")
    fullyConnectedLayer(48, 'Name', 'CriticStateFC2')
    reluLayer("Name","state_relu2")
    fullyConnectedLayer(48,"Name","CriticStateFC3")
    ];
actionPath = [
    imageInputLayer([1 num_action 1], 'Normalization', 'none', 'Name', 'action')
    fullyConnectedLayer(48, 'Name', 'CriticActionFC1')
    ];
commonPath = [
    additionLayer(2,'Name', 'add')
    reluLayer('Name','CriticCommonRelu1')
    fullyConnectedLayer(1, 'Name', 'output')
    ];
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC3','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
% Configuration options
criticOpts = rlRepresentationOptions(...
    'LearnRate',           1e-3,...
    'GradientThreshold',   1,...
    'UseDevice',           "cpu");
% Representation
critic = rlQValueRepresentation(...
    criticNetwork,...
    ObservationInfo,...
    ActionInfo,...
    'Observation',{'state'},...
    'Action',{'action'},...
    criticOpts);


%% Actor Network
% No actor for DQN


%% Agent
% Configuration options
agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN',            false, ...
    'ExperienceBufferLength',  100000, ...
    'TargetUpdateMethod',      'smoothing',...
    'TargetSmoothFactor',      1e-3, ...
    'DiscountFactor',          0.99, ...
    'MiniBatchSize',           64, ...
    'NumStepsToLookAhead',     1);
agentOpts.EpsilonGreedyExploration.Epsilon      = 0.999;
agentOpts.EpsilonGreedyExploration.EpsilonDecay = 0.0001;
agentOpts.EpsilonGreedyExploration.EpsilonMin   = 0.1;
% DQN agent
agent = rlDQNAgent(critic,agentOpts);
 
%% Train Agent
% Configuration options
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',                 2000, ...
    'MaxStepsPerEpisode',          dataset.validNum, ...
    'ScoreAveragingWindowLength',  20, ...
    'StopTrainingCriteria',        "AverageReward", ...
    'StopTrainingValue',           1000, ...
    'Verbose',                     true, ....    %显示训练过程 command window
    'Plots',                       "none", ...   %显示训练过程 episode manager none
    'UseParallel',                 true);
trainingInfo = train(agent,envTrain,trainOpts); %在envTrain中依据trainOpts训练agent agent中已包含了critic和actor两部分


%% Simulate DQN Agent
% Simulation On envTrain
simOptsTrain       =  rlSimulationOptions('MaxSteps',dataset.validNum);
experience_train   =  sim(envTrain,agent,simOptsTrain);
[action_train,~]   =  find(squeeze(experience_train.Action.CombinationWeights.Data)==1);

% Simulation On envDeploy
simOptsDeploy      =  rlSimulationOptions('MaxSteps',dataset.testNum);
experience_deploy  =  sim(envDeploy,agent,simOptsDeploy);
[action_deploy,~]  =  find(squeeze(experience_deploy.Action.CombinationWeights.Data)==1);


%% Get dynamic weights
nsga2_dqn_weights = mooResults.solutions(action_deploy,:)';


%% Figure
fileName = 'DQN';
root = ['C:\Users\liuhui116\Desktop\铁路风工程\图片\',fileName,'\'];
fileType = '-djpeg';
resolution = '-r600';

% Plot reward train
figureName='Episode reward of DQN during training';
figure('Name',figureName);
plot(trainingInfo.EpisodeReward, ...
    'LineWidth',     1);
hold on;
plot(trainingInfo.AverageReward, ...
    'LineWidth',     1);
legend('Episode reward','Average reward', ...
    'Location',      'SouthEast', ...
    'Orientation',   'vertical', ...
    'FontSize',      11, ...
	'FontName',      'Times New Roman');
xlabel('Episode', ...
    'FontSize',      12, ...
	'FontName',      'Times New Roman', ...
    'FontWeight',    'bold');
ylabel('Reward', ...
    'FontSize',      12, ...
	'FontName',      'Times New Roman', ...
    'FontWeight',    'bold');
set(gcf,'Position',[200,200,400,300]);
print(gcf,[root,figureName],fileType,resolution);

% Plot selection results
load compromiseIndex;
figureName='Selection results of the Pareto optimal solutions in the testing set';
figure('Name',figureName);
plot(action_deploy,'o');
hold on;
line([1,dataset.testNum],[compromiseIndex,compromiseIndex], ...
    'Color','red', ...
    'LineWidth',     1);
legend('Dynamic','Static','Location','bestoutside','Orientation','vertical')
xlabel('Times (3-min)', ...
    'FontSize',      12, ...
	'FontName',      'Times New Roman', ...
    'FontWeight',    'bold');
ylabel('Selected solution', ...
    'FontSize',      12, ...
	'FontName',      'Times New Roman', ...
    'FontWeight',    'bold');
set(gcf,'Position',[200,200,500,300]);
print(gcf,[root,figureName],fileType,resolution);

% Plot reward during sim in envTrain
figureName='Reward for each step of the reinforcement learning agent in the training environment';
figure('Name',figureName);
plot(experience_train.Reward.Data, ...
    'LineWidth',     1);
xlabel('Times (3-min)', ...
    'FontSize',      12, ...
	'FontName',      'Times New Roman', ...
    'FontWeight',    'bold');
ylabel('Instant reward', ...
    'FontSize',      12, ...
	'FontName',      'Times New Roman', ...
    'FontWeight',    'bold');
% yticks([-envConstants.RewardValue, +envConstants.RewardValue]);
% yticklabels({'Punishment','Reward'});
set(gcf,'Position',[200,200,400,300]);
print(gcf,[root,figureName],fileType,resolution);

% Plot reward during sim in envDeploy
figureName='Reward for each step of the reinforcement learning agent in the deployment environment';
figure('Name',figureName);
plot(experience_deploy.Reward.Data, ...
    'LineWidth',     1);
xlabel('Times (3-min)', ...
    'FontSize',      12, ...
	'FontName',      'Times New Roman', ...
    'FontWeight',    'bold');
ylabel('Instant reward', ...
    'FontSize',      12, ...
	'FontName',      'Times New Roman', ...
    'FontWeight',    'bold');
set(gcf,'Position',[200,200,400,300]);
print(gcf,[root,figureName],fileType,resolution);


%% Save variables
% Environments
fullpath = mfilename('fullpath');
[path,name]=fileparts(fullpath);
save(['.\myStatus\',name,'_',datestr(datetime,'yyyy-mm-dd_HH-MM-SS')]);
% Key parameters
save('.\myModel\nsga2_dqn_weights','nsga2_dqn_weights');

