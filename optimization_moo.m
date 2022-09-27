%% NSGA-II优化集成权重系数

%% Clear All
clear;clc;close all;


%% File details
fileName = 'NSGA-II';
root = ['C:\Users\liuhui116\Desktop\铁路风工程\图片\',fileName,'\'];
fileType = '-djpeg';
resolution = '-r600';


%% Load Dataset
results_on_valid=xlsread('results_on_valid.xlsx');
save('.\myResults\results_on_valid','results_on_valid');
pValue = results_on_valid(:,2:4)';
tValue = results_on_valid(:,1)';


%% NSGA-II
population_size              =  50;
total_number_of_generations  =  200;
mooResults=nsga_2(population_size,total_number_of_generations);


%% Pareto Front
% Compromise
compromise_method = 'topsis';
pfError=zeros(4,mooResults.number_of_archive);
for i=1:mooResults.number_of_archive
    w                 =  mooResults.solutions(i,:);
    pValueSum         =  w * pValue;
    allError          =  myError(tValue,pValueSum);
    pfError(1,i)      =  allError.mae;
    pfError(2,i)      =  allError.mape;
    pfError(3,i)      =  allError.rmse;
    pfError(4,i)      =  allError.sde;
end
switch(lower(compromise_method))
    case 'min_mae'
        [~,index]             =  min(pfError(1,:));
        compromiseName        =  'Minimize MAE';
    case 'min_mape'
        [~,index]             =  min(pfError(2,:));
        compromiseName        =  'Minimize MAPE';
    case 'min_rmse'
        [~,index]             =  min(pfError(3,:));
        compromiseName        =  'Minimize RMSE';
    case 'min_sde'
        [~,index]             =  min(pfError(4,:));
        compromiseName        =  'Minimize SDE';
    case 'topsis'
        decisionMakingMatrix  =  mooResults.paretoFront;
        lambdaWeight          =  ones(1,mooResults.number_of_objectives) * +1;
        criteriaSign          =  ones(1,mooResults.number_of_objectives) * -1;
        topsisResults         =  topsis(decisionMakingMatrix,lambdaWeight,criteriaSign);
        index                 =  topsisResults.rankFirst;
        compromiseName        =  'TOPSIS';
    otherwise
        disp('指定方法错误');
        return;
end
disp(['折中解选择方法：',compromiseName]);
compromiseIndex = index;
compromiseSolution = mooResults.solutions(index,:);
compromiseParetoFront = mooResults.paretoFront(index,:);


figureName = 'Pareto front of NSGA-II';
figure('Name',figureName);
plot(mooResults.paretoFront(:,1),mooResults.paretoFront(:,2),'o', ...
    'LineWidth',1);
hold on;
plot(compromiseParetoFront(1),compromiseParetoFront(2),'r*', ...
    'MarkerSize',30, ...
    'LineWidth',1);
legend('Pareto front','Selected solution',...
    'Orientation',   'vertical',...
    'FontSize',      11,...
	'FontName',      'Times New Roman');
xlabel('MSE',...
    'FontSize',      12,...
	'FontName',      'Times New Roman',...
    'FontWeight',    'bold');
ylabel('SDE',...
    'FontSize',      12,...
	'FontName',      'Times New Roman',...
    'FontWeight',    'bold');
set(gcf,'Position',[200,200,400,350]);
print(gcf,[root,figureName],fileType,resolution);

% Plot loss curves
figureName = 'Convergence of the average objective function values of each generation during 100 iterations';
figure('Name',figureName);
myLabelY = {'MSE','Variance'};
for i = 1:2
    subplot(1,2,i);
    plot(mooResults.curve(i,1:100),...
        'LineWidth',1);
    xlabel('Iteration',...
        'FontSize',      12,...
        'FontName',      'Times New Roman',...
        'FontWeight',    'bold');
    ylabel(myLabelY{i},...
        'FontSize',      12,...
        'FontName',      'Times New Roman',...
        'FontWeight',    'bold');
end
set(gcf,'Position',[200,200,500,200]);
print(gcf,[root,figureName],fileType,resolution);


%% Optimal Result
% Base models
disp('各预测器 验证集');
myerror1 = myError(tValue,pValue(1,:),'show');
myerror2 = myError(tValue,pValue(2,:),'show');
myerror3 = myError(tValue,pValue(3,:),'show');

% Static ensemble
pValueSum = compromiseSolution * pValue;
disp('静态多目标集成 验证集');
myerror = myError(tValue,pValueSum,'show');


%% Save variables
% Environments
fullpath = mfilename('fullpath'); 
[~,name] = fileparts(fullpath);
save(['.\myStatus\',name,'_',datestr(datetime,'yyyy-mm-dd_HH-MM-SS')]);

% Key parameters
nsga2_weights = compromiseSolution;
save('.\myModel\mooResults','mooResults');
save('.\myModel\nsga2_weights','nsga2_weights');
save('.\myModel\compromiseIndex','compromiseIndex');
save('.\myModel\compromiseParetoFront','compromiseParetoFront');
save('.\myModel\population_size','population_size');