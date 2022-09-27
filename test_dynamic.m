%% 集成预测器预测(动态)

%% 清空环境变量
clc;clear;close all;


%% 测试数据提取
load dataset
testY=dataset.testY;
testXn=dataset.testXn;
outputps=dataset.outputps;


%% 读取模型
%预测器和权重向量
load bpnn;load enn;load nar;load dbn;load lstm;load gru;
load nsga2_weights;load nsga2_dqn_weights;


%% 测试
optimization_method  = 'nsga2_dqn';
base_model_type      = 'dl';

switch(lower(optimization_method))
    case 'nsga2_dqn'
        w = nsga2_dqn_weights;       %[3*200]矩阵
        optimalName = 'NSGA2-DQN';
    case 'nsga2_ddpg'
        w = nsga2_ddpg_weights;      %[3*200]矩阵
        optimalName = 'NSGA2-DDPG';
    otherwise
        disp('指定方法错误');
end

switch(lower(base_model_type))
    case 'ml'
        pValue1=myPredict('bpnn',bpnn,testXn,outputps);
        pValue2=myPredict('enn',enn,testXn,outputps);
        pValue3=myPredict('nar',nar,testXn,outputps);
        modelName={'BPNN','ENN','NAR','Dynamic ensemble'};
    case 'dl'
        pValue1=myPredict('dbn',dbn,testXn,outputps);
        pValue2=myPredict('lstm',lstm,testXn,outputps);
        pValue3=myPredict('gru',gru,testXn,outputps);
        modelName={'DBN','LSTM','GRU','Dynamic Ensemble'};
    otherwise
        disp('指定方法错误');
        return;
end

disp(['预测模型：',modelName{1},' ',modelName{2},' ',modelName{3}]);
disp(['优化算法：',optimalName]);

%集成
pValue       =  [pValue1; pValue2; pValue3];
pValueSum    =  sum(w .* pValue);


%% 误差分析
tValue=testY;
%基预测器
disp('基本预测器：');
myError1= myError(tValue,pValue1,'show');
myError2 = myError(tValue,pValue2,'show');
myError3 = myError(tValue,pValue3,'show');
%集成
disp('NSGA2集成预测器：');
myError4 = myError(tValue,nsga2_weights * pValue,'show');
disp('NSGA2-DQN集成预测器：');
myErrorSum = myError(tValue,pValueSum,'show');


%% 输出绘图
fileName = 'NSGA-II';
root = ['C:\Users\liuhui116\Desktop\铁路风工程\图片\',fileName,'\'];
fileType = '-djpeg';
resolution = '-r600';

x = 1:length(tValue);
pValue = [pValue1;pValue2;pValue3;pValueSum];
error  = [myError1;myError2;myError3;myErrorSum];

% 预测结果
for i = 1:4
    figureName = ['Prediction results of ',modelName{i}];
    if i == 4
        figureName = ['Prediction results of ',modelName{i},' ',optimalName];
    end
    figure('Name',figureName);
    plot(x,tValue,x,pValue(i,:));
    legend('True','Predicted');
    xlabel('Times (3-min)');
    ylabel('Wind speed (m/s)');
    set(gcf,'visible','off')
    set(gcf,'Position',[200,200,800,400]);
    print(gcf,[root,figureName],fileType,resolution);
end


%% 输出预测结果
xlsName = '.\myResults\results_on_testing_dynamic.xlsx';
xlswrite(xlsName,tValue',1,'A');
xlswrite(xlsName,pValue1',1,'B');
xlswrite(xlsName,pValue2',1,'C');
xlswrite(xlsName,pValue3',1,'D');
xlswrite(xlsName,pValueSum',1,'E');


%% 保存环境
%环境
fullpath = mfilename('fullpath');
[path,name] = fileparts(fullpath);
save(['.\myStatus\',name,'_',datestr(datetime,'yyyy-mm-dd_HH-MM-SS')]);
%重要参数

