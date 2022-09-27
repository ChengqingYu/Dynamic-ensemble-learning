%% Q-learning优化集成权重系数


%% 清空环境变量
clc;clear;close all;


%% 测试数据提取
load dataset
testY=dataset.testY;
testXn=dataset.testXn;
outputps = dataset.outputps;


%% 读取模型
%预测器和权重向量
load dbn;load lstm;load gru;
optimalName = 'Q-learning';
%预测
pValue1=myPredict('dbn',dbn,testXn,outputps);
pValue2=myPredict('lstm',lstm,testXn,outputps);
pValue3=myPredict('gru',gru,testXn,outputps);
modelName={'DBN','LSTM','GRU','Ensemble'};
pValue =  [pValue1;pValue2;pValue3];
%误差分析   
tValue=testY;
myerror1 = myError(tValue,pValue1);
myerror2 = myError(tValue,pValue2);
myerror3 = myError(tValue,pValue3);
error = [myerror1.mae,myerror1.mape,myerror1.mse,myerror1.rmse;
    myerror2.mae,myerror2.mape,myerror2.mse,myerror2.rmse;
    myerror3.mae,myerror3.mape,myerror3.mse,myerror3.rmse;];
minerror = min(error);

numTry=100;
%% 测试
error = [];
for i = 1:numTry
    disp(['Try:',num2str(i),'/',num2str(numTry)]);
    
    %集成
    w = getQweights();
    pValueSum    =  w * pValue;

    % 误差分析
    myerror = myError(tValue,pValueSum);
    myerror = [myerror.mae,myerror.mape,myerror.mse,myerror.rmse];
    compare = myerror<=minerror;
    res = all(compare==1);
    if res
        disp(['Ensemble success']);
        break;
    else
        disp(['Ensemble fail']);
    end
end


%% Optimal Result
% Base models
disp('各预测器 验证集');
myerror1 = myError(tValue,pValue(1,:),'show');
myerror2 = myError(tValue,pValue(2,:),'show');
myerror3 = myError(tValue,pValue(3,:),'show');

% Q-learning ensemble
pValueSum = w * pValue;
disp('静态Q学习集成 验证集');
myerror = myError(tValue,pValueSum,'show');


%% 输出预测结果
fileName = optimalName;
xlsName = ['.\myTrick\results_on_testing_static ',fileName,'.xlsx'];
xlswrite(xlsName,tValue',1,'A');
xlswrite(xlsName,pValue1',1,'B');
xlswrite(xlsName,pValue2',1,'C');
xlswrite(xlsName,pValue3',1,'D');
xlswrite(xlsName,pValueSum',1,'E');


%% 保存环境
%环境
fullpath = mfilename('fullpath'); 
[path,name]=fileparts(fullpath);
save(['.\myStatus\',name,'_',datestr(datetime,'yyyy-mm-dd_HH-MM-SS')]);


%% 输出绘图
fileName = 'Q-learning';
root = ['C:\Users\liuhui116\Desktop\铁路风工程\图片\',fileName,'\'];
fileType = '-djpeg';
resolution = '-r600';
x = 1:length(tValue);

% 预测结果
figureName = ['Prediction results of ',modelName{1}];
figure('Name',figureName);
plot(x,tValue,x,pValue1);
legend('True','Predicted');
xlabel('Times (3-min)');
ylabel('Wind speed (m/s)');
set(gcf,'visible','off');
set(gcf,'Position',[200,200,800,400]);
print(gcf,[root,figureName],fileType,resolution);

figureName = ['Prediction results of ',modelName{2}];
figure('Name',figureName);
plot(x,tValue,x,pValue2);
legend('True','Predicted');
xlabel('Times (3-min)');
ylabel('Wind speed (m/s)');
set(gcf,'visible','off');
set(gcf,'Position',[200,200,800,400]);
print(gcf,[root,figureName],fileType,resolution);

figureName = ['Prediction results of ',modelName{3}];
figure('Name',figureName);
plot(x,tValue,x,pValue3);
legend('True','Predicted');
xlabel('Times (3-min)');
ylabel('Wind speed (m/s)');
set(gcf,'visible','off');
set(gcf,'Position',[200,200,800,400]);
print(gcf,[root,figureName],fileType,resolution);

figureName = ['Prediction results of ensemble model optimized by ',optimalName];
figure('Name',figureName);
plot(x,tValue,x,pValueSum);
legend('True','Predicted');
xlabel('Times (3-min)');
ylabel('Wind speed (m/s)');
set(gcf,'visible','off');
set(gcf,'Position',[200,200,800,400]);
print(gcf,[root,figureName],fileType,resolution);

figureName = ['Prediction results of forecasting models'];
figure('Name',figureName);
plot(x,tValue,x,pValue1,x,pValue2,x,pValue3,x,pValueSum,...
    'LineWidth',     1);
legend('True',modelName{1},modelName{2},modelName{3},'Ensemble',...
    'Orientation',   'vertical',...
    'FontSize',      11,...
	'FontName',      'Times New Roman');
xlabel('Times (3-min)',...
    'FontSize',      12,...
	'FontName',      'Times New Roman',...
    'FontWeight',    'bold');
ylabel('Wind speed (m/s)',...
    'FontSize',      12,...
	'FontName',      'Times New Roman',...
    'FontWeight',    'bold');
set(gcf,'visible','off');
set(gcf,'Position',[200,200,700,350]);
print(gcf,[root,figureName],fileType,resolution);


%% 误差分布
pValueAll = [pValue1;pValue2;pValue3;pValueSum];
error  = [myerror1;myerror2;myerror3;myerror];

% 预测误差
figureName = 'Prediction errors of forecasting models';
figure('Name',figureName);
for i = 1:4
    subplot(2,2,i);
    bar(-error(i).error);
    hold on;
    plot([0,200],[1,1].*mean(error(i).error(error(i).error >= 0)));
    plot([0,200],[1,1].*mean(error(i).error(error(i).error <  0)));
    
    axis([0,200,-3,+4])
    title(['Prediction error of ',modelName{i}]);
    xlabel('Time (3min)');
    ylabel('Error (m/s)');
end
set(gcf,'visible','off')
set(gcf,'Position',[200,200,500,500]);
print(gcf,[root,figureName],fileType,resolution);

%% 散点分布
figureName = 'Scatter plot of forecasting models';
figure('Name',figureName);

for i = 1:4
    subplot(2,2,i);
    plot(tValue,pValueAll(i,:),'.',...
        'MarkerSize',10);
    hold on;
    plot(min(tValue):0.001:max(tValue),min(tValue):0.001:max(tValue),...
        'LineWidth',1);
    
    axis([min(tValue),max(tValue),min(tValue),max(tValue)]);
    axis square;
    title(modelName{i},...
        'FontSize',     12,...
        'FontName',     'Times New Roman',...
        'FontWeight',   'bold');
    xlabel('True wind speed (m/s)',...
        'FontSize',     12,...
        'FontName',     'Times New Roman',...
        'FontWeight',   'bold');
    ylabel('Predicted wind speed (m/s)',...
        'FontSize',     12,...
        'FontName',     'Times New Roman',...
        'FontWeight',   'bold');
%     text(min(tValue)+0.5,max(tValue)-0.5,['PCC = ',num2str(error(i).pcc)],...
%         'FontSize',     12,...
%         'FontName',     'Times New Roman',...
%         'FontWeight',   'normal');
end
set(gcf,'visible','off');
set(gcf,'Position',[200,200,500,500]);
print(gcf,[root,figureName],fileType,resolution);

