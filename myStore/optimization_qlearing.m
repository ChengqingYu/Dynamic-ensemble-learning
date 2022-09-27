%% Q学习优化集成权重系数

%% 清空环境变量
clc;clear;close all
disp('Q-Learning 集成权重优化');

%% 数据处理
global pValue tValue w1 w2 w3 MaxY MaxX kr actoffsets Lt
results_on_valid=xlsread('results_on_valid.xlsx');
save('.\myResults\results_on_valid','results_on_valid');

pValue = results_on_valid(:,2:4)';
tValue = results_on_valid(:,1)';
kr=0;

%% 根据参数条件，初始化
dt=0:0.001:1;
[w1,w2]=meshgrid(dt);    %创建网格坐标
w3=1-w1-w2;              %权重系数和为1
Lt=length(dt);           %网格尺寸
rmse=ones(Lt,Lt)*100;    %建立与网格同尺寸的矩阵 存储在各权重下的RMSE

%% 全局搜索,寻找最优
% for i=1:Lt
%     for j=1:Lt
%         if w3(i,j)>0
%             y=w1(i,j)*x1+w2(i,j)*x2+w3(i,j)*x3;
%             rmse(i,j)=rms(y-tValue);        %均方根误差
%         end
%     end
% end
% [fi,fj]=find(rmse==min(min(rmse)));     %最低rmse对应的网格坐标
% W1=w1(fi,fj);W2=w2(fi,fj);W3=w3(fi,fj);
% pValue = W1*x1+W2*x2+W3*x3;
% myerror = myError(tValue,pValue);
% minError=min(min(rmse));
% rmse(find(rmse == 100))=0;
% maxError=max(max(rmse));
% disp('全局搜索----最优权重：');
% myDisp({'w1','w2','w3'},[w1(fi,fj),w2(fi,fj),w3(fi,fj)]);
% myDisp({'MAE','MAPE','MSE','RMSE','SDE'},[myerror.mae,myerror.mape,myerror.mse,myerror.rmse,myerror.sde]);
% myDisp({'全局最低误差'},[minError]);
% myDisp({'全局最高误差'},[maxError]);
% W=[W1,W2,W3];
% save('.\myModel\gweights','W');

%% 参数初始化
MaxX                 =  Lt;              %X坐标最大取值
MaxY                 =  Lt;              %Y坐标最大取值
actoffsets           =  [-1,1;0,1;1,1;-1,0;0,0;1,0;-1,-1;0,-1;1,-1;];
ActionNum            =  9;               %动作数：在某个位置有9种动作（九宫格）
alpha                =  0.3;             %学习率
gamma                =  0.95;            %折扣系数
lambda               =  0.5;
epsilon              =  0.5;             %学习参数（随机概率阈值）
epsilon_decay_rate   =  0.995;
trials               =  10000;           %设置最大尝试次数
maxiter              =  MaxX*MaxY;       %每次尝试最大迭代次数

Q=zeros(MaxX*MaxY,ActionNum);            %初始化Q值
sumQTable=zeros(1,trials);               %每次迭代的Q表累加和
change=zeros(1,trials-1);                %相邻两个Qsum的变化

%% 动图参数
h1 = animatedline;
x = 1:trials;

%% 迭代求解
for i=1:trials
    % 随机确定初始位置
    start=randi(Lt,1,2); 
    while w3(start(1),start(2))<0
        start=randi(Lt,1,2);
    end
    
    currentState.map = start;
    currentState.qTable = (start(1)-1)*Lt+start(2);       %地图坐标转Q表行号
    startState = currentState;
    
    % 开始探索路径
    for j=1:maxiter
        %选择动作
        if rand<epsilon
            action=ceil(rand*ActionNum);   
        else
            qRow = Q(currentState.qTable,:);
            topactions = find(qRow==max(qRow));                             %最大Q值动作的索引：1-9
            action = topactions(ceil(rand*length(topactions)));             %选择最优秀动作（当多个动作同时拥有最高Q值时随机选择一个可行动作）
        end
        %执行动作 转移状态 获得奖赏
        [state,reward]=performAction(currentState,action);
        %更新Q表
        Q(currentState.qTable,action) = Q(currentState.qTable,action) + alpha * (reward+gamma*max(Q(state.qTable,:))-Q(currentState.qTable,action));
        %实际移动
        currentState = state;
        if startState.qTable==currentState.qTable && j>10000                %移动到了原点且探索了足够远 提前终止本次探索
            break;
        end
    end
    sumQTable(i) = sum(sum(Q));
    clc;
    disp('Q-Learning 集成权重优化');
    disp(['Trial: ',num2str(i),'/',num2str(trials),'  Sum of Q table:',num2str(sumQTable(i))]);
    addpoints(h1,x(i),sumQTable(i));
    drawnow;
%     change(i-1)=(sumQTable(i)-sumQTable(i-1))./sumQTable(i-1)*100;

    %降低探索概率
    epsilon = epsilon * epsilon_decay_rate;
end

%% 结果分析
plot(sumQTable);
%% 最大Q值状态
[maxQstate.qTable,~]     =  find(Q==max(max(Q)));
maxQstate.map            =  qTable2Map(maxQstate.qTable);
W1                       =  w1(maxQstate.map(1),maxQstate.map(2));
W2                       =  w2(maxQstate.map(1),maxQstate.map(2));
W3                       =  w3(maxQstate.map(1),maxQstate.map(2));
qlearning_weights_maxQ   =  [W1,W2,W3];
pValueSum                =  qlearning_weights_maxQ*pValue;
myerrorMaxQ              =  myError(tValue,pValueSum);

%% 最终状态
W1                        =  w1(currentState.map(1),currentState.map(2));
W2                        =  w2(currentState.map(1),currentState.map(2));
W3                        =  w3(currentState.map(1),currentState.map(2));
qlearning_weights_final   =  [W1,W2,W3];
pValueSum                 =  qlearning_weights_final*pValue;
myerrorFinal              =  myError(tValue,pValueSum);

if myerrorFinal.mape<myerrorMaxQ.mape
    myerror=myerrorFinal;
    qlearning_weights = qlearning_weights_maxQ;
    disp('依据最后状态选取最优权重');
else
    myerror=myerrorMaxQ;
    qlearning_weights = qlearning_weights_final;
    disp('依据最大Q值状态选取最优权重');
end

myDisp({'w1','w2','w3'},[W1,W2,W3]);
myDisp({'MAE','MAPE','MSE','RMSE','SDE'},[myerror.mae,myerror.mape,myerror.mse,myerror.rmse,myerror.sde]);

%% 变量保存
%环境
fullpath = mfilename('fullpath');
[~,name]=fileparts(fullpath);
save(['.\myStatus\',name]);
clear fullpath path name;
%重要参数
save('.\myModel\qlearning_weights','qlearning_weights');