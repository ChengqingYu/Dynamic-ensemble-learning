%% SARSA优化集成权重系数

%% 清空环境变量
clc;clear;close all
disp('SARSA 集成权重优化');
%% 导入数据
[num,txt,raw]=xlsread('数据.xls');
global x1 x2 x3 tValue w1 w2 w3 MaxY MaxX kr actoffsets Lt
kr=0;
tValue=num(:,1);  %真值
x1=num(:,2);  %预测1
x2=num(:,3);  %预测2
x3=num(:,4);  %预测3
%% 根据参数条件，初始化
dt=0:0.001:1;
[w1,w2]=meshgrid(dt);    %创建网格坐标
w3=1-w1-w2;              %权重系数和为1
Lt=length(dt);           %网格尺寸

%% 参数初始化
MaxX=Lt;           %X坐标最大取值
MaxY=Lt;           %Y坐标最大取值
actoffsets=[-1,1;0,1;1,1;-1,0;0,0;1,0;-1,-1;0,-1;1,-1;];
ActionNum     =  9;       %动作数：在某个位置有9种动作（九宫格）
alpha         =  0.3;         %学习率
gamma         =  0.95;        %折扣系数
lambda        =  0.5;
epsilon       =  0.5;       %学习参数（随机概率阈值）
trials        =  100;        %设置最大尝试次数
maxiter=prod([MaxX MaxY]);               %每次尝试最大迭代次数
Q=zeros(prod([MaxX MaxY]),ActionNum);    %初始化Q值
sumQTable=zeros(1,trials);               %每次迭代的Q表累加和
change=zeros(1,trials-1);                %相邻两个Qsum的变化

%% 动图参数
h2 = animatedline;
axis([1 trials 0 100])
x = 1:trials;

%% 迭代求解
for i=1:trials
    clc;disp(['Trial: ',num2str(i),'/',num2str(trials)]);
    % 随机确定初始位置
    start=randi(Lt,1,2); 
    while w3(start(1),start(2))<0
        start=randi(Lt,1,2);
    end
    
    currentState.map = start;
    currentState.qTable = (start(1)-1)*Lt+start(2);       %地图坐标转Q表行号
    startState = currentState;
    
    %选择动作
    if rand<epsilon
        action=ceil(rand*ActionNum);
    else
        qRow = Q(currentState.qTable,:);
        topactions = find(qRow==max(qRow));                             %最大Q值动作的索引：1-9
        action = topactions(ceil(rand*length(topactions)));             %选择最优秀动作（当多个动作同时拥有最高Q值时随机选择一个可行动作）
    end
        
    %开始探索路径
    for j=1:maxiter
        %执行动作 转移状态 获得奖赏
        [state,reward]=performAction(currentState,action);
        %选择动作
        if rand<epsilon
            actionNew=ceil(rand*ActionNum);
        else
            qRow = Q(state.qTable,:);
            topactions = find(qRow==max(qRow));                                %最大Q值动作的索引：1-9
            actionNew = topactions(ceil(rand*length(topactions)));             %选择最优秀动作（当多个动作同时拥有最高Q值时随机选择一个可行动作）
        end
        %更新Q表
        Q(currentState.qTable,action) = Q(currentState.qTable,action) + alpha * (reward+gamma*Q(state.qTable,actionNew)-Q(currentState.qTable,action));
        %实际移动
        currentState = state;
        action = actionNew;
        %降低探索概率
        epsilon=epsilon/sqrt(i);
        if startState.qTable==currentState.qTable && j>10000  %移动到了原点且探索了足够远
            break;
        end
    end
    sumQTable(i) = sum(sum(Q));
    if i~= 1
        change(i-1)=(sumQTable(i)-sumQTable(i-1))./sumQTable(i-1)*100;
        myDisp({'迭代次数','Q表总和变化（%）'},[i,change(i-1)]);
        addpoints(h2,x(i),change(i-1));
        drawnow;
    else
        myDisp({'迭代次数','Q表总和'},[i,sumQTable(i)]);
    end
end

%% 结果分析
plot(sumQTable);
%最大Q值状态
[maxQstate.qTable,~] = find(Q==max(max(Q)));
maxQstate.map = qTable2Map(maxQstate.qTable);
W1=w1(maxQstate.map(1),maxQstate.map(2));
W2=w2(maxQstate.map(1),maxQstate.map(2));
W3=w3(maxQstate.map(1),maxQstate.map(2));
pValue=W1*x1+W2*x2+W3*x3;
myerrorMaxQ = myError(tValue,pValue);

%最终状态
W1=w1(currentState.map(1),currentState.map(2));
W2=w2(currentState.map(1),currentState.map(2));
W3=w3(currentState.map(1),currentState.map(2));
pValue=W1*x1+W2*x2+W3*x3;
myerrorFinal = myError(tValue,pValue);


if myerrorFinal.mape<myerrorMaxQ.mape
    myerror=myerrorFinal;
else
    myerror=myerrorMaxQ;
    W1=w1(maxQstate.map(1),maxQstate.map(2));
    W2=w2(maxQstate.map(1),maxQstate.map(2));
    W3=w3(maxQstate.map(1),maxQstate.map(2));
end

sarsa_weights = [W1,W2,W3];
myDisp({'w1','w2','w3'},[W1,W2,W3]);
myDisp({'MAE','MAPE','MSE','RMSE','SDE'},[myerror.mae,myerror.mape,myerror.mse,myerror.rmse,myerror.sde]);

%% 变量保存
%环境
fullpath = mfilename('fullpath'); 
[path,name]=fileparts(fullpath);
save(['.\myStatus\',name]);
clear fullpath path name;
%重要参数
save('.\myModel\sarsa_weights','sarsa_weights');

