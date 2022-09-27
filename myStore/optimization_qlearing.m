%% Qѧϰ�Ż�����Ȩ��ϵ��

%% ��ջ�������
clc;clear;close all
disp('Q-Learning ����Ȩ���Ż�');

%% ���ݴ���
global pValue tValue w1 w2 w3 MaxY MaxX kr actoffsets Lt
results_on_valid=xlsread('results_on_valid.xlsx');
save('.\myResults\results_on_valid','results_on_valid');

pValue = results_on_valid(:,2:4)';
tValue = results_on_valid(:,1)';
kr=0;

%% ���ݲ�����������ʼ��
dt=0:0.001:1;
[w1,w2]=meshgrid(dt);    %������������
w3=1-w1-w2;              %Ȩ��ϵ����Ϊ1
Lt=length(dt);           %����ߴ�
rmse=ones(Lt,Lt)*100;    %����������ͬ�ߴ�ľ��� �洢�ڸ�Ȩ���µ�RMSE

%% ȫ������,Ѱ������
% for i=1:Lt
%     for j=1:Lt
%         if w3(i,j)>0
%             y=w1(i,j)*x1+w2(i,j)*x2+w3(i,j)*x3;
%             rmse(i,j)=rms(y-tValue);        %���������
%         end
%     end
% end
% [fi,fj]=find(rmse==min(min(rmse)));     %���rmse��Ӧ����������
% W1=w1(fi,fj);W2=w2(fi,fj);W3=w3(fi,fj);
% pValue = W1*x1+W2*x2+W3*x3;
% myerror = myError(tValue,pValue);
% minError=min(min(rmse));
% rmse(find(rmse == 100))=0;
% maxError=max(max(rmse));
% disp('ȫ������----����Ȩ�أ�');
% myDisp({'w1','w2','w3'},[w1(fi,fj),w2(fi,fj),w3(fi,fj)]);
% myDisp({'MAE','MAPE','MSE','RMSE','SDE'},[myerror.mae,myerror.mape,myerror.mse,myerror.rmse,myerror.sde]);
% myDisp({'ȫ��������'},[minError]);
% myDisp({'ȫ��������'},[maxError]);
% W=[W1,W2,W3];
% save('.\myModel\gweights','W');

%% ������ʼ��
MaxX                 =  Lt;              %X�������ȡֵ
MaxY                 =  Lt;              %Y�������ȡֵ
actoffsets           =  [-1,1;0,1;1,1;-1,0;0,0;1,0;-1,-1;0,-1;1,-1;];
ActionNum            =  9;               %����������ĳ��λ����9�ֶ������Ź���
alpha                =  0.3;             %ѧϰ��
gamma                =  0.95;            %�ۿ�ϵ��
lambda               =  0.5;
epsilon              =  0.5;             %ѧϰ���������������ֵ��
epsilon_decay_rate   =  0.995;
trials               =  10000;           %��������Դ���
maxiter              =  MaxX*MaxY;       %ÿ�γ�������������

Q=zeros(MaxX*MaxY,ActionNum);            %��ʼ��Qֵ
sumQTable=zeros(1,trials);               %ÿ�ε�����Q���ۼӺ�
change=zeros(1,trials-1);                %��������Qsum�ı仯

%% ��ͼ����
h1 = animatedline;
x = 1:trials;

%% �������
for i=1:trials
    % ���ȷ����ʼλ��
    start=randi(Lt,1,2); 
    while w3(start(1),start(2))<0
        start=randi(Lt,1,2);
    end
    
    currentState.map = start;
    currentState.qTable = (start(1)-1)*Lt+start(2);       %��ͼ����תQ���к�
    startState = currentState;
    
    % ��ʼ̽��·��
    for j=1:maxiter
        %ѡ����
        if rand<epsilon
            action=ceil(rand*ActionNum);   
        else
            qRow = Q(currentState.qTable,:);
            topactions = find(qRow==max(qRow));                             %���Qֵ������������1-9
            action = topactions(ceil(rand*length(topactions)));             %ѡ�������㶯�������������ͬʱӵ�����Qֵʱ���ѡ��һ�����ж�����
        end
        %ִ�ж��� ת��״̬ ��ý���
        [state,reward]=performAction(currentState,action);
        %����Q��
        Q(currentState.qTable,action) = Q(currentState.qTable,action) + alpha * (reward+gamma*max(Q(state.qTable,:))-Q(currentState.qTable,action));
        %ʵ���ƶ�
        currentState = state;
        if startState.qTable==currentState.qTable && j>10000                %�ƶ�����ԭ����̽�����㹻Զ ��ǰ��ֹ����̽��
            break;
        end
    end
    sumQTable(i) = sum(sum(Q));
    clc;
    disp('Q-Learning ����Ȩ���Ż�');
    disp(['Trial: ',num2str(i),'/',num2str(trials),'  Sum of Q table:',num2str(sumQTable(i))]);
    addpoints(h1,x(i),sumQTable(i));
    drawnow;
%     change(i-1)=(sumQTable(i)-sumQTable(i-1))./sumQTable(i-1)*100;

    %����̽������
    epsilon = epsilon * epsilon_decay_rate;
end

%% �������
plot(sumQTable);
%% ���Qֵ״̬
[maxQstate.qTable,~]     =  find(Q==max(max(Q)));
maxQstate.map            =  qTable2Map(maxQstate.qTable);
W1                       =  w1(maxQstate.map(1),maxQstate.map(2));
W2                       =  w2(maxQstate.map(1),maxQstate.map(2));
W3                       =  w3(maxQstate.map(1),maxQstate.map(2));
qlearning_weights_maxQ   =  [W1,W2,W3];
pValueSum                =  qlearning_weights_maxQ*pValue;
myerrorMaxQ              =  myError(tValue,pValueSum);

%% ����״̬
W1                        =  w1(currentState.map(1),currentState.map(2));
W2                        =  w2(currentState.map(1),currentState.map(2));
W3                        =  w3(currentState.map(1),currentState.map(2));
qlearning_weights_final   =  [W1,W2,W3];
pValueSum                 =  qlearning_weights_final*pValue;
myerrorFinal              =  myError(tValue,pValueSum);

if myerrorFinal.mape<myerrorMaxQ.mape
    myerror=myerrorFinal;
    qlearning_weights = qlearning_weights_maxQ;
    disp('�������״̬ѡȡ����Ȩ��');
else
    myerror=myerrorMaxQ;
    qlearning_weights = qlearning_weights_final;
    disp('�������Qֵ״̬ѡȡ����Ȩ��');
end

myDisp({'w1','w2','w3'},[W1,W2,W3]);
myDisp({'MAE','MAPE','MSE','RMSE','SDE'},[myerror.mae,myerror.mape,myerror.mse,myerror.rmse,myerror.sde]);

%% ��������
%����
fullpath = mfilename('fullpath');
[~,name]=fileparts(fullpath);
save(['.\myStatus\',name]);
clear fullpath path name;
%��Ҫ����
save('.\myModel\qlearning_weights','qlearning_weights');