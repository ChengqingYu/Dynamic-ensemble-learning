function qlearning_weights = getQweights()

%% Load Dataset
global pValue tValue w1 w2 w3 MaxY MaxX kr
results_on_valid=xlsread('results_on_valid.xlsx');
save('.\myResults\results_on_valid','results_on_valid');
pValue = results_on_valid(:,2:4)';
tValue = results_on_valid(:,1)';

%% Q-learning
%% Environments
kr=0;
dt=0:0.001:1;
[w1,w2]=meshgrid(dt);
w3=1-w1-w2;
Lt=length(dt);
rmse=ones(Lt,Lt)*100;

% Initialize parameters
MaxX          =  Lt;           %X坐标最大取值
MaxY          =  Lt;           %Y坐标最大取值
ActionNum     =  9;            %动作数：在某个位置有9种动作（九宫格）
actoffsets    =  [-1 0;-1 1;-1 -1;1 0;1 1;1 -1;0 1;0 0;0 -1];
alpha         =  0.3;          %学习率
gamma         =  0.95;         %折扣系数
lambda        =  0.5;
epsilon       =  0.5;          %学习参数（随机概率阈值）
trials        =  100;          %最大尝试次数
maxiter       =  100;          %最大迭代次数
convgoal      =  0.25;         %收敛目标
avgtrials     =  100;           %收敛时平均迭代次数
k             =  1;            %收敛性数组下标
Q             =  zeros(prod([ActionNum MaxX MaxY]),1);    %初始化Q值（一维矩阵，实际为，1001^2 * 9矩阵的展开）
Q_s_all       =  [];

% Trial
% reward_store = zeros(1,trials);
% reward_episode = zeros(1,maxiter);
for i=1:trials    
    % Randomize initial state
    action  =  0;
    E       =  0*Q;
    state   =  randi(Lt,1,2); 
    while w3(state(1),state(2))<0
        state = randi(Lt,1,2);
    end
    
    % Step
    for j=1:maxiter
        % Calculate Q value
        if j>1
            ix       =  ndi2lin([1 state(1) state(2)],[ActionNum MaxX MaxY]);                %新状态Q值索引 新状态state执行动作1的位置
            qix      =  ndi2lin([action Prestate(1) Prestate(2)],[ActionNum MaxX MaxY]);     %旧状态Q值索引 旧状态prestate执行action动作的位置(s,a)
            delta    =  reward+gamma*max(Q(ix:ix+ActionNum-1))-Q(qix);
            E(qix)   =  1;
            Q        =  Q+alpha*delta*E;                                                     %通过E确保只更新qix位置的Q值
            E        =  gamma*lambda*E*~exploring;                                           %若探索则E全部清0 若不探索则保持该qix位置为1
            Q_s      =  [Q(qix) state];
            Q_s_all  =  [Q_s_all; Q_s];                                             %连续保存每一次的[前一状态s执行选定动作a的Q值,下一状态] 第二列即为采样轨迹
        end
        
        % Choose an action
        if rand < epsilon                   % random
            action      =  ceil(rand*ActionNum);
            exploring   =  1;
        else                                % based on max Q
            ix          =  ndi2lin([1 state(1) state(2)],[ActionNum MaxX MaxY]);    %当前状态对应第一个动作的Q值索引
            ix          =  ix:ix+ActionNum-1;                                       %当前状态对应的所有动作的Q值索引 长度为9
            topactions  =  find(Q(ix) == max(Q(ix)));                               %最大Q值动作的索引：1-9
            action      =  topactions(ceil(rand*length(topactions)));               %选择最优秀动作（当多个动作同时拥有最高Q值时随机选择一个可行动作）[1-9]
            exploring   =  0;
        end
        
        epsilon         =  epsilon / trials;                  % Reduce randomness
        Prestate        =  state;                             % Store current state
        [state,reward]  =  step(state,action,actoffsets);     % Perform the action
%         reward_episode(j) = reward;
    end
    
%     reward_store(i) = mean(reward_episode);
    stats(k)=j;   % Store step num
    if k>avgtrials && std(stats(length(stats)-avgtrials+1:length(stats)))<convgoal && j<300
        break;    % End the current trail
    end
    k=k+1;
end

%% Get Final Weights
[mq,mi]      =  max(Q_s_all(:,1));  %最大Q值及其位置
Max_state    =  Q_s_all(mi,2:3);    %最大Q值对应的下一状态
w_end        =  [w1(state(1),state(2)),w2(state(1),state(2)),w3(state(1),state(2))];
w_max        =  [w1(Max_state(1),Max_state(2)),w2(Max_state(1),Max_state(2)),w3(Max_state(1),Max_state(2))];
yS1          =  w_end * pValue;
yS2          =  w_max * pValue;
rmse1        =  rms(yS1-tValue);
rmse2        =  rms(yS2-tValue);
if rmse1 > rmse2
    w = w_max;
else
    w = w_end;
end

%% Save variables
qlearning_weights=w;

end

