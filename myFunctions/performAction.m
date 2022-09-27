function [state,reward]=performAction(currentState,action)

global pValue tValue w1 w2 w3 actoffsets Lt MaxY MaxX

pValueSum=w1(currentState.map(1),currentState.map(2))*pValue(1,:)+w2(currentState.map(1),currentState.map(2))*pValue(2,:)+w3(currentState.map(1),currentState.map(2))*pValue(3,:);
rmse_old=rms(pValueSum-tValue);

state.map=currentState.map+actoffsets(action,:);
if(state.map(1)<1) state.map(1)=1; end
if(state.map(2)<1) state.map(2)=1; end
if(state.map(1)>MaxX) state.map(1)=MaxX; end
if(state.map(2)>MaxY) state.map(2)=MaxY; end
state.qTable = (state.map(1)-1)*Lt+state.map(2);

pValueSum=w1(state.map(1),state.map(2))*pValue(1,:)+w2(state.map(1),state.map(2))*pValue(2,:)+w3(state.map(1),state.map(2))*pValue(3,:);
rmse_new=rms(pValueSum-tValue);

if(rmse_new<rmse_old)  %½±
    reward=1+rmse_old-rmse_new;
else                   %³Í
    reward=-1+rmse_old-rmse_new;
end

end

