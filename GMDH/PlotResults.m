%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML120
% Project Title: Time-Series Prediction using GMDH
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function PlotResults(Targets, Outputs, Title)

    Errors = Targets - Outputs;
    MSE = mean(Errors.^2);
    RMSE = sqrt(MSE);
    ErrorMean = mean(Errors);
    ErrorStd = std(Errors);
    
    subplot(2,2,[1 2]);
    plot(Targets);
    hold on;
    plot(Outputs);
    legend('Targets','Outputs');
    ylabel('Targets and Outputs');
    grid on;
    title(Title);
    
    subplot(2,2,3);
    plot(Errors);
    title(['MSE = ' num2str(MSE) ', RMSE = ' num2str(RMSE)]);
    ylabel('Errors');
    grid on;
    
    subplot(2,2,4);
    histfit(Errors, 50);
    title(['Error Mean = ' num2str(ErrorMean) ', Error StD = ' num2str(ErrorStd)]);

end