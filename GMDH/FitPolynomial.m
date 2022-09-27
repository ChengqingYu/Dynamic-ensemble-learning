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

function p = FitPolynomial(x1, Y1, x2, Y2, vars)
    
    X1 = CreateRegressorsMatrix(x1);
    c = Y1*pinv(X1);
    
    Y1hat = c*X1;
    e1 = Y1- Y1hat;
    MSE1 = mean(e1.^2);
    RMSE1 = sqrt(MSE1);
    
    f = @(x) c*CreateRegressorsMatrix(x);
    
    Y2hat = f(x2);
    e2 = Y2- Y2hat;
    MSE2 = mean(e2.^2);
    RMSE2 = sqrt(MSE2);
    
    p.vars = vars;
    p.c = c;
    p.f = f;
    p.Y1hat = Y1hat;
    p.MSE1 = MSE1;
    p.RMSE1 = RMSE1;
    p.Y2hat = Y2hat;
    p.MSE2 = MSE2;
    p.RMSE2 = RMSE2;

end

function X = CreateRegressorsMatrix(x)

    X = [ones(1,size(x,2))
         x(1,:)
         x(2,:)
         x(1,:).^2
         x(2,:).^2
         x(1,:).*x(2,:)];

end