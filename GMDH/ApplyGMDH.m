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

function Yhat = ApplyGMDH(gmdh, X)

    nLayer = numel(gmdh.Layers);

    Z = X;
    for l=1:nLayer
        Z = GetLayerOutput(gmdh.Layers{l}, Z);
    end
    Yhat = Z;
    
end

function Z = GetLayerOutput(L, X)
    
    m = size(X,2);
    N = numel(L);
    Z = zeros(N,m);
    
    for k=1:N
        vars = L(k).vars;
        x = X(vars,:);
        Z(k,:) = L(k).f(x);
    end

end