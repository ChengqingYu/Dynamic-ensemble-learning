function [er,labels] = nntest(nn, x, y)
    labels = nnpredict(nn, x);
 
    er = sqrt(mean(labels-y).^2);
end
