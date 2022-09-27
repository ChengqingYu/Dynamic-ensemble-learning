function dbn = dbntrain(dbn, x, opts)
    n = numel(dbn.rbm);%层数

    dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts);%对第一层进行训练
    for i = 2 : n                              %若超过两层则for循环启用
        x = rbmup(dbn.rbm{i - 1}, x);
        dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, opts);
    end

end
