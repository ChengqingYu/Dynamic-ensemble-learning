function x = rbmup(rbm, x)
    %将rmb.c装换成列向量，并且扩展到成与x相同的列    
    x = sigm(repmat(rbm.c', size(x, 1), 1) + x * rbm.W');
    %输出层由输入层与系数W相乘后经sigm激活后输出
end
