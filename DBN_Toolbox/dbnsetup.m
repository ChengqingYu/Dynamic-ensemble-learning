function dbn = dbnsetup(dbn, x, opts)
    n = size(x, 2);%输入--样本维数
    dbn.sizes = [n, dbn.sizes];%将样本数与提取的特征向量组合

    for u = 1 : numel(dbn.sizes) - 1 %如果dbn.size只有一个元素则构成单层RBM否则为多层 
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;

        dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u));
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);%层输入
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);%层输出;
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
