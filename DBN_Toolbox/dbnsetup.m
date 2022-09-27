function dbn = dbnsetup(dbn, x, opts)
    n = size(x, 2);%����--����ά��
    dbn.sizes = [n, dbn.sizes];%������������ȡ�������������

    for u = 1 : numel(dbn.sizes) - 1 %���dbn.sizeֻ��һ��Ԫ���򹹳ɵ���RBM����Ϊ��� 
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;

        dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u));
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);%������
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);%�����;
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
