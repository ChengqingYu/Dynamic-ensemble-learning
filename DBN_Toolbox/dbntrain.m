function dbn = dbntrain(dbn, x, opts)
    n = numel(dbn.rbm);%����

    dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts);%�Ե�һ�����ѵ��
    for i = 2 : n                              %������������forѭ������
        x = rbmup(dbn.rbm{i - 1}, x);
        dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, opts);
    end

end
