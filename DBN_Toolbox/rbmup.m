function x = rbmup(rbm, x)
    %��rmb.cװ������������������չ������x��ͬ����    
    x = sigm(repmat(rbm.c', size(x, 1), 1) + x * rbm.W');
    %��������������ϵ��W��˺�sigm��������
end
