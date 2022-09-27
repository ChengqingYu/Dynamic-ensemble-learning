function testY_out = myPredict(name,model,testXn,outputps)
% testY_out = myPredict(name,model,testXn,outputps)
% name        input   Ԥ��������
% model       input   Ԥ����
% testxn      input   ��һ����������
% testyn_out  output  Ԥ����

switch(lower(name))
    case 'bpnn'
        testYn_out=sim(model,testXn);
        testY_out=mapminmax('reverse',testYn_out,outputps);
    case 'enn'
        testYn_out=sim(model,testXn);
        testY_out=mapminmax('reverse',testYn_out,outputps);
    case 'grnn'
        testYn_out=sim(model,testXn);
        testY_out=mapminmax('reverse',testYn_out,outputps);
    case 'nar'
        testYn_out=sim(model,testXn);
        testY_out=mapminmax('reverse',testYn_out,outputps);
    case 'dbn'
        [~,testYn_out] = nntest(model,testXn',ones(length(testXn'),1));
        testY_out = mapminmax('reverse',testYn_out',outputps);
    case 'bilstm'
        testYn_out = predict(model,con2seq(testXn)');
        testY_out = mapminmax('reverse',testYn_out',outputps);
    case 'lstm'
        testYn_out = predict(model,con2seq(testXn)');
        testY_out = mapminmax('reverse',testYn_out',outputps);
    case 'gru'
        testYn_out = predict(model,con2seq(testXn)');
        testY_out = mapminmax('reverse',testYn_out',outputps);
    otherwise
        disp('ָ����������');
end

end

