function testY_out = myPredict(name,model,testXn,outputps)
% testY_out = myPredict(name,model,testXn,outputps)
% name        input   预测器名称
% model       input   预测器
% testxn      input   归一化测试特征
% testyn_out  output  预测结果

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
        disp('指定方法错误');
end

end

