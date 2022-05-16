function [s,yhat,s,z,r,c] = forward(x,y,uz,ur,uc,wz,wr,wc,bz,br,bc,v,bv,s0)
    
    [vocabularySize,sentenceSize] = size(x);
    memSize = size(v,2);

    s = zeros(memSize, sentenceSize);
    yhat = zeros(vocabularySize, sentenceSize);
    loss = zeros(sentenceSize, 1);
    z = zeros(memSize, sentenceSize);
    r = zeros(memSize, sentenceSize);
    c = zeros(memSize, sentenceSize);
    
    z(:,1) = sigmoid(uz*x(:,1) + wz*s0 + bz);
    r(:,1) = sigmoid(ur*x(:,1) + wr*s0 + br);
    c(:,1) = tanh(uc*x(:,1) + wc*(s0.*r(:,1)) + bc);
    s(:,1) = (1 - z(:,1)).*c(:,1) + z(:,1).*s0;
    
    yhat(:,1) = softmax(v*s(:,1) + bv);
    loss(1) = sum(-y(:,1).*log(yhat(:,1)));

    for t = 2:sentenceSize
        z(:,t) = sigmoid(uz*x(:,t) + wz*s(:,t-1) + bz);
        r(:,t) = sigmoid(ur*x(:,t) + wr*s(:,t-1) + br);
        c(:,t) = tanh(uc*x(:,t) + wc*(s(:,t-1).*r(:,t)) + bc);
        s(:,t) = (t - z(:,t)).*c(:,t) + z(:,t).*s(:,t-1);
        
        yhat(:,t) = softmax(v*s(:,t) + bv);
        loss(t) = sum(-y(:,t).*log(yhat(:,t)));
    end

end