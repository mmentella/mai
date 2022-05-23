function [s,yhat,loss,z,r,c] = forward(x,y,uz,ur,uc,wz,wr,wc,bz,br,bc,v,bv,s0)
    
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
        s(:,t) = (1 - z(:,t)).*c(:,t) + z(:,t).*s(:,t-1);
        
        yhat(:,t) = softmax(v*s(:,t) + bv);
        loss(t) = sum(-y(:,t).*log(yhat(:,t)));
    end

end

function [dv,dbv,duz,dur,duc,dwz,dwr,dwc,dbz,dbr,dbc,ds0] = backward( ...
    x,y,uz,ur,uc,wz,wr,wc,bz,br,bc,v,bv,s0)
    
    [s,yhat,loss,z,r,c] = forward(x,y,uz,ur,uc,wz,wr,wc,bz,br,bc,v,bv,s0);
    [~, sentenceSize] = size(x);

    deltay = yhat - y;
    dbv = sum(deltay,2);

    dv = zeros(size(v));
    for i=1:sentenceSize
        dv = dv + deltay(:,t)*s(:,t)';
    end

    ds0 = zeros(size(s0));

    duz = zeros(size(uz));
    dur = zeros(size(ur));
    duc = zeros(size(uc));
    dwz = zeros(size(wz));
    dwr = zeros(size(wr));
    dwc = zeros(size(wc));
    dbz = zeros(size(bz));
    dbr = zeros(size(br));
    dbc = zeros(size(bc));

    dsSingle = v'*deltay;

    dst = zeros(size(dsSingle,1),1);
    for t=sentenceSize:-1:2
        dst = dst + dsSingle(:,t)
        dstcopy = dst;
        dtanhinput = (dst.*(1-z(:,t))).*(1-c(:,t).*c(:,t));

        dbc = dbc + dtanhinput;
        duc = duc + dtanhinput*x(:,t)';
        dwc = dwc+ dtanhinput*(s(:,t-1).*r(:,t))';

        dsr = wc'*dtanhinput;
        dst = dsr.*r(:,t);
        dsigInputR = dsr.*s(:,t-1).*r(:,t).*(1-r(:,t));
        dbr = dbr + dsigInputR;
        dur = dur + dsigInputR*x(:,t)';
        dwr = dwr + dsigInputR*s(:,t-1)';
        
        dst = dst + wr'*dsigInputR;
        dst = dst + dstcopy.*z(:,t);

        dz = dstcopy.*(s(:,t-1)-c(:,t));
        dsigInputZ = dz.*z(:,t).*(1-z(:,t));

        dbz = dbz + dsigInputZ;
        duz = duz + dsigInputZ*x(:,t)';
        dwz = dwz + dsigInputZ*s(:,t-1)';

        dst = dst + wz'*dsigInputZ;        
    end

    dst = dst + dsSingle(:,1);

    dtanhInput = (dst.*(1-z(:,1)).*(1-c(:,1).*c(:,1)));
    dbc = dbc + dtanhInput;
    duc = duc + dtanhInput*x(:,1)';
    dwc = dwc + dtanhInput*(s0.*r(:,1))';
    dsr = wc'*dtanhInput;
    ds0 = ds0 + dsr.*r(:,1);
    dsigInputR = dsr.*s0.*r(:,1).*(1-r(:,1));
    dbr = dbr + dsigInputR;
    dur = dur + dsigInputR*x(:,1)';
    dwr = dwr + dsigInputR*s0';
    ds0 = ds0 + wz'*dsigInputZ;
    
end
