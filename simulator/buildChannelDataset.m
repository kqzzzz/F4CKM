function dataset = buildChannelDataset(rays, rxPos, txPos, fc, OFDMConfig)
    
    numChan = numel(rays);
    dataset = struct('TxPos', cell(numChan,1),...
        'RxPos',[], 'LastXPts', [], 'RxID',[], 'Frequency',[], 'CSI',[]);
    
    startTime = tic;
    reportInterval = 0.1;
    nextReport = reportInterval;
    
    % main loop
    for i = 1:numChan
        [cfrRaw, lastXptlst] = generateMIMOChannelResponse(rays{i}, fc, OFDMConfig);
        
        dataset(i).TxPos = txPos;
        dataset(i).RxPos = rxPos{i};
        dataset(i).LastXPts = lastXptlst;
        dataset(i).RxID = i;
        dataset(i).Frequency = fc/1e9;
        dataset(i).CSI = cfrRaw;
    
        progress = i/numChan;
        if progress >= nextReport || i == numChan
            elapsed = toc(startTime); 
            
            progressBar = [repmat('=',1, floor(progress*20)) '>' repmat(' ',1, 20-floor(progress*20))];
            fprintf('[%s] %.1f%% Runtime: %.2fs\n',...
                   progressBar, progress*100, elapsed);
            
            nextReport = nextReport + reportInterval;
        end
    end
    
end