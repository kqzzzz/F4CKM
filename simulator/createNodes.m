function [AP, STAs, posSTAs] = createNodes(apCfg, staCfg, envDims, fc)
    
    AP = createAP(apCfg.centerPos, fc,...
        apCfg.arraySize, apCfg.azimuth, apCfg.elevation);
    
    RandStream.setGlobalStream(RandStream("mt19937ar","Seed",staCfg.seed));
    if strcmp(staCfg.distribution, 'uniform')
        refFreq = floor(fc / 1e8) * 1e8;
        staSep = physconst('LightSpeed') / refFreq;
        [posSTAs, STAs] = createSTAs(envDims, staCfg.arraySize, staSep, "uniform", fc);
    else
        [posSTAs, STAs] = createSTAs(envDims, staCfg.arraySize, staCfg.numSTAs, "random", fc);
    end
end