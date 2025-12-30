function dataset = generateMIMOOFDMData(mapFile, envDims, apCfg, staCfg, fc, OFDMConfig)

    % viewer = siteviewer("SceneModel",mapFile,"Transparency",0.25);
    [AP, STAs, posSTAs] = createNodes(apCfg, staCfg, envDims, fc);
    % show(AP);
    
    % ray tracing
    pm = propagationModel("raytracing",...
        "Method","sbr",...
        "CoordinateSystem","cartesian",...
        "MaxNumReflections",2, ...
        "SurfaceMaterial","concrete");
    
    tic;
    rays = raytrace(AP, STAs, pm, "Map", mapFile);
    elapsedTime = toc;
    disp(['Raytracing at ', num2str(fc/1e9) , 'GHz completed. Runtime：', num2str(elapsedTime), ' s']);
    
    [cleanedRays, rxPositions] = processRayData(rays, posSTAs, staCfg.arraySize);
    fprintf('Empty rays removed.\n')
    
    fprintf('CSI dataset generating...\n')
    dataset = buildChannelDataset(cleanedRays, rxPositions,...
        apCfg.centerPos, fc, OFDMConfig);
end