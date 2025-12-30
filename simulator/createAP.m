function AP = createAP(centerPos, freq, arraySize, azimuth, elevation)

    c = physconst('lightspeed');
    lambda = c / freq;
    d = lambda / 2;
    
    rows = arraySize(1);
    cols = arraySize(2);
    
    [colGrid, rowGrid] = meshgrid(0:cols-1, 0:rows-1);
    
    x_local = (colGrid(:) - (cols-1)/2) * d;
    y_local = (rowGrid(:) - (rows-1)/2) * d;
    z_local = zeros(size(x_local));
    
    localPos = [x_local, y_local, z_local];
    
    Rz = rotz(azimuth);
    Ry = roty(elevation);
    R = Ry * Rz;
    
    globalPos = (R * localPos')' + centerPos';
    
    globalPos = [globalPos; centerPos'];
    
    antPosAP = globalPos';
    AP = txsite("cartesian", ...
        "AntennaPosition", antPosAP,...
        "TransmitterFrequency", freq, ...
        "TransmitterPower", 1);
end