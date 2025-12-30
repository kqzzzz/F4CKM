function [elementPositions, STAs] = createSTAs(env_dims, arraySize, val, distribution, freq)

    c = physconst('lightspeed');
    lambda = c / freq;
    d = lambda / 2;

    if distribution == "uniform"
        x_centers = (env_dims(1,1)+0.05) : val : (env_dims(1,2)-0.05);
        y_centers = (env_dims(2,1)+0.05) : val : (env_dims(2,2)-0.05);
        z_fixed = 1.2;
        [X, Y] = meshgrid(x_centers, y_centers);
        Z = z_fixed * ones(size(X));
        centers = [X(:), Y(:), Z(:)];
    else
        numSTA = val;
        centers = [rand(numSTA,1)*(env_dims(1,2)-env_dims(1,1)-0.1) + (env_dims(1,1)+0.05),...
                   rand(numSTA,1)*(env_dims(2,2)-env_dims(2,1)-0.1) + (env_dims(2,1)+0.05),...
                   rand(numSTA,1)*(env_dims(3,2)-env_dims(3,1)) + env_dims(3,1)];
    end

    numRX = size(centers, 1);
    rows = arraySize(1);
    cols = arraySize(2);
    
    [colGrid, rowGrid] = meshgrid(0:cols-1, 0:rows-1);
    x_local = (colGrid(:) - (cols-1)/2) * d;
    y_local = (rowGrid(:) - (rows-1)/2) * d;
    localPos = [x_local, y_local, zeros(rows*cols,1)];

    elementPositions = zeros((rows*cols + 1)*numRX, 3);
    for i = 1:numRX
        globalAntennas = localPos + centers(i,:);
        
        allElements = [globalAntennas; centers(i,:)];
        
        elementPositions((i-1)*(rows*cols + 1)+1 : i*(rows*cols + 1), :) = allElements;
    end

    elementPositions = elementPositions';
    STAs = rxsite('cartesian',...
        'AntennaPosition', elementPositions,...
        'AntennaAngle', 0,...
        'ReceiverSensitivity', -85);
    centers = centers';
end