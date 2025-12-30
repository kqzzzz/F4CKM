function [cleanedRays, rxPos] = processRayData(rays, posSTAs, arraySize)
    
    numElementsPerSTA = prod(arraySize) + 1;
    [~, totalColumns] = size(rays);
    numSTAs = totalColumns / numElementsPerSTA;
    
    rxBlocks = cell(1, numSTAs);
    posBlocks = cell(1, numSTAs);
    for i = 1:numSTAs
        startIdx = (i-1)*numElementsPerSTA + 1;
        endIdx = i*numElementsPerSTA;
        rxBlocks{i} = rays(:, startIdx:endIdx);
        posBlocks{i} = posSTAs(:, startIdx:endIdx);
    end
    
    isValid = cellfun(@(b) ~any(cellfun(@isempty, b(:))), rxBlocks);
    cleanedRays = rxBlocks(isValid);
    rxPos = posBlocks(:,isValid);
end