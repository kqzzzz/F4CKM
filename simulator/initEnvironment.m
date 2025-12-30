function [envDims, vertex] = initEnvironment(mapFileName)
    [vertex,~] = stlread(mapFileName);
    xy_offset = 0.1;
    z_offset = 0.1;
    x_range = [min(vertex.Points(:,1))+xy_offset, max(vertex.Points(:,1))-xy_offset];
    y_range = [min(vertex.Points(:,2))+xy_offset, max(vertex.Points(:,2))-xy_offset];
    z_range = [min(vertex.Points(:,3)), max(vertex.Points(:,3))-z_offset];
    envDims = [x_range; y_range; z_range];
end