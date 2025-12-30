function [alignUL, alignDL] = alignByRxPos(ulData, dlData)

pos2str = @(p) sprintf('%.6f,%.6f,%.6f', p(1), p(2), p(3));

ulMap = containers.Map();
for i = 1:numel(ulData)
    key = pos2str(ulData(i).RxPos(:, end));
    ulMap(key) = i;
end

dlMap = containers.Map();
for i = 1:numel(dlData)
    key = pos2str(dlData(i).RxPos(:, end));
    dlMap(key) = i;
end

commonKeys = intersect(ulMap.keys(), dlMap.keys());

ulIdx = cellfun(@(k) ulMap(k), commonKeys);
dlIdx = cellfun(@(k) dlMap(k), commonKeys);

[~, order] = sort(ulIdx);
alignUL = ulData(ulIdx(order));
alignDL = dlData(dlIdx(order));

end