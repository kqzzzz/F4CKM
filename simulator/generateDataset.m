function generateDataset(varargin)

    % close all; clear; clc;
    
    parser = inputParser;
    addParameter(parser, 'env', 'conferenceroom', @ischar);
    addParameter(parser, 'numSTAs', 2000, @isnumeric);
    addParameter(parser, 'seed', 1233, @isnumeric);
    addParameter(parser, 'distribution', 'ramdom', @ischar);
    addParameter(parser, 'apSize', [4,4], @isnumeric);
    addParameter(parser, 'staSize', [2,2], @isnumeric);
    addParameter(parser, 'fc', [2.415e9, 2.465e9], @isnumeric);
    %varargin = {'env', 'conferenceroom', 'seed', 1233, 'numSTAs', 2, 'distribution', 'random', 'apSize', [4,4], 'staSize', [2,2], 'fc', [2.415e9, 2.465e9]};
    parse(parser, varargin{:});
    params = parser.Results;

    %% ==================== Initialization ====================
    
    mapFileName = ['models/', params.env, '.stl'];
    [envDims, ~] = initEnvironment(mapFileName);

    % AP params
    if strcmp(params.env, 'conferenceroom')
        apConfig.centerPos = [-1.2; 0.0; 1.7];
    elseif strcmp(params.env, 'bedroom')
        apConfig.centerPos = [-1.2; 0.0; 2.7];
    elseif strcmp(params.env, 'office')
        apConfig.centerPos = [0.3; 3.7; 2.7];
    end
    apConfig.arraySize = params.apSize;
    apConfig.azimuth = 30;
    apConfig.elevation = 20;

    % STA params
    staConfig.numSTAs = params.numSTAs;
    staConfig.distribution = params.distribution;
    staConfig.arraySize = params.staSize;
    staConfig.seed = params.seed;
    
    % OFDM params
    OFDMConfig.Bandwidth = 20e6;
    OFDMConfig.NumSC = 64;
    OFDMConfig.NumValidSC = 52;
    
    % frequency band
    fc = params.fc;
    
    % save path
    if strcmp(staConfig.distribution, 'random')
        savepath = "datasets/" + params.env + "_2.4GHz_random" + num2str(params.numSTAs) +".mat";
    else
        savepath = "datasets/" + params.env + "_2.4GHz_uniform" +".mat";
    end
    
    subDatasets = cell(1,2);
    for i = 1:2
        subDatasets{i} = generateMIMOOFDMData(...
            mapFileName, envDims,...
            apConfig, staConfig, fc(i), OFDMConfig);
    end
    
    % align datasets by Rx positions
    [cleanUL, cleanDL] = alignByRxPos(subDatasets{1}, subDatasets{2});
    disp(['Aligned dataset size：', num2str(numel(cleanUL))]);
    
    % save dataset
    save(savepath, 'cleanUL', 'cleanDL', '-v7.3');
    fprintf('Dataset saved to: %s\n', savepath); 
