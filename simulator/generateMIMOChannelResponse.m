function [cfr, lastXptlst] = generateMIMOChannelResponse(rays, fc, OFDMConfig)

Nt = size(rays,1);
Nr = size(rays,2);
sc_spacing = OFDMConfig.Bandwidth / OFDMConfig.NumSC;
numSCs = OFDMConfig.NumValidSC;
sc_indices = (-floor(numSCs/2):ceil(numSCs/2)-1).';

cfr = zeros(numSCs, Nr, Nt);

for tx = 1:Nt
    for rx = 1:Nr
        pathData = rays{tx,rx};
        
        fscs = fc + sc_indices * sc_spacing; 
        
        antennaCFR = zeros(numSCs, 1);
        delays = [pathData.PropagationDelay];
        pathLosses = [pathData.PathLoss];
        phaseShifts = [pathData.PhaseShift];
        
        for sc = 1:numSCs
            fsc = fscs(sc);
            
            pl_factors = pathLosses + 20 * log10(fsc/fc);
            phase_terms = phaseShifts + 2*pi*(fsc - fc).*delays;
            
            coeffs = 10.^(-pl_factors/20) .* exp(-1j*phase_terms);
            
            antennaCFR(sc) = sum(coeffs);
        end
        
        cfr(:, rx, tx) = antennaCFR;
    end
end

pathData = rays{tx,rx};
txLocs = [pathData.TransmitterLocation];
lastintLocs = nan(3, numel(pathData));

for i = 1:numel(pathData)
    if ~isempty(pathData(i).Interactions) && isfield(pathData(i).Interactions, 'Location')
        intLocs = [pathData(i).Interactions.Location];
        lastintLocs(:,i) = intLocs(:, end);
    else
        lastintLocs(:,i) = nan(3,1);
    end
end

isLOS = [pathData.LineOfSight];
lastXptlst = txLocs .* isLOS + lastintLocs .* ~isLOS;

end