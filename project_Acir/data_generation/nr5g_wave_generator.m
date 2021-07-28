%% Generate 5G Downlink FRC Waveform
% TODO: Adapt this generator from MATLAB R2021b version!
frequency_ranges = ["FR1", "FR2"];
channel_bandwidths = [5, 10, 15, 20, 25, 30, 40, 50, 100, 200];
subcarrier_spacings = [15, 30, 60, 120];
modulation_coding_schemes = ["QPSK", "64QAM", "256QAM"];

rb_values = [...
    25 52 79 106 133 160 216 270 NaN NaN;...
    11 24 38 51 65 78 106 133 273 NaN;...
    NaN 11 18 24 31 38 51 65 135 NaN;...
    NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN...
    ];
rb_values(:,:,2) = [...
    NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN;...
    NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN;...
    NaN NaN NaN NaN NaN NaN NaN 66 132 264;...
    NaN NaN NaN NaN NaN NaN NaN 32 66 132;...
    ];

for fr = frequency_ranges
    for cbw = channel_bandwidths
        for scs = subcarrier_spacings
            for mcs = modulation_coding_schemes
                try
                    nSizeGrid = rb_values(subcarrier_spacings == scs, channel_bandwidths == cbw, frequency_ranges == fr);

                    % Downlink FRC configuration:
                    cfgDLFRC = nrDLCarrierConfig;
                    cfgDLFRC.FrequencyRange = fr;
                    cfgDLFRC.ChannelBandwidth = cbw;
                    cfgDLFRC.NCellID = 1;
                    cfgDLFRC.NumSubframes = 10;
                    cfgDLFRC.WindowingPercent = 0;
                    cfgDLFRC.SampleRate = [];
                    cfgDLFRC.CarrierFrequency = 0;

                    % SCS specific carriers
                    scscarrier = nrSCSCarrierConfig;
                    scscarrier.SubcarrierSpacing = scs;
                    scscarrier.NSizeGrid = nSizeGrid;
                    scscarrier.NStartGrid = 0;

                    cfgDLFRC.SCSCarriers = {scscarrier};

                    % Bandwidth Parts
                    bwp = nrWavegenBWPConfig;
                    bwp.BandwidthPartID = 1;
                    bwp.Label = 'BWP1';
                    bwp.SubcarrierSpacing = scs;
                    bwp.CyclicPrefix = 'normal';
                    bwp.NSizeBWP = nSizeGrid;
                    bwp.NStartBWP = 0;

                    cfgDLFRC.BandwidthParts = {bwp};

                    % Synchronization Signals Burst
                    ssburst = nrWavegenSSBurstConfig;
                    ssburst.BlockPattern = 'Case A';
                    ssburst.TransmittedBlocks = [1 0 0 0];
                    ssburst.Period = 10;
                    ssburst.NCRBSSB = [];
                    ssburst.KSSB = 0;
                    ssburst.DataSource = 'MIB';
                    ssburst.DMRSTypeAPosition = 2;
                    ssburst.CellBarred = false;
                    ssburst.IntraFreqReselection = false;
                    ssburst.PDCCHConfigSIB1 = 0;
                    ssburst.SubcarrierSpacingCommon = scs;
                    ssburst.Enable = true;
                    ssburst.Power = 0;

                    cfgDLFRC.SSBurst = ssburst;

                    % CORESET and Search Space Configuration
                    coreset = nrCORESETConfig;
                    coreset.CORESETID = 1;
                    coreset.Label = 'CORESET1';
                    coreset.FrequencyResources = ones([1 4]);
                    coreset.Duration = 2;
                    coreset.CCEREGMapping = 'noninterleaved';
                    coreset.REGBundleSize = 2;
                    coreset.InterleaverSize = 2;
                    coreset.ShiftIndex = 0;

                    cfgDLFRC.CORESET = {coreset};

                    % Search Spaces
                    searchspace = nrSearchSpaceConfig;
                    searchspace.SearchSpaceID = 1;
                    searchspace.Label = 'SearchSpace1';
                    searchspace.CORESETID = 1;
                    searchspace.SearchSpaceType = 'common';
                    searchspace.StartSymbolWithinSlot = 0;
                    searchspace.SlotPeriodAndOffset = [1 0];
                    searchspace.Duration = 1;
                    searchspace.NumCandidates = [8 8 4 2 0];

                    cfgDLFRC.SearchSpaces = {searchspace};

                    % PDCCH Instances Configuration
                    pdcch = nrWavegenPDCCHConfig;
                    pdcch.Enable = false;
                    pdcch.Label = 'PDCCH1';
                    pdcch.Power = 0;
                    pdcch.BandwidthPartID = 1;
                    pdcch.SearchSpaceID = 1;
                    pdcch.AggregationLevel = 1;
                    pdcch.AllocatedCandidate = 1;
                    pdcch.SlotAllocation = 1:9;
                    pdcch.Period = 10;
                    pdcch.Coding = false;
                    pdcch.DataBlockSize = 20;
                    pdcch.DataSource = 0;
                    pdcch.RNTI = 0;
                    pdcch.DMRSScramblingID = 1;
                    pdcch.DMRSPower = 0;

                    cfgDLFRC.PDCCH = {pdcch};

                    % PDSCH Instances Configuration
                    pdsch = nrWavegenPDSCHConfig;
                    pdsch.Enable = true;
                    pdsch.Label = 'Full-band PDSCH sequence';
                    pdsch.Power = 0;
                    pdsch.BandwidthPartID = 1;
                    pdsch.Modulation = mcs;
                    pdsch.NumLayers = 1;
                    pdsch.MappingType = 'A';
                    pdsch.ReservedCORESET = [];
                    pdsch.SymbolAllocation = [2 12];
                    pdsch.SlotAllocation = 1:9;
                    pdsch.Period = 10;
                    pdsch.PRBSet = 0:24;
                    pdsch.VRBToPRBInterleaving = 0;
                    pdsch.VRBBundleSize = 2;
                    pdsch.NID = [];
                    pdsch.RNTI = 1;
                    pdsch.Coding = true;
                    pdsch.TargetCodeRate = 0.30078125;
                    pdsch.TBScaling = 1;
                    pdsch.XOverhead = 0;
                    pdsch.RVSequence = 0;
                    pdsch.DataSource = 'PN9';
                    pdsch.DMRSPower = 3;
                    pdsch.EnablePTRS = false;
                    pdsch.PTRSPower = 0;

                    % PDSCH Reserved PRB
                    pdschReservedPRB = nrPDSCHReservedConfig;
                    pdschReservedPRB.PRBSet = 0:2;
                    pdschReservedPRB.SymbolSet = [0 1];
                    pdschReservedPRB.Period = 1;

                    pdsch.ReservedPRB = {pdschReservedPRB};

                    % PDSCH DM-RS
                    pdschDMRS = nrPDSCHDMRSConfig;
                    pdschDMRS.DMRSConfigurationType = 1;
                    pdschDMRS.DMRSReferencePoint = 'CRB0';
                    pdschDMRS.DMRSTypeAPosition = 2;
                    pdschDMRS.DMRSAdditionalPosition = 2;
                    pdschDMRS.DMRSLength = 1;
                    pdschDMRS.CustomSymbolSet = [];
                    pdschDMRS.DMRSPortSet = [];
                    pdschDMRS.NIDNSCID = [];
                    pdschDMRS.NSCID = 0;
                    pdschDMRS.NumCDMGroupsWithoutData = 2;

                    pdsch.DMRS = pdschDMRS;

                    % PDSCH PT-RS
                    pdschPTRS = nrPDSCHPTRSConfig;
                    pdschPTRS.TimeDensity = 1;
                    pdschPTRS.FrequencyDensity = 2;
                    pdschPTRS.REOffset = '00';
                    pdschPTRS.PTRSPortSet = [];

                    pdsch.PTRS = pdschPTRS;

                    cfgDLFRC.PDSCH = {pdsch};

                    % CSI-RS Instances Configuration
                    csirs = nrWavegenCSIRSConfig;
                    csirs.Enable = false;
                    csirs.Label = 'CSIRS1';
                    csirs.Power = 0;
                    csirs.BandwidthPartID = 1;
                    csirs.CSIRSType = {'nzp'};
                    csirs.CSIRSPeriod = 'on';
                    csirs.RowNumber = 1;
                    csirs.Density = {'three'};
                    csirs.SymbolLocations = {0};
                    csirs.SubcarrierLocations = {0};
                    csirs.NumRB = 25;
                    csirs.RBOffset = 0;
                    csirs.NID = 1;

                    cfgDLFRC.CSIRS = {csirs};

                    % waveform generation
                    [waveform, info] = nrWaveformGenerator(cfgDLFRC);
                    fprintf("Generating: %s %d %d %s\n", fr, cbw, scs, mcs);
                catch
                    warning("Skipping: %s %d %d %s\n", fr, cbw, scs, mcs);
                end
            end
        end
    end
end