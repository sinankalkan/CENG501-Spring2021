%% Generate Downlink RMC Waveform
reference_channels = ["R.0","R.1","R.2","R.3","R.4","R.5","R.6","R.7","R.8","R.9","R.10","R.11","R.12","R.13","R.14","R.25","R.26","R.27","R.28","R.31-3A","R.31-4","R.43","R.44","R.45","R.45-1","R.48","R.50","R.51","R.68-1","R.105","R.6-27RB","R.12-9RB","R.11-45RB"];
duplex_modes = ["FDD", "TDD"];
transmission_schemes = ["Port0", "TxDiversity", "SpatialMux", "CDD", "Port5", "Port7-14"];

for rc = reference_channels
    for dm = duplex_modes
        for ts = transmission_schemes
            try
                % Downlink RMC configuration
                cfg = struct('RC', rc, ...
                    'DuplexMode', dm, ...
                    'NCellID', 0, ...
                    'TotSubframes', 1, ...
                    'NumCodewords', 1, ...
                    'Windowing', 0, ...
                    'AntennaPort', 1);

                cfg.OCNGPDSCHEnable = 'Off';
                cfg.OCNGPDCCHEnable = 'Off';
                cfg.PDSCH.TxScheme = ts;
                cfg.PDSCH.RNTI = 1;
                cfg.PDSCH.Rho = 0;
                cfg.PDSCH.RVSeq = [0 1 2 3];
                cfg.PDSCH.NHARQProcesses = 8;
                cfg.PDSCH.PMISet = 1;
                cfg = lteRMCDL(cfg);

                % input bit source
                in = randi([0, 1], 1000, 1);

                % waveform generation
                [waveform, grid, config] = lteRMCDLTool(cfg, in);

                % TODO: introduce noise impairment to waveform
                snr = 15;
                waveform = awgn(waveform, snr, 'measured');

                % write into file
                rb = config.NDLRB;
                m = config.PDSCH.Modulation{1};
                [symbols, time] = size(waveform);
                waveform = reshape(waveform, [symbols*time, 1]);
                waveformTable = table(waveform, 'VariableNames', {'I+Qi'});
                writetable(waveformTable, outputFileName(snr, rc, dm, ts, m, rb));

                fprintf("Generating: %s %s %s %s %d\n", rc, dm, ts, m, rb);
            catch
                warning("Skipping: %s %s %s", rc, dm, ts);
            end
        end
    end
end

function fileName = outputFileName(snr, rc, dm, ts, m, rb)
    fileName = sprintf("generated_data/lte_snr%d/lte_%s_%s_%s_%s_%d.txt", snr, rc, dm, ts, m, rb);
end
