%% Generate 802.11n/ac (OFDM) Waveform
channel_bandwidths = ["CBW20", "CBW40", "CBW80", "CBW160"];
channel_codings = ["BCC", "LDPC"];
modulation_coding_schemes = 0:1:9;
guard_intervals = ["Short", "Long"];

for cbw = channel_bandwidths
    for cc = channel_codings
        for mcs = modulation_coding_schemes
            for gi = guard_intervals
                try                    
                    % 802.11n/ac (OFDM) configuration
                    vhtCfg = wlanVHTConfig('ChannelBandwidth', cbw, ...
                        'NumUsers', 1, ...
                        'NumTransmitAntennas', 1, ...
                        'NumSpaceTimeStreams', [1], ...
                        'SpatialMapping', 'Direct', ...
                        'STBC', false, ...
                        'MCS', mcs, ...
                        'ChannelCoding', cc, ...
                        'APEPLength', 1024, ...
                        'GuardInterval', gi, ...
                        'GroupID', 63, ...
                        'PartialAID', 275);

                    % number of packets
                    numPackets = 5;
                    % input bit source
                    in = randi([0, 1], 1000, 1);

                    % waveform generation
                    waveform = wlanWaveformGenerator(in, vhtCfg, ...
                        'NumPackets', numPackets, ...
                        'IdleTime', 0, ...
                        'ScramblerInitialization', 93, ...
                        'WindowTransitionTime', 1e-07);
                    
                    % introduce noise impairment to waveform
                    snr = 15;
                    waveform = awgn(waveform, snr, 'measured');

                    % write into file
                    [symbols, time] = size(waveform);
                    waveform = reshape(waveform, [symbols*time, 1]);
                    waveformTable = table(waveform, 'VariableNames', {'I+Qi'});
                    writetable(waveformTable, outputFileName(snr, cbw, cc, mcs, gi));

                    fprintf("Generating: %s %s %s %s\n", cbw, cc, modulationCodingScheme(mcs), gi);
                catch
                    warning("Skipping: %s %s %s %s", cbw, cc, modulationCodingScheme(mcs), gi);
                end
            end
        end
    end
end

function strMcs = modulationCodingScheme(mcs)
    switch mcs
        case 0
            strMcs = "BPSK1-2";
        case 1
            strMcs = "QPSK1-2";
        case 2
            strMcs = "QPSK3-4";
        case 3
            strMcs = "16QAM1-2";
        case 4
            strMcs = "16QAM3-4";
        case 5
            strMcs = "64QAM2-3";
        case 6
            strMcs = "64QAM3-4";
        case 7
            strMcs = "64QAM5-6";
        case 8
            strMcs = "256QAM3-4";
        case 9
            strMcs = "256QAM5-6";
    end
end

function fileName = outputFileName(snr, cbw, cc, mcs, gi)
    fileName = sprintf("generated_data/wifi_snr%d/wifi_%s_%s_%s_%s.txt", snr, cbw, cc, modulationCodingScheme(mcs), gi);
end