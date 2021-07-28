folder_path = "saved_data";
file_extension = "*.mat";
path = fullfile(folder_path, file_extension);
file_info = dir(path);
file_names = {file_info.name};

for fname = file_names
    fpath = fullfile(folder_path, fname);
    load(fpath)
    
    fr = waveStruct.config.waveform.FrequencyRange;
    mcs = strsplit(waveStruct.config.waveform.MCS,",");
	mcs = mcs{1};
    scs = waveStruct.config.waveform.SubcarrierSpacing;
    cbw = waveStruct.config.waveform.ChannelBandwidth;
    waveform = waveStruct.waveform;
    
    % introduce noise impairment to waveform
    snr = 15;
    waveform = awgn(waveform, snr, 'measured');

    % write into file
    [symbols, time] = size(waveform);
    waveform = reshape(waveform, [symbols*time, 1]);
    waveformTable = table(waveform, 'VariableNames', {'I+Qi'});
    writetable(waveformTable, outputFileName(snr, fr, mcs, scs, cbw));
    
    fprintf("Saving: %s %s %d %d\n", fr, mcs, scs, cbw);
end

function fileName = outputFileName(snr, fr, mcs, scs, cbw)
    fileName = sprintf("generated_data/nr5g_snr%d/nr5g_%s_%s_%d_%d.txt", snr, fr, mcs, scs, cbw);
end