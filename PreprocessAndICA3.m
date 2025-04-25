% --- 데이터 로드 및 기본 설정 ---
load('DREAMER.mat');

theta_band = [4, 8];
alpha_band = [8, 13];
beta_band = [13, 30];

sample_rate = 128;
feature_matrix = [];
labels_matrix = [];

eeglab;

% --- 밴드패스 필터 설정 (4 ~ 30 Hz 대역) ---
filter_order = 4 * fix(sample_rate / 2);
low_cutoff = 4 / (sample_rate / 2);
high_cutoff = 30 / (sample_rate / 2);
[b, a] = butter(filter_order, [low_cutoff, high_cutoff], 'bandpass');

% --- 데이터 전처리 및 특징 추출 (trial 단위) ---
for participant_idx = 1:23
    disp(['Processing Participant ', num2str(participant_idx), ' of 23']);
    participant = DREAMER.Data{1, participant_idx};

    all_trials = {}; 
    valid_trials = [];

    % --- 각 trial을 저장 (18개의 trial) ---
    for experiment_idx = 1:18
        disp(['  Processing trial ', num2str(experiment_idx), ' for Participant ', num2str(participant_idx)]);
        
        stimuli_data = participant.EEG.stimuli{experiment_idx, 1};

        % --- EEGLAB의 ICA를 사용해 아티팩트 제거 ---
        EEG = pop_importdata('data', stimuli_data', 'srate', sample_rate);
        EEG = pop_eegfiltnew(EEG, low_cutoff, high_cutoff);

        EEG = pop_runica(EEG, 'extended', 1);

        comps_to_remove = detect_artifact_components(EEG);

        if ~isempty(comps_to_remove)
            num_components = size(EEG.icawinv, 2);
            comps_to_remove = comps_to_remove(comps_to_remove <= num_components);
            
            remaining_components = num_components - length(comps_to_remove);
            if remaining_components > 0
                EEG = pop_subcomp(EEG, comps_to_remove, 0);
            else
                disp('Too many components selected for removal. Skipping component removal.');
            end
        else
            disp('No components to remove.');
        end

        if ~isempty(EEG.data)
            filtered_data = EEG.data';
            all_trials{experiment_idx} = filtered_data;
            valid_trials = [valid_trials, experiment_idx];
        else
            disp(['  Trial ', num2str(experiment_idx), ' contains too many artifacts and was removed.']);
        end
    end

    % --- 유효한 trial들로부터 특징 추출 ---
    for experiment_idx = valid_trials
        disp(['  Extracting features for Trial ', num2str(experiment_idx), ' for Participant ', num2str(participant_idx)]);

        filtered_data = all_trials{experiment_idx};

        % --- Welch 방법을 사용한 파워 스펙트럼 밀도 계산 ---
        log_psd_welch = compute_welch_psd(filtered_data, sample_rate, theta_band, alpha_band, beta_band);

        % --- FFT 방법을 사용한 파워 스펙트럼 계산 ---
        log_psd_fft = compute_fft_psd(filtered_data, sample_rate, theta_band, alpha_band, beta_band);

        feature_vector = [reshape(log_psd_welch, 1, []), reshape(log_psd_fft, 1, [])];

        feature_matrix = [feature_matrix; feature_vector];

        valence = participant.ScoreValence(experiment_idx);
        arousal = participant.ScoreArousal(experiment_idx);
        dominance = participant.ScoreDominance(experiment_idx);
        labels_matrix = [labels_matrix; [valence, arousal, dominance]];
    end
    
    disp(['Remaining valid trials for Participant ', num2str(participant_idx), ': ', num2str(valid_trials)]);
end


% --- Welch’s method를 통한 PSD 계산 함수 ---
function log_psd = compute_welch_psd(eeg_data, sample_rate, theta_band, alpha_band, beta_band)
    num_channels = size(eeg_data, 2);
    log_psd = zeros(3, num_channels);

    for channel_idx = 1:num_channels
        [pxx, f] = pwelch(eeg_data(:, channel_idx), sample_rate, [], [], sample_rate);
        log_psd(1, channel_idx) = log_band_power(pxx, f, theta_band);
        log_psd(2, channel_idx) = log_band_power(pxx, f, alpha_band);
        log_psd(3, channel_idx) = log_band_power(pxx, f, beta_band);
    end
end

% --- FFT를 통한 PSD 계산 함수 ---
function log_psd = compute_fft_psd(eeg_data, sample_rate, theta_band, alpha_band, beta_band)
    num_channels = size(eeg_data, 2);
    log_psd = zeros(3, num_channels);

    for channel_idx = 1:num_channels
        nfft = 2^nextpow2(size(eeg_data, 1));
        fft_data = fft(eeg_data(:, channel_idx), nfft);
        f = sample_rate / 2 * linspace(0, 1, nfft / 2 + 1);
        pxx = abs(fft_data(1:nfft/2+1)).^2 / sample_rate;
        log_psd(1, channel_idx) = log_band_power(pxx, f, theta_band);
        log_psd(2, channel_idx) = log_band_power(pxx, f, alpha_band);
        log_psd(3, channel_idx) = log_band_power(pxx, f, beta_band);
    end
end

% --- 특정 주파수 대역의 로그 파워 계산 함수 ---
function log_power = log_band_power(pxx, f, band)
    band_idx = find(f >= band(1) & f <= band(2));
    band_power = sum(pxx(band_idx));
    log_power = log(band_power);
end

% --- 아티팩트 컴포넌트 자동 감지 함수 ---
function comps_to_remove = detect_artifact_components(EEG)
    blink_thresh = 0.8;
    blink_comps = find(max(abs(EEG.icawinv(1:2, :))) > blink_thresh);
    muscle_thresh = 20;
    muscle_comps = find(mean(abs(EEG.icawinv(3:end, :))) > muscle_thresh);
    comps_to_remove = unique([blink_comps; muscle_comps]);
    num_components = size(EEG.icawinv, 2);
    comps_to_remove = comps_to_remove(comps_to_remove <= num_components);
    if isempty(comps_to_remove)
        disp('No components to remove.');
    end
end
