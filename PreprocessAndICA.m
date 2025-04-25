clear all;
clc;

% --- 데이터 로드 및 기본 설정 ---
load('DREAMER.mat');

% EEG 전처리 대역 설정 (theta, alpha, beta)
theta_band = [4, 8];
alpha_band = [8, 13];
beta_band = [13, 20];

sample_rate = 128;
feature_matrix = [];
labels_matrix = [];

% --- 데이터 전처리 및 특징 추출 ---
for participant_idx = 1:length(DREAMER.Data)
    disp(['Processing Participant ', num2str(participant_idx), ' of ', num2str(length(DREAMER.Data))]);
    participant = DREAMER.Data{1, participant_idx};

    % 모든 trial 데이터를 저장할 배열 (사용자가 선택한 trial을 제거하기 위함)
    all_trials = {};

    % --- 18개의 trial을 동시에 플로팅 ---
    figure;
    for experiment_idx = 1:18
        disp(['  Plotting Trial ', num2str(experiment_idx), ' for Participant ', num2str(participant_idx)]);

        % 1. EEG 데이터 불러오기
        stimuli_data = participant.EEG.stimuli{experiment_idx, 1};

        % 모든 채널을 하나의 subplot에 플롯
        subplot(6, 3, experiment_idx);  % 6x3 grid에 플롯
        plot(linspace(0, size(stimuli_data, 1) / sample_rate, size(stimuli_data, 1)), stimuli_data);
        title(['Trial ', num2str(experiment_idx)]);
        xlabel('Time (s)');
        ylabel('Amplitude (\muV)');
        
        % 모든 trial 데이터를 배열에 저장
        all_trials{experiment_idx} = stimuli_data;
    end

    % --- 플롯을 고해상도 JPG 파일로 저장 ---
    filename = ['participant_', num2str(participant_idx), '_trials_plot.jpg'];
    print(gcf, filename, '-djpeg', '-r300');  % '-r300'은 300 DPI 해상도 설정
    disp(['Saved high-resolution plot for Participant ', num2str(participant_idx), ' as ', filename]);

    % 사용자가 제거할 trial 입력
    artifact_trials = input('Enter the artifact trial numbers to remove (e.g., [2, 4, 6]): ');

    % 선택된 trial을 제거
    for i = 1:length(artifact_trials)
        trial_to_remove = artifact_trials(i);
        disp(['  Removing Trial ', num2str(trial_to_remove), ' for Participant ', num2str(participant_idx)]);
        all_trials{trial_to_remove} = [];  % 제거된 trial을 빈 값으로 설정
    end

    % --- 제거되지 않은 trial들로 특징 추출 ---
    for experiment_idx = 1:18
        if isempty(all_trials{experiment_idx})
            continue;  % 제거된 trial은 건너뜀
        end

        disp(['  Extracting features for Trial ', num2str(experiment_idx), ' for Participant ', num2str(participant_idx)]);

        % 필터링된 데이터 사용 (필터링 부분 생략 가능)
        filtered_data = all_trials{experiment_idx};

        % --- PSD 기반 특징 추출 ---
        % 1초 단위로 데이터를 나누어 처리합니다.
        num_samples = size(filtered_data, 1);
        window_size = sample_rate;  % 1초 창
        step_size = sample_rate;    % 겹침 없음
        num_windows = floor((num_samples - window_size) / step_size) + 1;

        % 각 window마다 PSD를 계산하고 로그 변환한 후 특징 벡터로 만듭니다.
        for window_idx = 1:num_windows
            start_idx = (window_idx - 1) * step_size + 1;
            end_idx = start_idx + window_size - 1;
            window_data = filtered_data(start_idx:end_idx, :);

            % 각 채널에 대해 PSD 계산
            log_psd = compute_log_psd(window_data, sample_rate, theta_band, alpha_band, beta_band);

            % 특징 벡터를 한 줄로 펼침
            feature_vector = reshape(log_psd, 1, []);

            % 특징 벡터를 feature_matrix에 추가
            feature_matrix = [feature_matrix; feature_vector];
        end

        % 라벨 저장 (valence, arousal, dominance)
        valence = participant.ScoreValence(experiment_idx);
        arousal = participant.ScoreArousal(experiment_idx);
        dominance = participant.ScoreDominance(experiment_idx);
        labels_matrix = [labels_matrix; [valence, arousal, dominance]];
    end
end

% --- PCA 차원 축소 적용 ---
disp('Applying PCA for dimensionality reduction...');
[coeff, score, latent] = pca(feature_matrix);

% 설명된 분산 비율을 기준으로 주성분 개수를 선택
explained_variance = cumsum(latent) / sum(latent);
num_components = find(explained_variance >= 0.95, 1);  % 95% 이상의 분산을 설명하는 주성분 개수
reduced_feature_matrix = score(:, 1:num_components);

% --- 전처리된 데이터를 저장 ---
save('DREAMER_preprocessed_filtered_PCA.mat', 'reduced_feature_matrix', 'labels_matrix');
disp('Preprocessing complete with PCA and data saved.');

% --- PSD 로그 변환 계산 함수 ---
function log_psd = compute_log_psd(eeg_data, sample_rate, theta_band, alpha_band, beta_band)
    num_channels = size(eeg_data, 2);  % 채널 수 확인
    log_psd = zeros(3, num_channels);  % theta, alpha, beta 대역에 대한 결과 저장

    for channel_idx = 1:num_channels
        % Welch 방법을 사용해 Power Spectral Density (PSD) 계산
        [pxx, f] = pwelch(eeg_data(:, channel_idx), sample_rate, [], [], sample_rate);

        % 각 대역에서의 로그 파워 계산
        log_psd(1, channel_idx) = log_band_power(pxx, f, theta_band);
        log_psd(2, channel_idx) = log_band_power(pxx, f, alpha_band);
        log_psd(3, channel_idx) = log_band_power(pxx, f, beta_band);
    end
end

% --- 특정 주파수 대역의 로그 파워 계산 함수 ---
function log_power = log_band_power(pxx, f, band)
    % 해당 대역의 인덱스를 찾고 그 대역에 대한 파워 계산
    band_idx = find(f >= band(1) & f <= band(2));
    band_power = sum(pxx(band_idx));  % 해당 대역의 파워 합계
    log_power = log(band_power);      % 로그 변환된 파워 반환
end
