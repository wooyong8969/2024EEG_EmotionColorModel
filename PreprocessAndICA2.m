% --- 데이터 로드 및 기본 설정 ---
load('DREAMER.mat');  % 데이터 로드

% 주파수 대역 설정
theta_band = [4, 8];
alpha_band = [8, 13];
beta_band = [13, 30];

sample_rate = 128;  % 샘플링 레이트
feature_matrix = [];
labels_matrix = [];

% --- 밴드패스 필터 설정 (4 ~ 30 Hz 대역) ---
% 필터 계수 계산 (FIR 필터 사용)
filter_order = 4 * fix(sample_rate/2); % 필터 차수
low_cutoff = 4 / (sample_rate / 2);    % 노멀라이즈된 하한 주파수
high_cutoff = 30 / (sample_rate / 2);  % 노멀라이즈된 상한 주파수
[b, a] = butter(filter_order, [low_cutoff, high_cutoff], 'bandpass');  % 밴드패스 필터 계수

% --- 참가자별로 제거할 trial 번호 설정 ---
artifact_trials_list = {
    [6, 7, 8, 10], ...  % 1번째 참가자
    [10, 18], ...       % 2번째 참가자
    [1], ...            % 3번째 참가자
    [1, 2, 3, 4, 5, 6, 7, 8, 9], ... % 4번째 참가자
    [4, 5, 6, 7, 8, 9, 17], ...      % 5번째 참가자
    [3, 6, 10, 12, 13, 15], ...      % 6번째 참가자
    [12], ...            % 7번째 참가자
    [6, 10, 13], ...     % 8번째 참가자
    [6, 10], ...         % 9번째 참가자
    [1, 2, 4, 10, 11, 12, 13], ...  % 10번째 참가자
    [10, 12], ...        % 11번째 참가자
    [1], ...             % 12번째 참가자
    [11, 12, 13, 14, 15, 16, 17, 18], ... % 13번째 참가자
    [3, 4, 5], ...       % 14번째 참가자
    [1, 2, 3, 7, 8, 9, 16, 18], ... % 15번째 참가자
    [2, 3, 4, 13, 17, 18], ...      % 16번째 참가자
    [4, 12, 16, 17], ...            % 17번째 참가자
    [10, 12], ...        % 18번째 참가자
    [2, 3, 4, 6, 8, 9, 10, 11, 14, 17, 18], ... % 19번째 참가자
    [10, 11, 12, 13, 14, 15, 16, 17, 18], ...  % 20번째 참가자
    [3, 4, 5, 6, 7, 10, 15], ...  % 21번째 참가자
    [], ...             % 22번째 참가자
    [3, 10, 11, 12, 13, 14, 15, 16]  % 23번째 참가자
};

% --- 데이터 전처리 및 특징 추출 (trial 단위) ---
for participant_idx = 1:23
    disp(['Processing Participant ', num2str(participant_idx), ' of 23']);
    participant = DREAMER.Data{1, participant_idx};

    % 현재 참가자에 해당하는 제거할 trial 리스트
    artifact_trials = artifact_trials_list{participant_idx};

    all_trials = {}; % 모든 trial 데이터를 저장할 배열

    % --- 18개의 trial을 저장 ---
    for experiment_idx = 1:18
        % 1. EEG 데이터 불러오기 (trial 전체)
        stimuli_data = participant.EEG.stimuli{experiment_idx, 1};

        % 모든 trial 데이터를 배열에 저장
        all_trials{experiment_idx} = stimuli_data;
    end

    % --- 선택된 trial을 제거 ---
    valid_trials = setdiff(1:18, artifact_trials); % artifact_trials 제외한 유효 trial 목록

    % --- 제거되지 않은 trial들로 특징 추출 ---
    for experiment_idx = valid_trials
        disp(['  Extracting features for Trial ', num2str(experiment_idx), ' for Participant ', num2str(participant_idx)]);

        % 필터링된 데이터 사용
        filtered_data = all_trials{experiment_idx};

        % --- PSD 기반 특징 추출 (trial 단위) ---
        % trial 전체 데이터를 사용해 각 채널의 PSD를 계산하고 로그 변환한 후 특징 벡터로 변환
        log_psd = compute_log_psd(filtered_data, sample_rate, theta_band, alpha_band, beta_band);

        % 특징 벡터를 한 줄로 펼침
        feature_vector = reshape(log_psd, 1, []);  % trial 전체에서 하나의 특징 벡터 생성

        % 특징 벡터를 feature_matrix에 추가
        feature_matrix = [feature_matrix; feature_vector];

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
