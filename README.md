# EEG 기반 감정 예측 모델 및 색채심리학 검증

> 기간: 2024년  
> 주제: 뇌파 데이터 기반 감정 예측 모델 개발 및 색채심리 이론 정량 검증

## 개요

본 프로젝트는 EEG(뇌파) 데이터를 이용하여 인간의 감정 상태(Valence, Arousal, Dominance)를 예측하고,  
이를 통해 색채심리학적 이론을 정량적으로 검증할 수 있는 기반을 마련하고자 수행되었습니다.  
감정 예측 모델은 Random Forest와 Support Vector Machine(SVM) 알고리즘을 활용하여 개발되었습니다.

※ 본 탐구에서는 색채 자극 기반 뇌파 데이터셋을 확보하지 못하여, 감정 예측 모델 구축까지만 완료하였습니다.  
색채 자극과 감정 간의 상관관계 분석은 향후 연구 과제로 설정하였습니다.

## 사용 데이터셋

- **DREAMER Dataset**
  - 23명의 참가자가 감정 점수(Valence, Arousal, Dominance)를 자가 평가한 EEG 및 ECG 데이터
  - [DREAMER 데이터셋 링크](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions)

## 탐구 과정

### 1. 데이터 전처리

- Bandpass 필터링 (4–30 Hz) 적용
- ICA(Independent Component Analysis)를 통한 아티팩트 제거
- DREAMER 데이터셋의 23명의 참가자 EEG 데이터에 대해 trial 단위로 전처리 수행

### 2. 특징 추출

- Welch's method 및 FFT를 사용하여 EEG 신호의 Power Spectral Density(PSD) 계산
- theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz) 주파수 대역별 로그 파워 추출
- 최종적으로 각 trial에 대해 특징 벡터를 구성

### 3. 모델 학습 및 평가

- 분류 모델: Random Forest, Support Vector Machine (SVM)
- 예측 대상: Valence, Arousal, Dominance 감정 점수 범주화
- 교차 검증을 통한 모델 평가 수행

### 4. 분석 및 결과

- Random Forest 모델이 SVM 모델보다 전반적으로 높은 정확도를 기록하였습니다.
- 특히 theta, alpha, beta 밴드 파워가 감정 구분에 중요한 역할을 하는 것으로 분석되었습니다.
- EEG 데이터의 특징만으로도 감정 상태를 일정 수준 이상 예측할 수 있음을 확인하였습니다.

## 결론 및 한계

- 본 탐구를 통해 뇌파 기반 감정 예측이 가능함을 확인하였으며, 색채 자극과 감정 반응의 정량적 분석 가능성도 모색하였습니다.
- 그러나 색채 자극 기반 EEG 데이터셋이 확보되지 않아, 실제 색채 자극-감정 매핑 검증은 수행하지 못하였습니다.
- 향후 연구에서는 색채 자극을 제시한 상태에서 EEG 데이터를 수집하고, 감정 상태 예측 모델과 연계하여 색채심리학 이론을 검증할 계획입니다.

## 참고
- DREAMER Dataset
- EEGLAB Toolbox
