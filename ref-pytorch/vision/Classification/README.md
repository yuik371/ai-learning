# Image Classification - Technical Study Notes
주제: Hyperparameter

### Hyperparameter란?
- 딥러닝 모델의 학습 과정을 제어하기 위해 학습 전에 사용자가 설정하는 값임.
    - 여기서, 모델은 입력 데이터로부터 특정 출력 결과를 예측하도록 설계된 수학적 구조

| 구분    | Parameter        | Hyperparameter            |
| ----- | ---------------- | ------------------------- |
| 의미    | 모델이 학습하는 값       | 학습 전에 설정하는 값              |
| 결정 주체 | 학습 알고리즘          | 사용자 또는 탐색 알고리즘            |
| 변경 시점 | 학습 과정에서 지속적으로 변경 | 실험 설정에 따라 변경              |
| 대표 사례 | Weight, Bias     | Learning Rate, Batch Size |

Note 1 | Parameter and Hyperparameter
이미지 분류 모델이 고양이와 강아지를 구분한다고 가정하자. Parameter 는 모델이 학습 데이터를 통해 자동으로 학습하고 고양이와 강아지를 구분하는 데 필요한 가중치(weight)와 편향(bias) 등의 값이고, 
Hyperparameter는 모델의 학습 방식과 과정을 제어하기 위해 사용자가 설정하는 학습률(Learning Rate), 배치 크기(Batch Size)등의 값이다. 

## 1.2 Hyperparameter Interdependence란?

하이퍼파라미터 상호의존성은 하나의 하이퍼파라미터가 변경되었을 때 다른 하이퍼파라미터의 적절한 설정이나 학습 결과에도 영향을 미치는 관계를 의미합니다.

딥러닝에서는 각 하이퍼파라미터가 독립적으로 작동하지 않습니다.

예를 들어 Batch Size를 증가시키면 Gradient의 변동성이 달라질 수 있으며, 이에 따라 Learning Rate를 재조정해야 할 수도 있습니다.

주요 상호의존성 구조

Batch Size

한 번에 처리하는 데이터 수

Gradient Noise

기울기 추정의 변동성

Learning Rate

매개변수 업데이트 크기

Training Dynamics

학습 속도, 안정성, 수렴 특성

## 1.3 Batch Size (배치 크기)

Batch Size는 한 번의 학습 단계에서 모델이 처리하는 데이터 샘플의 개수입니다.

기본 개념부터 구분해야 합니다.

- Dataset: 전체 학습 데이터
- Batch: 전체 데이터 중 한 번에 처리하는 데이터 묶음
- Iteration: 하나의 배치를 처리하고 매개변수를 업데이트하는 학습 반복
- Epoch: 전체 학습 데이터를 한 번 사용하는 과정

예를 들어 학습 데이터가 10,000장이고 Batch Size가 100이라면 다음과 같습니다.

\\[ Iterations=\frac{10,000}{100}=100 \\]

따라서 1 Epoch에는 100번의 Iteration이 발생합니다. (마지막 불완전 배치를 버리지 않는 경우)

## 1.4 Learning Rate (학습률)

Learning Rate는 모델의 Parameter를 업데이트할 때 적용하는 학습 단계의 크기입니다.

SGD(Stochastic Gradient Descent)는 손실 함수의 기울기를 이용하여 매개변수를 업데이트합니다.

\\[ \theta\_{t+1}=\theta_t-\eta\nabla\_\theta L(\theta_t) \\]

- \\(\theta\\): 모델의 매개변수
- \\(\eta\\): Learning Rate
- \\(\nabla\_\theta L\\): 매개변수에 대한 손실 함수의 기울기
- \\(L\\): Loss Function

기본적으로 알아야 할 개념

- Loss Function: 모델의 예측과 실제 정답 사이의 차이를 계산하는 함수
- Gradient: 매개변수가 변할 때 손실 함수가 변화하는 방향과 크기를 나타내는 값
- Gradient Descent: 손실이 감소하는 방향으로 매개변수를 업데이트하는 방법

Learning Rate가 너무 크면 최적화 과정에서 발산하거나 최적점 주변에서 진동할 수 있습니다. 너무 작으면 학습이 느려질 수 있습니다.

## 1.5 Batch Size와 Learning Rate의 관계

두 하이퍼파라미터는 대표적인 상호의존 관계입니다.

Linear Scaling Rule (선형 학습률 스케일링)

배치 크기가 증가하면 학습률도 비례하여 증가시키는 경험적 규칙입니다.

\\[ \eta\_{\text{new}}= \eta\_{\text{base}} \times \frac{B\_{\text{new}}}{B\_{\text{base}}} \\]

- \\(\eta\\): Learning Rate
- \\(B\\): Global Batch Size

이 규칙은 대규모 분산 학습에서 자주 활용되지만 모든 모델과 환경에 무조건 적용되는 법칙은 아닙니다. 초기 학습 안정성을 위해 Warmup을 함께 사용하는 경우도 있습니다.

## 1.6 Gradient Noise (기울기 노이즈)

Gradient Noise는 미니배치에서 계산한 기울기가 전체 데이터에서 계산한 기울기와 차이를 보이는 현상입니다.

이러한 차이는 미니배치에 포함되는 샘플의 구성에 따라 달라집니다.

- 작은 Batch Size: 일반적으로 기울기 추정의 변동성이 큼
- 큰 Batch Size: 일반적으로 기울기 추정의 변동성이 작음

이 차이는 학습 안정성, 수렴 과정 및 일반화 성능과 관련이 있습니다.

## 1.7 Momentum (모멘텀)

Momentum은 이전 업데이트 방향을 현재 업데이트에 반영하는 최적화 기법입니다.

\\[ v_t=\mu v\_{t-1}+g_t \\]

\\[ \theta\_{t+1}=\theta_t-\eta v_t \\]

- \\(v_t\\): 누적된 업데이트 방향
- \\(\mu\\): Momentum 계수
- \\(g_t\\): 현재 기울기
- \\(\eta\\): Learning Rate

Momentum은 이전 기울기 방향을 누적하여 최적화 과정의 진동을 줄이고 진행을 가속할 수 있습니다.

## 1.8 Weight Decay (가중치 감쇠)

Weight Decay는 모델 매개변수가 지나치게 커지는 것을 억제하기 위한 정규화 기법입니다.

일반적인 SGD에서 Weight Decay는 다음과 같은 업데이트 형태로 나타납니다.

\\[ \theta\_{t+1} = \theta_t-\eta(g_t+\lambda\theta_t) \\]

- \\(\lambda\\): Weight Decay 계수
- \\(\theta\\): 모델의 매개변수
- \\(g_t\\): 손실 함수의 기울기

관련 기초 개념: Regularization (정규화)

정규화는 모델이 학습 데이터에 지나치게 적응하는 과적합(Overfitting)을 방지하고 일반화 성능을 개선하기 위한 기법입니다.

참고로 AdamW는 일반적인 Adam에 L2 페널티를 결합한 방식과 달리 Weight Decay를 기울기 기반 업데이트와 분리하여 적용합니다.

## 1.9 Learning Rate Scheduler (학습률 스케줄러)

학습 과정에서 Learning Rate를 변경하는 방법입니다.

| Scheduler         | 특징                 |
| ----------------- | ------------------ |
| StepLR            | 지정된 Epoch마다 학습률 감소 |
| MultiStepLR       | 지정한 Epoch에서 학습률 감소 |
| ExponentialLR     | 매 Epoch 일정 비율로 감소  |
| CosineAnnealingLR | 코사인 곡선 형태로 감소      |
| ReduceLROnPlateau | 검증 성능이 정체되면 학습률 감소 |

Warmup

학습 초기 Learning Rate를 점진적으로 증가시키는 전략입니다. 큰 학습률로 바로 시작했을 때 발생할 수 있는 초기 학습 불안정성을 완화하는 데 사용합니다.