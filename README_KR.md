# PyTorch Template

[English](README.md) | [한글](README_KR.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-integrated-blue.svg)](https://optuna.org/)
[![W&B](https://img.shields.io/badge/Weights_%26_Biases-supported-FFBE00.svg)](https://wandb.ai/)

PyTorch 기반 딥러닝 연구를 위한 모듈형 확장 가능 템플릿입니다. 모델, 옵티마이저, 스케줄러, 손실 함수, 콜백 등 실험의 모든 요소를 하나의 YAML 파일에 정의하고 단일 명령어로 실행할 수 있습니다.

## 주요 기능

- **YAML 기반 설정** — 모든 실험 설정을 YAML로 관리합니다. 불변(frozen) 및 검증된 설정으로 잘못된 구성을 방지합니다.
- **콜백 기반 학습** — 우선순위 기반 콜백을 통한 확장 가능한 학습 루프를 제공합니다. 핵심 코드를 수정하지 않고 로깅, 체크포인트, 조기 종료 등의 동작을 추가할 수 있습니다.
- **설정 가능한 손실 함수 및 지표** — YAML을 통해 손실 함수를 교체할 수 있습니다 (`torch.nn.CrossEntropyLoss`, 사용자 정의 손실 함수 등). 내장 지표 레지스트리(MSE, MAE, R2)를 제공하며, importlib을 통한 확장이 가능합니다.
- **체크포인트 및 재개** — 모델, 옵티마이저, 스케줄러, RNG 상태의 완전한 상태 체크포인트를 지원하며, `SeedManifest`를 통한 다중 시드 재개가 가능합니다.
- **실행 출처 추적** — 실행마다 Python/PyTorch/CUDA 버전, GPU 정보, git 커밋, 환경 변수를 자동으로 기록합니다.
- **하이퍼파라미터 최적화** — 사용자 정의 PFL 프루너와 심층 병합 설정 오버라이드를 갖춘 Optuna 통합을 제공합니다.
- **실험 추적** — 콜백을 통한 Weights & Biases 로깅을 원활하게 지원합니다.
- **CLI** — `typer` 기반 CLI로 `train`, `validate`, `preview`, `doctor`, `analyze` 하위 명령어를 제공합니다.
- **재현성** — Python, NumPy, PyTorch 전반에 걸친 결정론적 시드 관리를 지원합니다.

## 콜백 아키텍처

학습 루프는 정의된 훅 지점에서 이벤트를 발생시킵니다. 각 관심사(로깅, 조기 종료, 체크포인트)는 독립적이며 우선순위가 정해진 콜백입니다:

![Callback Execution Flow](assets/callback_flow.png)

| 콜백 | 우선순위 | 훅 | 목적 |
|------|---------|-----|------|
| `NaNDetectionCallback` | 5 | `on_epoch_end` | NaN 손실 감지 및 중단 신호 |
| `OptimizerModeCallback` | 10 | `on_train_epoch_begin`, `on_val_begin` | SPlus/ScheduleFree 학습/평가 모드 전환 |
| `LossPredictionCallback` | 70 | `on_val_end` | 조기 프루닝을 위한 최종 손실 예측 |
| `WandbLoggingCallback` | 80 | `on_epoch_end` | W&B에 지표 로깅 |
| `PrunerCallback` | 85 | `on_val_end` | Optuna 프루너에 보고 |
| `EarlyStoppingCallback` | 90 | `on_val_end` | 지표 모니터링 및 중단 신호 |
| `CheckpointCallback` | 95 | `on_epoch_end` | 주기적/최적 체크포인트 저장 |

사용자 정의 동작을 추가하려면 `TrainingCallback`을 상속하고 콜백 리스트에 추가하기만 하면 됩니다. 학습 루프를 전혀 수정할 필요가 없습니다.

## 빠른 시작

1.  **클론:**
    ```bash
    git clone https://github.com/<your-username>/<your-repo>.git
    cd <your-repo>
    ```

2.  **의존성 설치** ([uv](https://github.com/astral-sh/uv) 권장):
    ```bash
    uv venv && source .venv/bin/activate
    uv pip install -U torch wandb rich beaupy numpy optuna matplotlib scienceplots typer tqdm pyyaml pytorch-optimizer pytorch-scheduler
    ```

3.  **환경 검증:**
    ```bash
    python -m cli doctor
    ```

4.  **설정 미리보기** (학습 없이 설정만 확인):
    ```bash
    python -m cli preview configs/run_template.yaml
    ```

5.  **학습:**
    ```bash
    python -m cli train configs/run_template.yaml
    # 또는 디바이스 지정:
    python -m cli train configs/run_template.yaml --device cpu
    ```

6.  **하이퍼파라미터 최적화:**
    ```bash
    python -m cli train configs/run_template.yaml --optimize-config configs/optimize_template.yaml
    ```

7.  **결과 분석:**
    ```bash
    python -m cli analyze
    # 또는 비대화형 모드:
    python -m cli analyze --project MyProject --group MyGroup --seed 42
    ```

> **레거시 CLI**: `python main.py --run_config configs/run_template.yaml` 명령어도 하위 호환성을 위해 계속 사용 가능합니다.

## 프로젝트 구조

```
pytorch_template/
├── cli.py                 # Typer CLI 진입점 (train, validate, preview, doctor, analyze)
├── main.py                # 레거시 argparse 진입점
├── config.py              # RunConfig (불변, 검증됨) + OptimizeConfig
├── util.py                # Trainer, run(), 데이터 로딩, 분석 헬퍼
├── callbacks.py           # 콜백 시스템 (8개 내장 콜백 + CallbackRunner)
├── metrics.py             # 지표 레지스트리 (MSE, MAE, R2 + importlib 확장)
├── checkpoint.py          # CheckpointManager + SeedManifest
├── provenance.py          # 환경 캡처 + 설정 해싱
├── model.py               # 모델 아키텍처 (MLP)
├── pruner.py              # Optuna용 PFL 프루너
├── configs/
│   ├── run_template.yaml
│   └── optimize_template.yaml
├── recipes/
│   ├── regression/        # 사인파 회귀 (MLP + MSELoss)
│   └── classification/    # FashionMNIST 분류 (CNN + CrossEntropyLoss)
├── tests/                 # 36개 단위 테스트
└── runs/                  # 실험 출력 (자동 생성)
```

## 설정

### 실행 설정 (`run_template.yaml`)

```yaml
project: PyTorch_Template
device: cuda:0
net: model.MLP
optimizer: pytorch_optimizer.SPlus
scheduler: pytorch_scheduler.ExpHyperbolicLRScheduler
criterion: torch.nn.MSELoss          # importlib을 통한 임의의 손실 함수
criterion_config: {}                  # criterion 생성자에 전달할 인자
epochs: 50
batch_size: 256
seeds: [89, 231, 928, 814, 269]
net_config:
  nodes: 64
  layers: 4
optimizer_config:
  lr: 1.e-3
  eps: 1.e-10
scheduler_config:
  total_steps: 50
  upper_bound: 250
  min_lr: 1.e-5
early_stopping_config:
  enabled: false
  patience: 10
  mode: min
  min_delta: 0.0001
checkpoint_config:
  enabled: false
  save_every_n_epochs: 10
  keep_last_k: 3
  save_best: true
  monitor: val_loss
  mode: min
```

**주요 필드:**

| 필드 | 설명 |
|------|------|
| `net` | `module.Class` 형식의 모델 클래스 경로 |
| `optimizer` | 옵티마이저 클래스 경로 (`torch.optim.*`, `pytorch_optimizer.*`, 사용자 정의 지원) |
| `scheduler` | 스케줄러 클래스 경로 (`torch.optim.lr_scheduler.*`, `pytorch_scheduler.*`, 사용자 정의 지원) |
| `criterion` | 손실 함수 클래스 경로 (예: `torch.nn.MSELoss`, `torch.nn.CrossEntropyLoss`) |
| `criterion_config` | criterion 생성자에 전달할 인자 |
| `seeds` | 랜덤 시드 목록 — 각 시드는 별도의 학습 실행에 해당 |
| `checkpoint_config` | 설정 가능한 정책을 갖춘 주기적/최적 체크포인트 저장 |

모든 모듈 경로는 런타임에 `importlib`을 통해 해석됩니다. 설정은 생성 후 **불변(frozen)** 상태가 됩니다. 수정된 복사본을 생성하려면 `config.with_overrides(field=value)`를 사용하십시오.

### 최적화 설정

전체 템플릿은 [`configs/optimize_template.yaml`](configs/optimize_template.yaml)을 참조하십시오. 주요 섹션: `search_space`, `sampler`, `pruner`.

## 커스터마이징

### 사용자 정의 모델 추가

`model.py` 또는 새 파일에 모델 클래스를 생성합니다. 생성자는 `(hparams: dict, device: str)`를 인자로 받아야 합니다:

```python
# my_model.py
class MyTransformer(nn.Module):
    def __init__(self, hparams, device="cpu"):
        super().__init__()
        # hparams는 YAML의 net_config에서 전달됩니다
        ...
```

```yaml
net: my_model.MyTransformer
net_config:
  d_model: 256
  nhead: 8
```

### 사용자 정의 콜백 추가

`TrainingCallback`을 상속하고 훅 메서드를 오버라이드합니다:

```python
# my_callbacks.py
from callbacks import TrainingCallback

class GradientClipCallback(TrainingCallback):
    priority = 15  # OptimizerMode 이후 초기에 실행

    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def on_train_step_end(self, trainer, batch_idx, loss, **kwargs):
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), self.max_norm)
```

그런 다음 학습 스크립트의 콜백 리스트에 추가하거나 `run()`을 확장합니다.

### 사용자 정의 지표 추가

내장 이름 또는 importlib 경로를 등록합니다:

```python
from metrics import MetricRegistry

registry = MetricRegistry(["mse", "mae", "r2", "my_module.MyCustomMetric"])
results = registry.compute(y_pred, y_true)
# {"mse": 0.012, "mae": 0.089, "r2": 0.95, "my_custom_metric": ...}
```

### 손실 함수 교체

YAML에서 한 줄만 변경하면 됩니다. 코드 수정은 필요하지 않습니다:

```yaml
# 회귀
criterion: torch.nn.MSELoss

# 분류
criterion: torch.nn.CrossEntropyLoss

# 사용자 정의
criterion: my_losses.FocalLoss
criterion_config:
  gamma: 2.0
  alpha: 0.25
```

### 다른 스케줄러 사용

```yaml
# PyTorch 내장
scheduler: torch.optim.lr_scheduler.CosineAnnealingLR
scheduler_config:
  T_max: 50
  eta_min: 1.e-5

# ExpHyperbolicLR (pytorch-scheduler 패키지)
scheduler: pytorch_scheduler.ExpHyperbolicLRScheduler
scheduler_config:
  total_steps: 50
  upper_bound: 250
  min_lr: 1.e-5
```

### 데이터 로딩 커스터마이징

`util.py`의 `load_data()`를 수정하여 `(train_dataset, val_dataset)`를 반환하도록 합니다. 예시(회귀 + 분류)는 [`recipes/`](recipes/)를 참조하십시오.

## 예제 레시피

| 레시피 | 과제 | 모델 | 손실 함수 | 설정 |
|--------|------|------|----------|------|
| [`recipes/regression/`](recipes/regression/) | 사인파 피팅 | MLP (64 노드, 4 레이어) | MSELoss | [config.yaml](recipes/regression/config.yaml) |
| [`recipes/classification/`](recipes/classification/) | FashionMNIST | SimpleCNN (32 채널) | CrossEntropyLoss | [config.yaml](recipes/classification/config.yaml) |

```bash
python -m cli train recipes/regression/config.yaml --device cpu
```

## CLI 참조

| 명령어 | 설명 |
|--------|------|
| `python -m cli train <config> [--device] [--optimize-config]` | 선택적 HPO를 포함한 모델 학습 |
| `python -m cli validate <config>` | 학습 없이 설정 검증 |
| `python -m cli preview <config>` | 모델 아키텍처 및 설정 요약 표시 |
| `python -m cli doctor` | Python, PyTorch, CUDA, W&B, 패키지 확인 |
| `python -m cli analyze [--project] [--group] [--seed] [--device]` | 학습된 모델 분석 |

## 문서

구성 요소 및 커스터마이징에 대한 심층 내용은 아래를 참조하십시오:

* **[프로젝트 문서](https://axect.github.io/pytorch_template)** ([Tutorial-Codebase-Knowledge](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)로 생성)

## 기여

기여를 환영합니다! Pull Request를 자유롭게 제출해 주십시오.

## 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다.

## 감사의 글

- [pytorch-optimizer](https://github.com/kozistr/pytorch_optimizer) - SPlus를 포함한 옵티마이저 컬렉션.
- [pytorch-scheduler](https://github.com/Axect/pytorch_scheduler) - ExpHyperbolicLR를 포함한 학습률 스케줄러 컬렉션.

## 부록

<details>
<summary><strong>PFL (Predicted Final Loss) 프루너</strong></summary>

### 개요

PFL 프루너(`pruner.PFLPruner`)는 초기 단계의 지표를 기반으로 학습 실행의 최종 성능을 예측하여, 가능성이 낮은 Optuna 트라이얼을 조기에 가지치기합니다.

### 설정

```yaml
pruner:
  name: pruner.PFLPruner
  kwargs:
    n_startup_trials: 10
    n_warmup_epochs: 10
    top_k: 10
    target_epoch: 50
```

### 동작 원리

1. 처음 `n_startup_trials`개의 트라이얼은 기준 성능을 확립하기 위해 끝까지 실행됩니다.
2. 이후 트라이얼에서는 `n_warmup_epochs` 이후에만 프루닝을 고려합니다.
3. 프루너는 지수 곡선 피팅을 사용하여 현재 손실 이력으로부터 최종 손실을 예측합니다.
4. 예측된 최종 손실이 상위 k개의 완료된 트라이얼보다 나쁜 경우, 해당 트라이얼이 프루닝됩니다.
5. 시드 간 지표를 평균하여 다중 시드 실행을 지원합니다.

</details>
