<div align="center">

# PyTorch Template

[English](README.md) | [한글](README_KR.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-integrated-blue.svg)](https://optuna.org/)
[![W&B](https://img.shields.io/badge/Weights_%26_Biases-supported-FFBE00.svg)](https://wandb.ai/)

**YAML 하나. 명령어 하나. 완전한 연구 파이프라인.**

*단순한 PyTorch 보일러플레이트가 아닌 — CLI 도구와 AI 에이전트 skill을 갖춘 완전한 실험 파이프라인.*

</div>

```bash
# 1. GPU 시간을 낭비하기 전에 먼저 검증
python -m cli preflight configs/my_run.yaml --device cuda:0

# 2. 학습 (또는 HPO 실행)
python -m cli train configs/my_run.yaml --optimize-config configs/my_opt.yaml

# 3. HPO 결과 분석
python -m cli hpo-report --opt-config configs/my_opt.yaml

# 4. 최적 모델 분석
python -m cli analyze --project MyProject --group <group> --seed <seed>
```

---

## 왜 이 템플릿인가?

| 문제 | 해결책 |
|------|--------|
| 수 시간의 GPU 학습 후에야 설정 오류를 발견 | `preflight`가 수 초 만에 배치 1개의 순전파+역전파를 실행 — 학습 시작 전에 shape 불일치, 잘못된 import, NaN 그래디언트, GPU OOM을 모두 잡아냄 |
| 스크립트 전반에 흩어진 실험 설정 | 실험당 단일 frozen YAML — 실행 전 완전히 검증됨 |
| 프로젝트마다 학습 루프를 새로 작성 | 콜백 기반 루프 — 루프 자체를 수정하지 않고 확장 가능 |
| "내 머신에서는 됐는데..." | 실행마다 Python/PyTorch/CUDA/GPU/git 해시 전체 출처 기록 |
| 수동 하이퍼파라미터 튜닝 | Optuna + PFL 프루너로 가망 없는 트라이얼 조기 가지치기 |
| HPO가 끝나도 결과가 블랙박스 | `hpo-report`로 파라미터 중요도(fANOVA), 경계 경고, 상위-K 트라이얼 비교 확인 |
| 묵시적 과적합 또는 그래디언트 폭발 | `GradientMonitorCallback`과 `OverfitDetectionCallback`이 자동 감지 후 W&B에 로깅 |
| 지난달 결과를 재현할 수 없음 | 결정론적 다중 시드 학습 + 체크포인트 재개 |

---

<div align="center">

## 전체 파이프라인

```
configs/my_run.yaml          ← 모든 것을 YAML 하나에 정의
        │
        ▼
python -m cli preflight      ← 수 시간이 아닌 수 초 만에 오류 포착
        │
        ▼
python -m cli train \        ← Optuna + PFL 프루너로 HPO 실행
  --optimize-config ...
        │
        ▼
python -m cli hpo-report     ← 파라미터 중요도, 경계 경고, 상위-K 트라이얼
        │
        ▼
python -m cli train \        ← 최적 파라미터로 최종 다중 시드 학습
  configs/my_best.yaml
        │
        ▼
python -m cli analyze        ← 결과 검증 및 플롯 생성
```

</div>

**Phase 1 — 설정 생성.** YAML 하나에 모델, 옵티마이저, 스케줄러, criterion, 시드, 그리고 importlib을 통해 임의의 `load_data()` 함수를 가리키는 `data` 필드를 정의합니다. 코드 수정 없이 데이터셋 교체 가능.

**Phase 2 — Pre-flight 검사.** 전체 실행을 시작하기 전에, `preflight`가 배치 1개의 순전파+역전파 전체를 실행하고 모든 컴포넌트에 대해 pass/warn/fail을 보고합니다.

**Phase 3 — 진단 포함 학습.** 콜백 시스템이 학습 전반에 걸쳐 그래디언트 norm과 train/val 발산을 자동으로 모니터링하고, 모든 내용을 W&B에 로깅합니다.

**Phase 4 — Optuna + PFL 프루너를 활용한 HPO.** 커스텀 PFL (Predicted Final Loss) 프루너가 초기 손실 이력에 지수 감소 곡선을 피팅해, GPU 시간을 낭비하기 전에 트라이얼을 조기 가지치기합니다.

**Phase 5 — HPO 분석.** `hpo-report`가 Optuna SQLite 데이터베이스를 읽어 fANOVA 기반 파라미터 중요도, 경계 경고(탐색 공간 끝에 최적값이 있다면 범위를 넓혀야 한다는 신호), 상위-K 트라이얼 순위 표를 생성합니다.

**Phase 6 — 최종 학습.** 최적 HPO 파라미터를 `best.yaml`로 저장하고, 다중 시드로 확장해 완전히 학습합니다.

**Phase 7 — 분석.** `analyze`가 체크포인트를 인터랙티브하게 불러와 검증 세트에서 평가합니다.

---

## AI 보조 학습 (Claude Code Skill)

이 템플릿에는 전체 실험 생애주기를 안내하는 내장 [Claude Code](https://claude.ai/claude-code) skill이 포함되어 있습니다:

```
User: "FluxNet 모델 버전 0.3에 HPO 설정해줘"

Agent: configs/SolarFlux_v0.3/fluxnet_run.yaml 생성
       configs/SolarFlux_v0.3/fluxnet_opt.yaml 생성
       preflight를 실행해 설정 오류 확인
       최적 SPlus + ExpHyperbolicLR 기본값으로 HPO 실행
       hpo-report를 실행해 결과 분석
       최적 파라미터 추출 → fluxnet_best.yaml
       최종 다중 시드 학습 시작
```

이 skill에는 도메인 지식이 내장되어 있습니다: SPlus의 올바른 lr 범위(일반적인 1e-5~1e-2가 아닌 1e-3~1e+0), hyperbolic 스케줄러에서 `total_steps`를 HPO `epochs`와 동기화해서는 안 되는 이유, 그리고 경계 경고를 해석하는 방법. 일일이 외우지 않아도 연구 수준의 기본값을 바로 활용할 수 있습니다.

> 자세한 내용은 [`.claude/skills/pytorch-train/`](.claude/skills/pytorch-train/)을 참조하십시오.

### 기존 프로젝트 마이그레이션

이전 버전의 템플릿을 기반으로 한 프로젝트가 있다면, **pytorch-migrate** skill이 현재 버전을 감지하고 필요한 업데이트를 자동으로 적용합니다.

**글로벌 skill 설치** (최초 1회):

```bash
mkdir -p ~/.claude/skills
cp -r .claude/skills/pytorch-migrate ~/.claude/skills/
```

**다른 프로젝트에서 사용:**

```bash
cd ~/my-project  # pytorch_template 기반 프로젝트
# Claude Code에서:
/pytorch-migrate
```

skill이 누락된 기능(v1~v6)을 감지하고 필요한 마이그레이션만 적용하며, 사용자의 커스텀 모델, 데이터 로더, 콜백은 보존합니다.

## 문서 — 두 가지 Skill, 하나의 파이프라인

이 템플릿에는 같은 파이프라인을 가르치는 두 종류의 skill이 있습니다:

| | AI Agent Skill | Human Skill |
|---|---|---|
| **위치** | `.claude/skills/pytorch-train/` | [`docs/`](https://axect.github.io/pytorch_template) |
| **읽는 것** | 설정 규칙, 파라미터 범위, CLI 명령어 | 워크플로우 직관, 설계 결정, 트레이드오프 |
| **배우는 것** | *무엇을* 할지 | *왜* 하는지 |

**[Human Skill 가이드 읽기](https://axect.github.io/pytorch_template)** — 전체 파이프라인, 설정 심화, 콜백 시스템, HPO 전략, 커스터마이징까지 5개 챕터로 구성.

---

## 빠른 시작

```bash
git clone https://github.com/Axect/pytorch_template.git && cd pytorch_template

# 설치 (uv 권장)
uv venv && source .venv/bin/activate
uv pip install -U torch wandb rich beaupy numpy optuna matplotlib \
  scienceplots typer tqdm pyyaml pytorch-optimizer pytorch-scheduler

# 환경 확인
python -m cli doctor

# 학습 전 설정 검증
python -m cli preflight configs/run_template.yaml

# 모델 아키텍처 미리보기 (학습 없이)
python -m cli preview configs/run_template.yaml

# 학습
python -m cli train configs/run_template.yaml --device cuda:0

# 결과 분석
python -m cli analyze
```

---

## 동작 원리

### 모든 것은 YAML로

```yaml
project: MyProject
device: cuda:0
net: model.MLP                                         # importlib으로 해석 가능한 임의의 경로
optimizer: pytorch_optimizer.SPlus
scheduler: pytorch_scheduler.ExpHyperbolicLRScheduler
criterion: torch.nn.MSELoss
criterion_config: {}
data: recipes.regression.data.load_data                # 임의의 load_data() 함수를 가리킴
seeds: [89, 231, 928, 814, 269]                        # 다중 시드 재현성
epochs: 150
batch_size: 256
net_config:
  nodes: 64
  layers: 4
optimizer_config:
  lr: 1.e-1
  eps: 1.e-10
scheduler_config:
  total_steps: 150
  upper_bound: 300
  min_lr: 1.e-6
```

`data` 필드는 importlib을 통해 임의의 `load_data()` 함수를 가리킵니다. 한 줄만 바꾸면 데이터셋 교체 — `cli.py`나 `util.py` 수정 불필요.

모든 모듈 경로(모델, 옵티마이저, 스케줄러, criterion, 데이터)는 런타임에 해석됩니다. GPU 사이클이 단 한 번도 소모되기 전에 세 단계의 검증이 실행됩니다:

1. **구조적 검증** — 형식 검사, 비어 있지 않은 seeds, 양수인 epochs/batch_size, module.Class 형식
2. **런타임 검증** — CUDA 가용성, 모든 import 경로의 실제 해석 가능 여부
3. **의미론적 검증** — hyperbolic 스케줄러에서 `upper_bound >= total_steps`, lr 양수 여부, 시드 고유성, 조기 종료 patience와 epochs의 관계

### 콜백 아키텍처

학습 루프는 이벤트를 발생시키고, 각 동작은 독립적이고 우선순위가 정해진 콜백으로 처리됩니다:

![Callback Execution Flow](assets/callback_flow.png)

| 콜백 | 우선순위 | 목적 |
|------|---------|------|
| `NaNDetectionCallback` | 5 | NaN 손실 감지 및 중단 신호 |
| `OptimizerModeCallback` | 10 | SPlus/ScheduleFree 학습/평가 모드 전환 |
| `GradientMonitorCallback` | 12 | 그래디언트 폭발 감지, grad norm을 W&B에 로깅 |
| `LossPredictionCallback` | 70 | PFL 프루너를 위한 최종 손실 예측 |
| `OverfitDetectionCallback` | 75 | train/val 발산 감지, gap ratio를 W&B에 로깅 |
| `WandbLoggingCallback` | 80 | 모든 지표를 Weights & Biases에 로깅 |
| `PrunerCallback` | 85 | Optuna 프루너에 보고, TrialPruned 발생 |
| `EarlyStoppingCallback` | 90 | patience 기반 조기 종료 |
| `CheckpointCallback` | 95 | 주기적/최적 모델 체크포인트 저장 |

`TrainingCallback`을 상속하는 것만으로 사용자 정의 동작을 추가할 수 있습니다 — 학습 루프는 전혀 수정 불필요:

```python
class GradientClipCallback(TrainingCallback):
    priority = 15  # GradientMonitorCallback 바로 다음에 실행

    def on_train_step_end(self, trainer, **kwargs):
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
```

### Pre-flight 검사

모든 학습 실행 또는 HPO 작업 전에 실행하세요. 수 시간 후가 아닌 수 초 만에 문제를 잡아냅니다:

```
                         Pre-flight Check
┌─────────────────────────┬────────┬──────────────────────────────────────────┐
│ Check                   │ Status │ Detail                                   │
├─────────────────────────┼────────┼──────────────────────────────────────────┤
│ Import paths & device   │  PASS  │                                          │
│ Semantic validation     │  PASS  │                                          │
│ Object instantiation    │  PASS  │                                          │
│ Data loading            │  PASS  │ train=8000, val=2000                     │
│ Forward pass            │  PASS  │ output=(256, 1), loss=0.512341           │
│ Shape check             │  PASS  │                                          │
│ Gradient check          │  PASS  │ grad norm=0.034821                       │
│ Optimizer step          │  PASS  │                                          │
│ Scheduler step          │  PASS  │                                          │
│ GPU memory              │  PASS  │ peak=42.3 MB (1 batch)                   │
└─────────────────────────┴────────┴──────────────────────────────────────────┘
All pre-flight checks passed.
```

머신이 읽을 수 있는 출력이 필요하다면 `--json`을 사용하세요 (Claude Code skill이 자동 파싱에 활용):

```bash
python -m cli preflight configs/my_run.yaml --device cuda:0 --json
```

### HPO 분석

HPO 완료 후 `hpo-report`를 실행해 Optuna가 찾은 결과를 이해하세요 — 최적 파라미터가 무엇인지뿐 아니라, 탐색 공간이 충분히 넓었는지도 확인할 수 있습니다:

```
Study: my_study (MyProject_Opt.db)
Trials: 50 total, 38 completed, 11 pruned, 1 failed

Best Trial #23
  Value: 0.003241
  Group: MLP_n_64_l_4_SP_lr_2.3e-01_EHLS_t_150_u_300_m_1e-06
  optimizer_config_lr: 0.231
  net_config_layers: 4

         Parameter Importance
┌─────────────────────────┬──────────────────────────────────────┐
│ Parameter               │ Importance                           │
├─────────────────────────┼──────────────────────────────────────┤
│ optimizer_config_lr     │ 0.8741 ██████████████████████████    │
│ net_config_layers       │ 0.1259 ████                          │
└─────────────────────────┴──────────────────────────────────────┘

Boundary Warnings:
  optimizer_config_lr=0.231 at UPPER boundary [1e-3, 1e+0]
```

경계 경고는 옵티마이저가 더 넓은 탐색 범위에서 이득을 볼 수 있다는 의미입니다. Skill이 이를 자동으로 감지해 재실행 전에 범위를 넓히도록 안내합니다.

---

## 확장하기

<details>
<summary><strong>사용자 정의 모델</strong></summary>

```python
# my_model.py
class MyTransformer(nn.Module):
    def __init__(self, hparams: dict, device: str = "cpu"):
        super().__init__()
        # hparams는 YAML의 net_config에서 직접 전달됨
        self.d_model = hparams["d_model"]
```

```yaml
net: my_model.MyTransformer
net_config:
  d_model: 256
  nhead: 8
```

</details>

<details>
<summary><strong>사용자 정의 손실 함수</strong></summary>

```yaml
criterion: my_losses.FocalLoss
criterion_config:
  gamma: 2.0
  alpha: 0.25
```

</details>

<details>
<summary><strong>사용자 정의 지표</strong></summary>

```python
from metrics import MetricRegistry
registry = MetricRegistry(["mse", "mae", "r2", "my_module.MyMetric"])
results = registry.compute(y_pred, y_true)
```

</details>

<details>
<summary><strong>사용자 정의 데이터</strong></summary>

`(train_dataset, val_dataset)`을 반환하는 `load_data()` 함수가 있는 모듈을 생성하고, 설정에서 해당 경로를 지정하세요:

```python
# recipes/myproject/data.py
def load_data():
    # 데이터셋 구성
    return train_dataset, val_dataset
```

```yaml
data: recipes.myproject.data.load_data
```

`cli.py`나 `util.py` 수정 불필요. 완전한 예시는 [`recipes/regression/`](recipes/regression/) 및 [`recipes/classification/`](recipes/classification/)을 참조하십시오.

</details>

---

## 프로젝트 구조

```
pytorch_template/
├── cli.py              # CLI: train, preflight, validate, preview, doctor, analyze, hpo-report
├── config.py           # RunConfig (frozen, 3단계 검증) + OptimizeConfig
├── util.py             # Trainer, run(), 데이터 로딩
├── callbacks.py        # 9개 내장 콜백 + CallbackRunner
├── checkpoint.py       # CheckpointManager + SeedManifest
├── provenance.py       # 환경 캡처 + 설정 해싱
├── pruner.py           # Optuna용 PFL 프루너
├── metrics.py          # 지표 레지스트리 (MSE, MAE, R2)
├── model.py            # 내장 MLP
├── configs/            # YAML 설정 템플릿
├── recipes/            # 예제 레시피 (회귀, 분류)
├── tests/              # 단위 테스트
├── docs/               # 사람을 위한 Skill 가이드 (여러분이 읽는 문서)
└── .claude/skills/     # AI Skills (pytorch-train, pytorch-migrate)
```

---

## CLI 참조

| 명령어 | 설명 |
|--------|------|
| `python -m cli train <config> [--device DEV] [--optimize-config OPT]` | 학습 또는 HPO 실행 |
| `python -m cli preflight <config> [--device DEV] [--json]` | Pre-flight 검사: 배치 1개 순전파+역전파, shape/그래디언트/메모리 |
| `python -m cli validate <config>` | 구조적 + 런타임 설정 검증만 실행 |
| `python -m cli preview <config>` | 모델 아키텍처 및 파라미터 수 출력, 학습 없음 |
| `python -m cli doctor` | Python, PyTorch, CUDA, wandb, 필수 패키지 확인 |
| `python -m cli hpo-report [--db DB] [--opt-config OPT] [--top-k K] [--json]` | HPO 결과 분석: 파라미터 중요도, 경계 경고, 상위-K |
| `python -m cli analyze [--project P] [--group G] [--seed S] [--device DEV]` | 학습된 모델 체크포인트 평가 |

## 라이선스

[MIT](LICENSE)

## 감사의 글

- [pytorch-optimizer](https://github.com/kozistr/pytorch_optimizer) — SPlus를 포함한 옵티마이저 컬렉션
- [pytorch-scheduler](https://github.com/Axect/pytorch_scheduler) — ExpHyperbolicLR를 포함한 스케줄러 컬렉션
- [Optuna](https://optuna.org/) — 하이퍼파라미터 최적화 프레임워크
