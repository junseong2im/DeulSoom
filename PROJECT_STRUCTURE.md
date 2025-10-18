# DeulSoom - 프로젝트 구조 가이드

## 빠른 시작 가이드

### 새로운 팀원을 위한 3분 가이드
1. **학습하려면**: `scripts/training/` 확인
2. **API 수정하려면**: `app/` 또는 `fragrance_ai/api/` 확인
3. **테스트 돌리려면**: `tests/` 에서 실행
4. **배포하려면**: `docs/deployment/` 읽고 `scripts/deployment/` 실행
5. **문제 해결하려면**: `docs/operations/RUNBOOK.md` 참조

---

## 전체 디렉토리 구조

```
DeulSoom/
├── scripts/              # 실행 스크립트 (학습, 설정, 배포)
├── tests/                # 테스트 코드 (단위, 통합, 성능)
├── docs/                 # 문서 (개발, 배포, 운영)
├── data/                 # 데이터 (DB, 테스트 데이터)
├── models/               # 모델 체크포인트
├── docker/               # Docker 설정
├── services/             # 백그라운드 서비스
├── logs/                 # 로그 파일
│
├── app/                  # FastAPI 프로덕션 앱
├── fragrance_ai/         # 핵심 비즈니스 로직
├── archive/              # 레거시 코드 보관
│
└── (설정 파일들)         # setup.py, pyproject.toml, etc.
```

---

## 1. scripts/ - 실행 스크립트

모든 실행 가능한 스크립트가 모여있는 곳입니다.

### scripts/training/ - AI 모델 학습
```
train_universal_model.py     # UniversalFragranceGenerator 학습
train_ppo_real.py            # PPO 강화학습
train_real_fragrance.py      # 향수 생성 모델 학습
train_validator.py           # 검증 모델 학습
```

**사용법**:
```bash
cd scripts/training
python train_universal_model.py --batch_size 16 --max_epochs 100
```

### scripts/setup/ - 초기 설정
```
setup_database.py            # 데이터베이스 초기화
load_vector_data.py          # 벡터 데이터 로드
```

**사용법**:
```bash
cd scripts/setup
python setup_database.py
python load_vector_data.py
```

### scripts/validation/ - 검증
```
verify_rlhf.py              # RLHF 검증
```

### scripts/deployment/ - 배포
```
deploy.sh                   # 배포 스크립트
setup-ssl.sh                # SSL 설정
```

**사용법**:
```bash
cd scripts/deployment
bash deploy.sh production
```

### scripts/ops/ - 운영
```
CLEANUP_SCRIPT.sh           # 정리 스크립트
```

---

## 2. tests/ - 테스트 코드

모든 테스트가 체계적으로 분류되어 있습니다.

```
tests/
├── integration/            # 통합 테스트
│   ├── test_integration_complete.py
│   └── test_rlhf_complete.py
├── performance/            # 성능 테스트
│   ├── run_performance_tests.py
│   └── test_metrics_server.py
├── rl/                     # 강화학습 테스트
│   ├── test_rl_smoke.py
│   ├── test_rl_smoke_new.py
│   ├── test_moga_real.py
│   └── test_moga_stability.py
├── gpu/                    # GPU 테스트
│   └── test_qwen_gpu.py
├── html/                   # HTML 테스트 파일
│   ├── ai_test.html
│   ├── chat.html
│   └── fragrance_test.html
├── run_tests.py            # 전체 테스트 실행
├── run_exception_tests.py  # 예외 처리 테스트
├── run_tests.sh            # 테스트 쉘 스크립트
└── smoke_test_api.sh       # API 스모크 테스트
```

**사용법**:
```bash
cd tests
python run_tests.py                    # 전체 테스트
python integration/test_integration_complete.py  # 통합 테스트만
python performance/run_performance_tests.py      # 성능 테스트만
```

---

## 3. docs/ - 문서

모든 문서가 목적별로 분류되어 있습니다.

```
docs/
├── development/            # 개발 관련 문서 (14개)
│   ├── DEVELOPMENT_STATUS.md
│   ├── IMPLEMENTATION_RECORD.md
│   ├── PROJECT_CLEANUP_ANALYSIS.md
│   └── ...
├── deployment/             # 배포 가이드 (9개)
│   ├── DEPLOYMENT_GUIDE.md
│   ├── RELEASE_STRATEGY.md
│   ├── ROLLBACK_OPERATIONS_SUMMARY.md
│   └── ...
├── operations/             # 운영 가이드 (15개)
│   ├── RUNBOOK.md                    # 운영 매뉴얼 (시작점)
│   ├── OPERATIONS_GUIDE.md
│   ├── PRODUCTION_CHECKLIST.md
│   └── ...
└── archive/                # 구버전 문서 (9개)
    ├── GO_NOGO_GATE_GUIDE.md
    └── ...
```

**읽는 순서**:
1. 신입 개발자: `docs/development/DEVELOPMENT_STATUS.md`
2. 배포 담당자: `docs/deployment/DEPLOYMENT_GUIDE.md`
3. 운영 담당자: `docs/operations/RUNBOOK.md`

---

## 4. data/ - 데이터

```
data/
├── databases/              # 데이터베이스 파일
│   ├── fragrance.db
│   ├── fragrance_ai.db
│   └── fragrance_db.sqlite
└── test/                   # 테스트 데이터
    ├── test_dna_fast.json
    ├── test_feedback.json
    ├── test_ppo.json
    └── test_reinforce.json
```

---

## 5. models/ - 모델 체크포인트

```
models/
└── checkpoints/            # 학습된 모델 체크포인트
    ├── best_model.pth
    ├── fragrance_model_20250924_222018.pth
    ├── rlhf_checkpoint_ep10.pth
    └── ... (총 11개)
```

**사용법**:
- 최신 모델: `best_model.pth`
- 특정 버전: 날짜 포함된 파일명 참조

---

## 6. docker/ - Docker 설정

```
docker/
├── compose/                # docker-compose 파일
│   ├── docker-compose.yml           # 개발용
│   ├── docker-compose.production.yml  # 프로덕션
│   ├── docker-compose.scale.yml      # 스케일링
│   └── ... (총 8개)
└── dockerfiles/            # Dockerfile
    ├── Dockerfile
    ├── Dockerfile.production
    └── Dockerfile.cloud
```

**사용법**:
```bash
# 개발 환경
docker-compose -f docker/compose/docker-compose.yml up

# 프로덕션 환경
docker-compose -f docker/compose/docker-compose.production.yml up -d
```

---

## 7. app/ - FastAPI 프로덕션 앱

프로덕션 환경에서 실행되는 메인 애플리케이션입니다.

```
app/
├── main.py                 # FastAPI 앱 엔트리포인트
├── routers/                # API 라우터
├── schemas/                # Pydantic 스키마
└── ...
```

---

## 8. fragrance_ai/ - 핵심 비즈니스 로직

프로젝트의 핵심 코드가 들어있습니다.

```
fragrance_ai/
├── api/                    # API 레이어
│   ├── main.py            # API 메인
│   └── routes/            # API 엔드포인트
├── core/                   # 핵심 설정
│   ├── config.py          # 설정 관리
│   └── auth.py            # 인증
├── models/                 # AI 모델
│   ├── deep_learning_architecture.py  # UniversalFragranceGenerator
│   ├── conversation_llm.py            # 대화형 LLM
│   ├── advanced_generator.py          # LoRA 생성기
│   ├── embedding.py                   # 임베딩
│   └── rag_system.py                  # RAG
├── database/               # 데이터베이스
│   ├── models.py          # SQLAlchemy 모델
│   ├── connection.py      # DB 연결
│   └── schema.py          # DB 스키마
├── services/               # 비즈니스 로직
└── training/               # 학습 로직
```

---

## 9. services/ - 백그라운드 서비스

```
services/
└── ai_service.py           # AI 서비스
```

---

## 10. archive/ - 레거시 코드

사용하지 않지만 참고용으로 보관하는 코드입니다.

```
archive/
├── removed_files/          # 제거된 파일
├── legacy_apis/            # 레거시 API
├── data_collection/        # 데이터 수집 스크립트
└── tests_legacy/           # 레거시 테스트
```

---

## 일반적인 작업 시나리오

### 새 기능 추가
1. `fragrance_ai/models/` 에 모델 추가
2. `app/routers/` 에 API 엔드포인트 추가
3. `tests/` 에 테스트 추가
4. `docs/development/` 에 문서 작성

### 모델 학습
1. `scripts/training/` 에서 학습 스크립트 실행
2. 학습된 모델은 자동으로 `models/checkpoints/` 에 저장
3. TensorBoard 로그는 `runs/` 에 저장

### 배포
1. `docs/deployment/DEPLOYMENT_GUIDE.md` 읽기
2. `scripts/deployment/deploy.sh` 실행
3. `docs/operations/PRODUCTION_CHECKLIST.md` 확인

### 문제 해결
1. `logs/` 에서 로그 확인
2. `docs/operations/RUNBOOK.md` 참조
3. 테스트: `tests/run_tests.py` 실행

---

## 코딩 컨벤션

### 파일명
- Python: `snake_case.py`
- 클래스: `PascalCase`
- 함수/변수: `snake_case`
- 상수: `UPPER_SNAKE_CASE`

### 디렉토리 배치
- 실행 스크립트 -> `scripts/`
- 테스트 코드 -> `tests/`
- 비즈니스 로직 -> `fragrance_ai/`
- API 엔드포인트 -> `app/routers/` 또는 `fragrance_ai/api/routes/`

### 문서화
- 모든 함수에 docstring 작성
- 타입 힌트 필수
- 복잡한 로직은 주석 추가

---

## Git 워크플로우

### 브랜치 전략
- `master`: 프로덕션 코드
- `develop`: 개발 중인 코드
- `feature/*`: 새 기능
- `hotfix/*`: 긴급 수정

### 커밋 메시지
```
type: Subject

Body (optional)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**타입**:
- `feat`: 새 기능
- `fix`: 버그 수정
- `refactor`: 리팩토링
- `docs`: 문서
- `test`: 테스트
- `chore`: 기타

---

## 핵심 의존성

### Python 패키지
- FastAPI: 웹 프레임워크
- SQLAlchemy: ORM
- PyTorch: 딥러닝
- Transformers: LLM
- Sentence-BERT: 임베딩
- Pydantic: 데이터 검증

### 인프라
- PostgreSQL: 메인 데이터베이스
- SQLite: 개발 데이터베이스
- Redis: 캐시
- Nginx: 리버스 프록시
- Docker: 컨테이너화

---

## 문제 해결 체크리스트

### API 오류
1. `logs/api_server.log` 확인
2. `tests/smoke_test_api.sh` 실행
3. `docs/operations/RUNBOOK.md` 참조

### 데이터베이스 오류
1. `data/databases/` 확인
2. `scripts/setup/setup_database.py` 재실행
3. Alembic 마이그레이션 확인

### 모델 오류
1. `models/checkpoints/` 에 모델 파일 존재 확인
2. `scripts/training/` 에서 재학습
3. `tests/integration/` 테스트 실행

---

## 연락처 및 리소스

### 문서
- 개발: `docs/development/`
- 배포: `docs/deployment/`
- 운영: `docs/operations/`

### 중요한 파일
- 운영 매뉴얼: `docs/operations/RUNBOOK.md`
- 배포 가이드: `docs/deployment/DEPLOYMENT_GUIDE.md`
- 프로젝트 가이드: `PROJECT_GUIDE.md`

---

**마지막 업데이트**: 2025-10-19
**프로젝트 상태**: 활성 개발 중
