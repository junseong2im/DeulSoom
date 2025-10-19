# DeulSoom - AI 기반 향수 생성 시스템

## 프로젝트 개요

DeulSoom은 다목적 유전 알고리즘(MOGA)과 인간 피드백 강화학습(RLHF)을 통합한 AI 향수 생성 시스템입니다. 사용자의 요구사항과 피드백을 기반으로 최적의 향수 조합을 생성하고 지속적으로 개선합니다.

### 핵심 기능

- 자연어 기반 향수 생성 (LLM 활용)
- 다목적 최적화를 통한 향수 조합 생성
- 사용자 피드백 기반 실시간 향수 진화
- IFRA 규정 준수 자동 검증
- RESTful API 제공
- 관리자 대시보드 및 모니터링

### 기술 스택

**Backend**
- Python 3.10+
- FastAPI
- PyTorch
- PostgreSQL / SQLite
- Redis
- ChromaDB

**Frontend**
- Next.js 15
- TypeScript
- React

**AI/ML**
- Llama 3 8B
- Qwen 2.5 (7B/32B)
- Mistral 7B
- Sentence-BERT

**Infrastructure**
- Docker
- Kubernetes
- Nginx
- Grafana
- Prometheus

---

## 시스템 아키텍처

### 주요 컴포넌트

```
DeulSoom/
├── fragrance_ai/           # 핵심 AI 모듈
│   ├── api/               # REST API 엔드포인트
│   ├── models/            # AI 모델 (MOGA, RLHF, LLM)
│   ├── services/          # 비즈니스 로직
│   ├── core/              # 설정, 보안, 예외 처리
│   ├── database/          # ORM 모델 및 마이그레이션
│   ├── evaluation/        # 평가 메트릭
│   └── training/          # 학습 알고리즘
│
├── scripts/               # 유틸리티 스크립트
│   ├── setup/            # 초기 설정
│   ├── training/         # 모델 학습
│   └── deployment/       # 배포 스크립트
│
├── tests/                # 테스트 코드
├── data/                 # 데이터베이스 파일
├── docker/               # Docker 설정
├── commerce/             # Next.js 프론트엔드
└── docs/                 # 상세 문서
```

### API 구조

메인 API 엔드포인트: `fragrance_ai/api/main.py`

주요 라우터:
- `/api/v2/dna/*` - DNA 생성 및 진화
- `/api/v2/auth/*` - 인증 및 권한
- `/api/v2/admin/*` - 관리자 기능
- `/api/v2/public/*` - 공개 레시피

---

## 개발 환경 설정

### 1. 사전 요구사항

- Python 3.10 이상
- PostgreSQL 13 이상 (선택사항, SQLite 대체 가능)
- Redis 6 이상
- Node.js 18 이상 (프론트엔드 개발 시)
- CUDA 12.1 이상 (GPU 사용 시)

### 2. 저장소 클론

```bash
git clone https://github.com/junseong2im/DeulSoom.git
cd DeulSoom
```

### 3. 환경 변수 설정

```bash
# .env.example을 .env로 복사
cp .env.example .env

# .env 파일을 편집하여 실제 값 설정
# - DATABASE_URL: 데이터베이스 연결 문자열
# - REDIS_URL: Redis 연결 문자열
# - SECRET_KEY: JWT 토큰용 비밀키
# - API_KEY: 외부 API 키 (필요시)
```

환경 파일 가이드는 `ENV_FILES_GUIDE.md` 참조.

### 4. Python 가상 환경 생성

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. 데이터베이스 초기화

```bash
# Alembic 마이그레이션 실행
alembic upgrade head

# 초기 데이터 로드
python scripts/setup/setup_database.py
python scripts/setup/load_vector_data.py
```

### 6. 개발 서버 실행

```bash
# API 서버 실행
uvicorn fragrance_ai.api.main:app --reload --port 8000

# 또는 Makefile 사용
make dev
```

서버가 정상적으로 실행되면 http://localhost:8000/docs 에서 API 문서 확인 가능.

### 7. 프론트엔드 실행 (선택사항)

```bash
cd commerce
npm install
npm run dev
```

프론트엔드는 http://localhost:3000 에서 실행됨.

---

## 주요 모듈 설명

### 1. API 레이어 (`fragrance_ai/api/`)

**파일**: `main.py`
- FastAPI 애플리케이션 진입점
- CORS, 보안 미들웨어 설정
- 라우터 통합

**디렉토리**: `routes/`
- `dna_evolution.py`: DNA 생성 및 진화 엔드포인트
- `generation.py`: 향수 레시피 생성
- `auth.py`: 사용자 인증
- `admin.py`: 관리자 기능
- 기타: monitoring, search, stream 등

### 2. AI 모델 (`fragrance_ai/models/`)

**주요 파일**:
- `deep_learning_architecture.py`: UniversalFragranceGenerator (메인 생성 모델)
- `conversation_llm.py`: LLM 통합 (Qwen, Mistral, Llama)
- `embedding.py`: 한국어 특화 임베딩
- `rag_system.py`: RAG 기반 검색 및 생성

### 3. 학습 알고리즘 (`fragrance_ai/training/`)

**주요 파일**:
- `moga_optimizer_stable.py`: MOGA 최적화 알고리즘
- `ppo_engine.py`: PPO 강화학습 엔진
- `qwen_rlhf.py`: Qwen 모델 RLHF 학습
- `rlhf_complete.py`: 통합 RLHF 시스템

### 4. 비즈니스 로직 (`fragrance_ai/services/`)

**주요 파일**:
- `evolution_service.py`: DNA 진화 서비스
- `generation_service.py`: 레시피 생성 서비스
- `orchestrator_service.py`: 서비스 조율

### 5. 데이터베이스 (`fragrance_ai/database/`)

**주요 파일**:
- `models.py`: SQLAlchemy ORM 모델
- `connection.py`: 데이터베이스 연결 관리
- `schema.py`: 데이터베이스 스키마

### 6. 평가 시스템 (`fragrance_ai/evaluation/`)

**주요 파일**:
- `objectives.py`: 다목적 최적화 목적 함수
- `metrics.py`: 평가 메트릭 (다양성, 안정성 등)
- `advanced_evaluator.py`: 고급 평가 시스템

---

## API 사용 가이드

### 인증

대부분의 API는 JWT 토큰 기반 인증을 사용합니다.

```bash
# 로그인
curl -X POST http://localhost:8000/api/v2/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'

# 응답에서 access_token 획득 후 헤더에 포함
curl -X GET http://localhost:8000/api/v2/protected-endpoint \
  -H "Authorization: Bearer <access_token>"
```

### DNA 생성

```bash
curl -X POST http://localhost:8000/api/v2/dna/create \
  -H "Content-Type: application/json" \
  -d '{
    "brief": {
      "style": "fresh",
      "intensity": 0.7,
      "notes": ["citrus", "woody"]
    },
    "name": "Summer Breeze",
    "product_category": "eau_de_parfum"
  }'
```

### 진화 옵션 생성

```bash
curl -X POST http://localhost:8000/api/v2/dna/evolve/options \
  -H "Content-Type: application/json" \
  -d '{
    "dna_id": "dna_abc123",
    "brief": {
      "style": "fresh",
      "intensity": 0.7
    },
    "num_options": 3,
    "algorithm": "PPO"
  }'
```

### 피드백 제출

```bash
curl -X POST http://localhost:8000/api/v2/dna/evolve/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": "exp_xyz789",
    "chosen_id": "opt_1",
    "rating": 4.5
  }'
```

전체 API 문서: http://localhost:8000/docs

---

## 테스트

### 전체 테스트 실행

```bash
pytest tests/
```

### 특정 테스트 실행

```bash
# 단위 테스트
pytest tests/unit/

# 통합 테스트
pytest tests/integration/

# 성능 테스트
pytest tests/performance/
```

### 테스트 커버리지 확인

```bash
pytest --cov=fragrance_ai --cov-report=html tests/
```

---

## 배포

### Docker를 사용한 배포

#### 1. 개발 환경

```bash
docker-compose -f docker/compose/docker-compose.yml up -d
```

#### 2. 프로덕션 환경

```bash
# 환경 변수 설정
cp .env.prod .env

# 빌드 및 실행
docker-compose -f docker/compose/docker-compose.production.yml up -d
```

#### 3. 스케일링

```bash
docker-compose -f docker/compose/docker-compose.scale.yml up -d --scale api=3
```

### Kubernetes 배포

```bash
# ConfigMap 및 Secret 생성
kubectl create configmap fragrance-config --from-env-file=.env.prod
kubectl create secret generic fragrance-secrets --from-env-file=.env.prod

# 배포
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/ingress.yaml
```

### 배포 스크립트 사용

```bash
cd scripts/deployment
bash deploy.sh production
```

상세 배포 가이드: `docs/deployment/DEPLOYMENT_GUIDE.md`

---

## 트러블슈팅

### 일반적인 문제

#### 1. 데이터베이스 연결 실패

**증상**: `ConnectionError: could not connect to server`

**해결방법**:
- PostgreSQL 서비스가 실행 중인지 확인
- `.env` 파일의 `DATABASE_URL` 확인
- 방화벽 설정 확인

```bash
# PostgreSQL 상태 확인
sudo systemctl status postgresql

# 연결 테스트
psql -h localhost -U username -d database_name
```

#### 2. Redis 연결 실패

**증상**: `redis.exceptions.ConnectionError`

**해결방법**:
```bash
# Redis 서비스 확인
sudo systemctl status redis

# Redis 재시작
sudo systemctl restart redis
```

#### 3. 모델 로딩 실패

**증상**: `Model not found` 또는 메모리 부족

**해결방법**:
- 모델 파일이 `models/checkpoints/` 에 존재하는지 확인
- 충분한 메모리 확보 (최소 8GB 권장)
- GPU 사용 시 CUDA 버전 확인

```bash
# 모델 다운로드
python scripts/setup/download_models.py

# CUDA 확인
nvidia-smi
```

#### 4. Import 에러

**증상**: `ModuleNotFoundError: No module named 'fragrance_ai'`

**해결방법**:
```bash
# 가상환경 활성화 확인
which python

# 패키지 재설치
pip install -e .
```

### 로그 확인

```bash
# API 로그
tail -f logs/api_server.log

# 학습 로그
tail -f logs/training.log

# 에러 로그
tail -f logs/error.log
```

자세한 트러블슈팅: `docs/operations/RUNBOOK.md`

---

## 팀 협업 가이드

### 브랜치 전략

**메인 브랜치**:
- `master`: 프로덕션 배포 코드
- `develop`: 개발 통합 브랜치

**기능 브랜치**:
- `feature/*`: 새로운 기능 개발
- `bugfix/*`: 버그 수정
- `hotfix/*`: 긴급 수정
- `refactor/*`: 리팩토링

### 작업 프로세스

1. **이슈 생성**: GitHub Issues에 작업 내용 등록
2. **브랜치 생성**: `git checkout -b feature/issue-123-add-new-feature`
3. **코드 작성**: 기능 구현 및 테스트 작성
4. **커밋**: 명확한 커밋 메시지 작성
5. **Pull Request**: 코드 리뷰 요청
6. **리뷰**: 팀원 리뷰 및 피드백 반영
7. **머지**: 승인 후 develop 브랜치에 머지

### 커밋 메시지 컨벤션

```
<type>: <subject>

<body>
```

**Type**:
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `refactor`: 코드 리팩토링
- `docs`: 문서 수정
- `test`: 테스트 추가/수정
- `chore`: 빌드, 설정 변경

**예시**:
```
feat: Add DNA evolution feedback API

- Implement POST /api/v2/dna/evolve/feedback endpoint
- Add validation for rating parameter
- Update tests for feedback processing
```

### 코드 리뷰 체크리스트

- 코드가 요구사항을 충족하는가?
- 테스트가 포함되어 있는가?
- 코드 스타일 가이드를 준수하는가?
- 에러 처리가 적절한가?
- 문서가 업데이트되었는가?
- 성능 이슈가 없는가?

### 코드 스타일

- Python: PEP 8 준수
- 들여쓰기: 스페이스 4칸
- 최대 줄 길이: 100자
- 타입 힌트 사용 권장
- Docstring 필수 (함수, 클래스)

```python
def calculate_fitness(
    individual: Dict[str, Any],
    weights: Dict[str, float]
) -> float:
    """
    개체의 적합도를 계산합니다.

    Args:
        individual: 평가할 개체
        weights: 목적 함수별 가중치

    Returns:
        적합도 점수 (0.0 ~ 1.0)

    Raises:
        ValueError: individual이 유효하지 않은 경우
    """
    pass
```

---

## 인수인계 가이드

### 신규 팀원 온보딩

#### 1단계: 환경 설정 (1일차)

- [ ] 저장소 접근 권한 획득
- [ ] 개발 환경 설정 (Python, PostgreSQL, Redis 설치)
- [ ] 가상환경 생성 및 의존성 설치
- [ ] 로컬 서버 실행 확인

#### 2단계: 코드베이스 이해 (2-3일차)

- [ ] `PROJECT_STRUCTURE.md` 읽기
- [ ] 주요 모듈 구조 파악
- [ ] API 문서 확인 (http://localhost:8000/docs)
- [ ] 간단한 API 호출 테스트

#### 3단계: 기능 파악 (4-5일차)

- [ ] DNA 생성 플로우 이해
- [ ] MOGA 알고리즘 동작 방식 학습
- [ ] RLHF 학습 프로세스 이해
- [ ] 데이터베이스 스키마 확인

#### 4단계: 첫 작업 시작 (1주차 이후)

- [ ] 간단한 버그 수정 또는 문서 개선 작업 할당
- [ ] 코드 리뷰 프로세스 경험
- [ ] 팀 미팅 참여

### 주요 연락처

- **프로젝트 리드**: junseong2im@gmail.com
- **기술 문의**: GitHub Issues 또는 팀 채널
- **긴급 상황**: [담당자 연락처 입력]

### 주요 문서

- `PROJECT_STRUCTURE.md`: 프로젝트 구조 상세 가이드
- `PROJECT_GUIDE.md`: 프로젝트 전반적인 가이드
- `docs/development/`: 개발 관련 문서
- `docs/deployment/`: 배포 관련 문서
- `docs/operations/RUNBOOK.md`: 운영 매뉴얼

---

## 유지보수 및 모니터링

### 로그 관리

로그 파일 위치: `logs/`
- `api_server.log`: API 서버 로그
- `training.log`: 모델 학습 로그
- `error.log`: 에러 로그

로그 로테이션 설정: `logging/logstash/`

### 모니터링

**Grafana 대시보드**: http://localhost:3001 (기본 설정)

주요 메트릭:
- API 응답 시간
- 에러율
- 메모리 사용량
- GPU 사용률
- 데이터베이스 연결 수

**헬스체크 엔드포인트**:
- `GET /health`: 기본 헬스체크
- `GET /api/v2/monitoring`: 상세 모니터링 정보

### 백업

#### 데이터베이스 백업

```bash
# PostgreSQL 백업
pg_dump -U username database_name > backup_$(date +%Y%m%d).sql

# 복원
psql -U username database_name < backup_20250120.sql
```

#### 모델 체크포인트 백업

```bash
# 모델 백업
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/checkpoints/

# 복원
tar -xzf models_backup_20250120.tar.gz -C models/
```

### 정기 유지보수

**주간**:
- 로그 확인 및 분석
- 에러율 모니터링
- 성능 메트릭 검토

**월간**:
- 데이터베이스 백업
- 모델 체크포인트 백업
- 의존성 패키지 업데이트 검토
- 보안 패치 확인

**분기별**:
- 코드베이스 리팩토링 검토
- 문서 업데이트
- 성능 최적화 작업

---

## 성능 최적화

### 데이터베이스 최적화

```sql
-- 인덱스 생성
CREATE INDEX idx_dna_created_at ON dna_table(created_at);
CREATE INDEX idx_user_id ON user_table(user_id);

-- 쿼리 성능 분석
EXPLAIN ANALYZE SELECT * FROM dna_table WHERE created_at > '2025-01-01';
```

### 캐시 전략

- Redis를 사용한 API 응답 캐싱
- 모델 출력 캐싱 (동일한 입력에 대해)
- 데이터베이스 쿼리 결과 캐싱

### 비동기 처리

- Celery를 사용한 백그라운드 작업
- 긴 작업은 비동기로 처리 (모델 학습 등)

---

## 보안

### API 보안

- JWT 토큰 기반 인증
- HTTPS 강제 (프로덕션)
- Rate Limiting 적용
- CORS 설정 검증

### 데이터베이스 보안

- 비밀번호 해싱 (bcrypt)
- SQL Injection 방지 (ORM 사용)
- 민감한 정보 암호화

### 환경 변수 관리

- `.env` 파일은 절대 Git에 커밋하지 않음
- 프로덕션 비밀키는 별도 관리
- 환경별 설정 분리 (.env.dev, .env.prod)

---

## 라이선스

**프로젝트 라이선스**: Proprietary (상업적 이용 금지)

**사용 모델**:
- Qwen 2.5: Apache 2.0
- Mistral 7B: Apache 2.0
- Llama 3: Llama 3 Community License (조건부 상업 이용 가능)

상세 라이선스 정보: `LICENSE` 파일 참조

---

## 문의 및 지원

### 기술 지원

- **GitHub Issues**: https://github.com/junseong2im/DeulSoom/issues
- **이메일**: junseong2im@gmail.com
- **문서**: `docs/` 디렉토리 참조

### 기여 방법

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to the branch
5. Create Pull Request

기여 가이드라인: `CONTRIBUTING.md` (별도 작성 필요시)

---

## 버전 히스토리

- **v2.0.0** (2025-01): 프로젝트 구조 대규모 리팩토링
- **v1.5.0** (2024-12): RLHF 통합 완료
- **v1.0.0** (2024-10): 초기 릴리스

상세 변경 이력: `CHANGELOG.md`

---

**최종 업데이트**: 2025-10-20
**작성자**: Jun Seong Im
**연락처**: junseong2im@gmail.com
