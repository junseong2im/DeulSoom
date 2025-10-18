# 프로젝트 전체 파일 분석 및 정리 계획

## 📊 전체 현황
- **루트 Python 파일**: 50개
- **API 관련 파일**: 10개+
- **학습 스크립트**: 10개
- **데이터 수집 스크립트**: 10개+

---

## 🗑️ 제거 대상 파일 (안 쓰는 파일)

### 1. **중복/잘못된 학습 스크립트** ❌
```
✗ train_our_model.py           # 잘못된 학습 방식 (generate_recipe 사용)
✗ real_train_our_model.py       # 잘못된 학습 방식 (adaptation_score 사용)
✗ train_fragrance_model.py      # GPT-2 사용 (우리 모델 아님)
```
**이유**:
- 실제로 모델 파라미터를 학습하지 못함
- 어디에서도 import되지 않음
- UniversalFragranceGenerator가 진짜 우리 모델

### 2. **중복 API 파일** 🔄
```
✗ simple_api.py                 # 기본 테스트용
✗ simple_fragrance_api.py       # 기본 테스트용
✗ real_ai_api.py                # 중복 기능
✗ real_trained_api.py           # 중복 기능
✗ real_fragrance_api.py         # 중복 기능
✗ main_fragrance_api.py         # 레거시
✗ advanced_model_api.py         # 레거시
✗ admin_server.py               # 단순 관리용
✗ health_api_server.py          # 테스트용
```
**이유**:
- fragrance_ai/api/main.py가 메인 API
- app/ 디렉토리가 실제 프로덕션 엔트리포인트
- 나머지는 테스트/개발용 중복 파일

### 3. **데이터 수집 스크립트 (완료됨)** 📦
```
? add_famous_perfumes.py        # 데이터 추가 완료
? add_ifra_ingredients.py       # 데이터 추가 완료
? add_more_ingredients.py       # 데이터 추가 완료
? add_more_perfumes_part1.py    # 데이터 추가 완료
? add_more_perfumes_part2.py    # 데이터 추가 완료
? add_real_perfumes_bulk.py     # 데이터 추가 완료
? add_real_perfumes_simple.py   # 데이터 추가 완료
? add_webfetch_perfumes.py      # 데이터 추가 완료
? expand_fragrance_data.py      # 데이터 확장 완료
? generate_1000_perfumes.py     # 데이터 생성 완료
? create_perfume_recipes.py     # 레시피 생성 완료
```
**판단 필요**: 데이터가 이미 DB에 있다면 제거 가능

### 4. **크롤링 스크립트** 🕷️
```
? crawl_fragrantica.py          # 일회성 크롤링
? crawl_selenium.py             # 일회성 크롤링
```
**판단 필요**: 향후 데이터 업데이트에 필요한지 확인

### 5. **테스트/검증 스크립트** 🧪
```
? analyze_data.py               # 일회성 분석
? test_note_matching.py         # 테스트 파일
? demo_llm_ensemble.py          # 데모 파일
? demo_go_nogo_gate.py          # 데모 파일
? smoke_test_api.py             # 테스트는 tests/ 디렉토리로
? smoke_test_security.py        # 테스트는 tests/ 디렉토리로
? run_exception_tests.py        # 테스트는 tests/ 디렉토리로
? run_performance_tests.py      # 테스트는 tests/ 디렉토리로
```
**판단 필요**: tests/ 디렉토리로 이동 또는 제거

### 6. **학습/검증 파일 (RL/RLHF)** 🤖
```
? train_ppo_real.py             # PPO 학습 (사용 중?)
? train_real_fragrance.py       # 학습 스크립트 (사용 중?)
? train_validator.py            # 검증 스크립트
? test_integration_complete.py  # 통합 테스트
? test_rlhf_complete.py         # RLHF 테스트
? verify_rlhf.py                # RLHF 검증
? test_rl_smoke.py              # RL 테스트
? test_rl_smoke_new.py          # RL 테스트
? test_moga_stability.py        # MOGA 테스트
? test_moga_real.py             # MOGA 테스트
? test_qwen_gpu.py              # GPU 테스트
```
**판단 필요**: 실제로 사용 중인지 확인

---

## ✅ 유지해야 할 핵심 파일

### 프로덕션 API
```
✓ app/                          # FastAPI 프로덕션 앱
✓ fragrance_ai/api/main.py      # 메인 API
✓ fragrance_ai/api/routes/*.py  # API 라우터들
```

### 핵심 모델
```
✓ fragrance_ai/models/deep_learning_architecture.py  # UniversalFragranceGenerator (진짜 우리 모델)
✓ fragrance_ai/models/conversation_llm.py            # 대화형 LLM (사용 중)
✓ fragrance_ai/models/advanced_generator.py          # AdvancedFragranceGenerator
✓ fragrance_ai/models/embedding.py                   # 임베딩 모델
✓ fragrance_ai/models/rag_system.py                  # RAG 시스템
✓ fragrance_ai/models/master_perfumer.py             # 마스터 조향사
```

### 데이터베이스 & 설정
```
✓ fragrance_ai/database/                # DB 모델 및 연결
✓ fragrance_ai/core/                    # 코어 설정
✓ setup_database.py                     # DB 초기화
```

---

## 🚀 작업 계획

### 1단계: 확실히 제거 가능한 파일 (중복/잘못된 학습 스크립트)
```bash
rm train_our_model.py
rm real_train_our_model.py
rm train_fragrance_model.py
```

### 2단계: 중복 API 파일 제거
```bash
# scripts/ 디렉토리로 이동 (혹시 필요할 수 있으니)
mkdir -p archive/legacy_apis
mv simple_api.py archive/legacy_apis/
mv simple_fragrance_api.py archive/legacy_apis/
mv real_ai_api.py archive/legacy_apis/
mv real_trained_api.py archive/legacy_apis/
mv real_fragrance_api.py archive/legacy_apis/
mv main_fragrance_api.py archive/legacy_apis/
mv advanced_model_api.py archive/legacy_apis/
mv admin_server.py archive/legacy_apis/
mv health_api_server.py archive/legacy_apis/
```

### 3단계: 데이터 수집 스크립트 정리
```bash
mkdir -p archive/data_collection
mv add_*.py archive/data_collection/
mv crawl_*.py archive/data_collection/
mv expand_fragrance_data.py archive/data_collection/
mv generate_1000_perfumes.py archive/data_collection/
mv create_perfume_recipes.py archive/data_collection/
```

### 4단계: 테스트 파일 정리
```bash
# tests/ 디렉토리로 이동
mkdir -p tests/legacy
mv demo_*.py tests/legacy/
mv test_*.py tests/legacy/ (tests/ 디렉토리에 없는 것만)
mv *_test*.py tests/legacy/
```

### 5단계: 새로운 최적화 학습 스크립트 생성
```
✨ train_universal_model.py (새로 작성)
   - UniversalFragranceGenerator 전용
   - AdamW + Cosine Annealing
   - Mixed Precision Training (AMP)
   - Gradient Accumulation
   - Early Stopping
   - Best Model Checkpoint
```

---

## 📈 기대 효과

- **루트 파일 수**: 50개 → 약 15-20개로 감소
- **명확한 구조**: API, 모델, 스크립트 분리
- **유지보수성 향상**: 중복 제거로 코드 혼란 감소
- **성능 최적화**: 올바른 학습 스크립트로 모델 품질 향상
