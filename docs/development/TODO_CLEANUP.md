# 프로젝트 정리 작업 TODO

## 🎯 현재 상태
- ✅ 전체 파일 분석 완료
- ✅ 제거 대상 파일 식별 완료
- ⏸️ 실제 정리 작업 대기 중

---

## 📋 실행 순서

### 1️⃣ 정리 스크립트 실행 (10분)
```bash
# Windows
bash CLEANUP_SCRIPT.sh

# 또는 수동으로
mkdir -p archive/removed_files archive/legacy_apis archive/data_collection archive/tests_legacy
```

### 2️⃣ 확실히 제거할 파일 (즉시 실행 가능)
```bash
# 잘못된 학습 스크립트
rm train_our_model.py
rm real_train_our_model.py
rm train_fragrance_model.py
```

### 3️⃣ 아카이브할 파일 (복구 가능하도록)
```bash
# 중복 API 파일
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

### 4️⃣ 새 학습 스크립트 작성 (30-60분)
**파일명**: `train_universal_model.py`

**포함 내용**:
- UniversalFragranceGenerator 전용 학습
- AdamW 옵티마이저 + Cosine Annealing with Warm Restarts
- Mixed Precision Training (AMP)
- Gradient Accumulation (메모리 절약)
- Label Smoothing
- Early Stopping
- Best Model Checkpoint
- TensorBoard 로깅
- Learning Rate Finder

### 5️⃣ Git 커밋
```bash
git add .
git commit -m "refactor: Remove duplicate and incorrect training scripts

- Remove incorrect training scripts (train_our_model.py, real_train_our_model.py, train_fragrance_model.py)
- Archive legacy API files to archive/legacy_apis/
- Archive completed data collection scripts
- Add new optimized training script for UniversalFragranceGenerator

Related: Deep learning architecture cleanup"
```

---

## 📝 상세 작업 내역

### ❌ 제거할 파일 (3개)
| 파일 | 이유 | 상태 |
|------|------|------|
| train_our_model.py | 잘못된 학습 방식 | ⏸️ 대기 |
| real_train_our_model.py | 잘못된 학습 방식 | ⏸️ 대기 |
| train_fragrance_model.py | GPT-2 사용 (우리 모델 아님) | ⏸️ 대기 |

### 📦 아카이브할 파일 (20개+)
| 분류 | 파일 수 | 상태 |
|------|---------|------|
| 중복 API | 9개 | ⏸️ 대기 |
| 데이터 수집 | 11개 | ⏸️ 대기 (선택) |
| 테스트/데모 | 6개+ | ⏸️ 대기 |

### ✨ 새로 만들 파일 (1개)
| 파일 | 목적 | 상태 |
|------|------|------|
| train_universal_model.py | UniversalFragranceGenerator 최적화 학습 | ⏸️ 대기 |

---

## 🔍 주요 유지 파일 (변경 없음)

### 프로덕션 코드
```
✅ app/                              # FastAPI 프로덕션 앱
✅ fragrance_ai/api/main.py          # 메인 API
✅ fragrance_ai/api/routes/*.py      # API 라우터
✅ fragrance_ai/models/              # 모델 디렉토리
✅ fragrance_ai/database/            # DB 모델
✅ fragrance_ai/core/                # 코어 설정
```

### 핵심 모델
```
✅ deep_learning_architecture.py    # UniversalFragranceGenerator (우리 진짜 모델)
✅ conversation_llm.py               # ConversationalLLM (사용 중)
✅ advanced_generator.py             # AdvancedFragranceGenerator (LoRA)
✅ embedding.py                      # 임베딩
✅ rag_system.py                     # RAG
```

---

## 💡 다음 작업 시 체크리스트

### 시작 전
- [ ] `PROJECT_CLEANUP_ANALYSIS.md` 재확인
- [ ] 현재 Git 상태 확인 (`git status`)
- [ ] 중요 파일 백업 확인

### 정리 작업
- [ ] `CLEANUP_SCRIPT.sh` 실행 또는 수동 정리
- [ ] archive/ 디렉토리 생성 확인
- [ ] 제거/이동된 파일 목록 확인

### 학습 스크립트 작성
- [ ] `train_universal_model.py` 작성
- [ ] 데이터 로더 구현
- [ ] 학습 루프 구현
- [ ] 평가/체크포인트 구현
- [ ] 테스트 실행

### 마무리
- [ ] 루트 디렉토리 파일 수 확인 (50개 → 20개 목표)
- [ ] Git 커밋
- [ ] README 업데이트 (필요시)

---

## 📞 참고 문서
- **분석 보고서**: `PROJECT_CLEANUP_ANALYSIS.md`
- **실행 스크립트**: `CLEANUP_SCRIPT.sh`
- **모델 구조**: `fragrance_ai/models/deep_learning_architecture.py`

---

## ⚠️ 주의사항
1. **백업 필수**: 삭제 전 archive/ 디렉토리로 이동
2. **Git 확인**: 변경 사항 커밋 전 확인
3. **테스트**: 정리 후 주요 기능 동작 확인
4. **문서화**: 변경 사항 문서에 기록

---

생성일: 2025-10-18
다음 작업 예정일: TBD
