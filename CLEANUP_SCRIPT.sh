#!/bin/bash
# 프로젝트 정리 자동화 스크립트
# 실행 전 반드시 PROJECT_CLEANUP_ANALYSIS.md 확인!

echo "=================================="
echo "프로젝트 파일 정리 시작"
echo "=================================="

# 백업 디렉토리 생성
echo "📦 백업 디렉토리 생성..."
mkdir -p archive/removed_files
mkdir -p archive/legacy_apis
mkdir -p archive/data_collection
mkdir -p archive/tests_legacy

# 1단계: 잘못된 학습 스크립트 제거
echo ""
echo "🗑️  1단계: 잘못된 학습 스크립트 제거..."
echo "  - train_our_model.py"
echo "  - real_train_our_model.py"
echo "  - train_fragrance_model.py"

mv train_our_model.py archive/removed_files/ 2>/dev/null || echo "  ⚠️  train_our_model.py 없음"
mv real_train_our_model.py archive/removed_files/ 2>/dev/null || echo "  ⚠️  real_train_our_model.py 없음"
mv train_fragrance_model.py archive/removed_files/ 2>/dev/null || echo "  ⚠️  train_fragrance_model.py 없음"

# 2단계: 중복 API 파일 아카이브
echo ""
echo "📁 2단계: 중복/레거시 API 파일 아카이브..."
mv simple_api.py archive/legacy_apis/ 2>/dev/null || echo "  ⚠️  simple_api.py 없음"
mv simple_fragrance_api.py archive/legacy_apis/ 2>/dev/null || echo "  ⚠️  simple_fragrance_api.py 없음"
mv real_ai_api.py archive/legacy_apis/ 2>/dev/null || echo "  ⚠️  real_ai_api.py 없음"
mv real_trained_api.py archive/legacy_apis/ 2>/dev/null || echo "  ⚠️  real_trained_api.py 없음"
mv real_fragrance_api.py archive/legacy_apis/ 2>/dev/null || echo "  ⚠️  real_fragrance_api.py 없음"
mv main_fragrance_api.py archive/legacy_apis/ 2>/dev/null || echo "  ⚠️  main_fragrance_api.py 없음"
mv advanced_model_api.py archive/legacy_apis/ 2>/dev/null || echo "  ⚠️  advanced_model_api.py 없음"
mv admin_server.py archive/legacy_apis/ 2>/dev/null || echo "  ⚠️  admin_server.py 없음"
mv health_api_server.py archive/legacy_apis/ 2>/dev/null || echo "  ⚠️  health_api_server.py 없음"

# 3단계: 데이터 수집 스크립트 아카이브 (선택사항)
echo ""
echo "📊 3단계: 데이터 수집 스크립트 아카이브..."
echo "  (이미 데이터가 DB에 있다면 이동, 아니면 스킵)"
read -p "데이터 수집 스크립트를 아카이브하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    mv add_famous_perfumes.py archive/data_collection/ 2>/dev/null
    mv add_ifra_ingredients.py archive/data_collection/ 2>/dev/null
    mv add_more_ingredients.py archive/data_collection/ 2>/dev/null
    mv add_more_perfumes_part1.py archive/data_collection/ 2>/dev/null
    mv add_more_perfumes_part2.py archive/data_collection/ 2>/dev/null
    mv add_real_perfumes_bulk.py archive/data_collection/ 2>/dev/null
    mv add_real_perfumes_simple.py archive/data_collection/ 2>/dev/null
    mv add_webfetch_perfumes.py archive/data_collection/ 2>/dev/null
    mv expand_fragrance_data.py archive/data_collection/ 2>/dev/null
    mv generate_1000_perfumes.py archive/data_collection/ 2>/dev/null
    mv create_perfume_recipes.py archive/data_collection/ 2>/dev/null
    mv crawl_fragrantica.py archive/data_collection/ 2>/dev/null
    mv crawl_selenium.py archive/data_collection/ 2>/dev/null
    echo "  ✅ 데이터 수집 스크립트 아카이브 완료"
else
    echo "  ⏭️  데이터 수집 스크립트 유지"
fi

# 4단계: 테스트 파일 정리
echo ""
echo "🧪 4단계: 루트의 테스트/데모 파일 정리..."
mv demo_llm_ensemble.py archive/tests_legacy/ 2>/dev/null
mv demo_go_nogo_gate.py archive/tests_legacy/ 2>/dev/null
mv analyze_data.py archive/tests_legacy/ 2>/dev/null
mv test_note_matching.py archive/tests_legacy/ 2>/dev/null
mv smoke_test_api.py archive/tests_legacy/ 2>/dev/null
mv smoke_test_security.py archive/tests_legacy/ 2>/dev/null

# 완료 보고
echo ""
echo "=================================="
echo "✅ 정리 완료!"
echo "=================================="
echo ""
echo "📊 정리 결과:"
echo "  - 제거된 파일: $(ls -1 archive/removed_files/ 2>/dev/null | wc -l)개"
echo "  - 아카이브된 API: $(ls -1 archive/legacy_apis/ 2>/dev/null | wc -l)개"
echo "  - 아카이브된 데이터 스크립트: $(ls -1 archive/data_collection/ 2>/dev/null | wc -l)개"
echo "  - 아카이브된 테스트: $(ls -1 archive/tests_legacy/ 2>/dev/null | wc -l)개"
echo ""
echo "📁 아카이브 위치: ./archive/"
echo ""
echo "다음 단계:"
echo "  1. 새로운 학습 스크립트 작성: train_universal_model.py"
echo "  2. 프로젝트 구조 문서 업데이트"
echo "  3. Git 커밋"
