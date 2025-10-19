# Environment Files Guide

이 프로젝트는 다음과 같은 환경 변수 파일들을 사용합니다:

## 파일 구조

- **`.env.example`** - 템플릿 파일 (Git 추적됨)
  - 모든 필요한 환경 변수의 예시를 포함
  - 새로운 개발자가 참고할 수 있는 가이드
  - 민감한 정보는 포함하지 않음

- **`.env.dev`** - 개발 환경 설정
  - 로컬 개발 환경용 설정
  - Git에 커밋되어 팀원들과 공유

- **`.env.prod`** - 프로덕션 환경 설정
  - 프로덕션 서버용 설정
  - Git에 커밋되어 배포 자동화에 사용

- **`.env`** - 개인 로컬 설정 (Git 무시됨)
  - 각 개발자의 개인적인 설정
  - `.env.example`을 복사하여 생성
  - `.gitignore`에 포함되어 Git 추적 안 됨

## 사용 방법

### 1. 처음 시작하는 경우
```bash
cp .env.example .env
# .env 파일을 열어서 필요한 값들을 설정
```

### 2. 개발 환경 실행
```bash
# .env.dev 사용
docker-compose --env-file .env.dev up
```

### 3. 프로덕션 배포
```bash
# .env.prod 사용
docker-compose --env-file .env.prod up -d
```

## 주의사항

⚠️ **절대 민감한 정보를 Git에 커밋하지 마세요!**
- API 키, 비밀번호, 토큰 등은 `.env` 파일에만 저장
- `.env.dev`, `.env.prod`에는 개발/프로덕션용 공개 가능한 설정만 포함

## 제거된 파일들 (정리 완료)
- `.env.production` → `.env.prod`로 통합
- `.env.production.template` → `.env.example`로 통합  
- `.env.staging` → `.env.dev`로 통합
