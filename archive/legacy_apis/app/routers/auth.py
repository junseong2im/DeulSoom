"""
Authentication Router
사용자 인증 관련 REST API 엔드포인트
"""

from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from typing import Optional
import uuid
from datetime import datetime

from app.auth import (
    UserRegister, UserLogin, TokenResponse, TokenRefresh, UserProfile,
    create_access_token, create_refresh_token, refresh_access_token,
    verify_password, get_password_hash, get_current_user_id,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from fragrance_ai.database.models import User
from fragrance_ai.database.base import SessionLocal

router = APIRouter(prefix="/auth", tags=["Authentication"])


def get_db():
    """데이터베이스 세션 의존성"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    사용자 등록

    - 이메일 중복 확인
    - 비밀번호 해싱
    - JWT 토큰 발급
    """
    # 이메일 중복 확인
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # 사용자명 중복 확인
    existing_username = db.query(User).filter(User.username == user_data.username).first()
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )

    # 새 사용자 생성
    new_user = User(
        id=str(uuid.uuid4()),
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        hashed_password=get_password_hash(user_data.password),
        is_active=True,
        is_verified=False,
        role="customer"
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # JWT 토큰 생성
    access_token = create_access_token(
        data={"sub": new_user.id, "email": new_user.email}
    )
    refresh_token = create_refresh_token(
        data={"sub": new_user.id, "email": new_user.email}
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/login", response_model=TokenResponse)
async def login(login_data: UserLogin, db: Session = Depends(get_db)):
    """
    사용자 로그인

    - 이메일/비밀번호 검증
    - JWT 토큰 발급
    """
    # 사용자 조회
    user = db.query(User).filter(User.email == login_data.email).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 비밀번호 검증
    if not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 활성 사용자 확인
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )

    # JWT 토큰 생성
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.id, "email": user.email}
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(token_data: TokenRefresh):
    """
    Access Token 갱신

    - Refresh Token을 사용하여 새로운 Access Token 발급
    """
    new_access_token = refresh_access_token(token_data.refresh_token)

    if not new_access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=token_data.refresh_token,  # 기존 refresh token 재사용
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.get("/profile", response_model=UserProfile)
async def get_profile(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    현재 사용자 프로필 조회

    - JWT 토큰에서 사용자 ID 추출
    - 사용자 정보 반환
    """
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return UserProfile(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        is_verified=user.is_verified,
        created_at=user.created_at
    )


@router.post("/logout")
async def logout(user_id: str = Depends(get_current_user_id)):
    """
    로그아웃

    - JWT는 stateless이므로 서버에서 별도 처리 없음
    - 클라이언트에서 토큰 삭제 필요
    """
    return {
        "status": "success",
        "message": "Logged out successfully. Please remove tokens from client."
    }


@router.get("/verify")
async def verify_token_endpoint(user_id: str = Depends(get_current_user_id)):
    """
    토큰 검증

    - JWT 토큰이 유효한지 확인
    - 프론트엔드에서 토큰 유효성 체크 시 사용
    """
    return {
        "status": "valid",
        "user_id": user_id
    }


# ============================================================================
# Export
# ============================================================================

__all__ = ['router']
