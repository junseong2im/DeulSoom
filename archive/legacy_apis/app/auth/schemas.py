"""
Authentication Schemas
인증 관련 Pydantic 스키마
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class UserRegister(BaseModel):
    """사용자 등록 요청"""
    username: str = Field(..., min_length=3, max_length=50, description="사용자명")
    email: EmailStr = Field(..., description="이메일")
    password: str = Field(..., min_length=8, max_length=100, description="비밀번호")
    full_name: Optional[str] = Field(None, max_length=200, description="전체 이름")


class UserLogin(BaseModel):
    """사용자 로그인 요청"""
    email: EmailStr = Field(..., description="이메일")
    password: str = Field(..., description="비밀번호")


class TokenResponse(BaseModel):
    """토큰 응답"""
    access_token: str = Field(..., description="Access JWT Token")
    refresh_token: str = Field(..., description="Refresh JWT Token")
    token_type: str = Field(default="bearer", description="토큰 타입")
    expires_in: int = Field(..., description="Access Token 만료 시간 (초)")


class TokenRefresh(BaseModel):
    """토큰 리프레시 요청"""
    refresh_token: str = Field(..., description="Refresh JWT Token")


class UserProfile(BaseModel):
    """사용자 프로필"""
    id: str = Field(..., description="사용자 ID")
    username: str = Field(..., description="사용자명")
    email: EmailStr = Field(..., description="이메일")
    full_name: Optional[str] = Field(None, description="전체 이름")
    role: str = Field(default="customer", description="역할")
    is_active: bool = Field(default=True, description="활성 상태")
    is_verified: bool = Field(default=False, description="이메일 인증 여부")
    created_at: datetime = Field(..., description="가입 일시")

    class Config:
        from_attributes = True


class TokenPayload(BaseModel):
    """JWT 토큰 페이로드"""
    sub: str = Field(..., description="User ID")
    email: Optional[str] = Field(None, description="Email")
    exp: Optional[int] = Field(None, description="Expiration timestamp")
    iat: Optional[int] = Field(None, description="Issued at timestamp")
    type: str = Field(..., description="Token type (access/refresh)")


class PasswordChange(BaseModel):
    """비밀번호 변경 요청"""
    old_password: str = Field(..., description="현재 비밀번호")
    new_password: str = Field(..., min_length=8, max_length=100, description="새 비밀번호")


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'UserRegister',
    'UserLogin',
    'TokenResponse',
    'TokenRefresh',
    'UserProfile',
    'TokenPayload',
    'PasswordChange'
]
