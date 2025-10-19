"""
JWT Token Handler
JWT 토큰 생성, 검증, 리프레시 기능 제공
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import os

# JWT 설정
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production-12345678")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# 비밀번호 해싱
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """비밀번호 검증"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """비밀번호 해싱"""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    JWT Access Token 생성

    Args:
        data: 토큰에 포함할 데이터 (sub, email 등)
        expires_delta: 만료 시간 (기본 30분)

    Returns:
        JWT 토큰 문자열
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    JWT Refresh Token 생성

    Args:
        data: 토큰에 포함할 데이터

    Returns:
        Refresh JWT 토큰
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
    """
    JWT 토큰 검증

    Args:
        token: JWT 토큰
        token_type: 토큰 타입 ("access" or "refresh")

    Returns:
        토큰 페이로드 (검증 성공) or None (검증 실패)
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # 토큰 타입 확인
        if payload.get("type") != token_type:
            return None

        # 만료 시간 확인 (자동으로 처리되지만 명시적으로 확인)
        exp = payload.get("exp")
        if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
            return None

        return payload

    except JWTError:
        return None


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    JWT 토큰 디코딩 (검증 없이)

    Args:
        token: JWT 토큰

    Returns:
        토큰 페이로드
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_signature": False})
        return payload
    except JWTError:
        return None


def refresh_access_token(refresh_token: str) -> Optional[str]:
    """
    Refresh Token을 사용하여 새로운 Access Token 발급

    Args:
        refresh_token: Refresh JWT 토큰

    Returns:
        새로운 Access Token or None
    """
    payload = verify_token(refresh_token, token_type="refresh")

    if not payload:
        return None

    # 새로운 access token 생성
    user_id = payload.get("sub")
    email = payload.get("email")

    if not user_id:
        return None

    new_access_token = create_access_token(
        data={"sub": user_id, "email": email}
    )

    return new_access_token


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'verify_password',
    'get_password_hash',
    'create_access_token',
    'create_refresh_token',
    'verify_token',
    'decode_token',
    'refresh_access_token',
    'ACCESS_TOKEN_EXPIRE_MINUTES',
    'REFRESH_TOKEN_EXPIRE_DAYS'
]
