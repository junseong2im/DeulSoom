"""
Authentication Dependencies
FastAPI 인증 의존성 함수
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional

from app.auth.jwt_handler import verify_token
from app.auth.schemas import TokenPayload
from fragrance_ai.database.models import User

# HTTP Bearer 토큰 추출
security = HTTPBearer()


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    JWT 토큰에서 현재 사용자 ID 추출

    Args:
        credentials: HTTP Authorization Header의 Bearer 토큰

    Returns:
        사용자 ID

    Raises:
        HTTPException: 토큰이 유효하지 않은 경우
    """
    token = credentials.credentials

    # 토큰 검증
    payload = verify_token(token, token_type="access")

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_id


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """
    JWT 토큰에서 현재 사용자 ID 추출 (선택적)

    인증이 선택적인 엔드포인트에서 사용
    토큰이 없거나 유효하지 않으면 None 반환

    Args:
        credentials: HTTP Authorization Header의 Bearer 토큰

    Returns:
        사용자 ID or None
    """
    if not credentials:
        return None

    token = credentials.credentials
    payload = verify_token(token, token_type="access")

    if not payload:
        return None

    return payload.get("sub")


def get_token_payload(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenPayload:
    """
    JWT 토큰에서 전체 페이로드 추출

    Args:
        credentials: HTTP Authorization Header의 Bearer 토큰

    Returns:
        TokenPayload

    Raises:
        HTTPException: 토큰이 유효하지 않은 경우
    """
    token = credentials.credentials
    payload = verify_token(token, token_type="access")

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenPayload(**payload)


def verify_admin_role(
    user_id: str = Depends(get_current_user_id)
) -> str:
    """
    관리자 권한 검증

    Args:
        user_id: 현재 사용자 ID

    Returns:
        사용자 ID

    Raises:
        HTTPException: 관리자가 아닌 경우
    """
    # TODO: 데이터베이스에서 사용자 역할 확인
    # 현재는 간단히 user_id로만 확인 (실제로는 DB 조회 필요)

    # 임시: 나중에 DB 조회로 대체
    # from app.database import get_db
    # db = next(get_db())
    # user = db.query(User).filter(User.id == user_id).first()
    # if not user or user.role != "admin":
    #     raise HTTPException(...)

    return user_id


def verify_expert_role(
    user_id: str = Depends(get_current_user_id)
) -> str:
    """
    전문가 권한 검증

    Args:
        user_id: 현재 사용자 ID

    Returns:
        사용자 ID

    Raises:
        HTTPException: 전문가가 아닌 경우
    """
    # TODO: 데이터베이스에서 사용자 역할 확인
    return user_id


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'get_current_user_id',
    'get_current_user_optional',
    'get_token_payload',
    'verify_admin_role',
    'verify_expert_role'
]
