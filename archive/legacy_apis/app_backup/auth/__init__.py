"""
Authentication Module
JWT 기반 인증 시스템
"""

from app.auth.jwt_handler import (
    create_access_token,
    create_refresh_token,
    verify_token,
    refresh_access_token,
    verify_password,
    get_password_hash
)

from app.auth.dependencies import (
    get_current_user_id,
    get_current_user_optional,
    get_token_payload,
    verify_admin_role,
    verify_expert_role
)

from app.auth.schemas import (
    UserRegister,
    UserLogin,
    TokenResponse,
    TokenRefresh,
    UserProfile,
    TokenPayload,
    PasswordChange
)

__all__ = [
    # JWT Handler
    'create_access_token',
    'create_refresh_token',
    'verify_token',
    'refresh_access_token',
    'verify_password',
    'get_password_hash',

    # Dependencies
    'get_current_user_id',
    'get_current_user_optional',
    'get_token_payload',
    'verify_admin_role',
    'verify_expert_role',

    # Schemas
    'UserRegister',
    'UserLogin',
    'TokenResponse',
    'TokenRefresh',
    'UserProfile',
    'TokenPayload',
    'PasswordChange'
]
