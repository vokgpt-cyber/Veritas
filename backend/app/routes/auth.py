"""Authentication API routes."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.app.auth import (
    LoginRequest,
    TokenData,
    TokenResponse,
    authenticate_user,
    auth_config,
    change_user_password,
    create_access_token,
    deactivate_session,
    get_current_user,
)
from backend.core.audit import AuditAction, audit_log

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request, respecting X-Forwarded-For."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.post("/login", response_model=TokenResponse)
async def login(request_body: LoginRequest, request: Request) -> TokenResponse:
    """
    Authenticate user and return JWT access token.

    Args:
        request_body: Login credentials (username, password)
        request: FastAPI request object for IP extraction

    Returns:
        TokenResponse with access_token and expiry

    Raises:
        HTTPException 401: If credentials are invalid
    """
    client_ip = _get_client_ip(request)
    user = authenticate_user(request_body.username, request_body.password)

    if not user:
        logger.warning(f"Failed login attempt for user: {request_body.username}")
        audit_log(
            action=AuditAction.LOGIN_FAILURE,
            username=request_body.username,
            ip_address=client_ip,
            success=False,
            error="Invalid credentials",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token, expires_at = create_access_token(
        username=user["username"],
        role=user.get("role", "user"),
    )

    audit_log(
        action=AuditAction.LOGIN_SUCCESS,
        username=user["username"],
        ip_address=client_ip,
        details={"role": user.get("role", "user")},
    )

    logger.info(f"User logged in: {user['username']}")

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=auth_config.access_token_expire_minutes * 60,
        must_change_password=bool(user.get("must_change_password", False)),
        username=user["username"],
        role=user.get("role", "user"),
    )


@router.get("/me")
async def me(user: TokenData = Depends(get_current_user)) -> dict:
    """Return current authenticated user."""
    from backend.app.auth import get_user

    stored = get_user(user.username) or {}
    return {
        "username": user.username,
        "role": user.role,
        "must_change_password": bool(
            stored.get("must_change_password", False)
        ),
    }


@router.post("/change-password")
async def change_password(
    payload: dict,
    request: Request,
    user: TokenData = Depends(get_current_user),
) -> dict:
    """Change own password."""
    current_password = str(payload.get("current_password") or "")
    new_password = str(payload.get("new_password") or "")
    if len(new_password) < 8:
        raise HTTPException(
            status_code=400,
            detail="New password must be at least 8 characters",
        )
    try:
        change_user_password(user.username, current_password, new_password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    audit_log(
        action=AuditAction.USER_PASSWORD_CHANGE,
        username=user.username,
        ip_address=_get_client_ip(request),
    )
    return {"status": "ok"}


@router.post("/logout")
async def logout(
    request: Request,
    user: TokenData = Depends(get_current_user),
) -> dict:
    """Revoke the current persisted session."""
    deactivate_session(user.jti)
    audit_log(
        action=AuditAction.SESSION_LOGOUT,
        username=user.username,
        ip_address=_get_client_ip(request),
    )
    return {"status": "ok"}


@router.post("/verify")
async def verify_token(token: str) -> dict:
    """
    Verify if a token is valid (for frontend validation).

    Args:
        token: JWT token string

    Returns:
        Token validity status and user info
    """
    from backend.app.auth import decode_token

    try:
        token_data = decode_token(token)
        return {
            "valid": True,
            "username": token_data.username,
            "role": token_data.role,
        }
    except HTTPException:
        return {"valid": False}
