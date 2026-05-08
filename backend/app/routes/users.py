"""Admin user-management API for VERITAS 1.0."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from backend.app.auth import (
    TokenData,
    create_user,
    delete_user,
    list_users,
    require_admin,
    reset_user_password,
    update_user,
)
from backend.core.audit import AuditAction, audit_log

router = APIRouter(prefix="/api/users", tags=["users"])


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=2, max_length=80)
    password: str = Field(min_length=8, max_length=256)
    role: str = "operator"
    must_change_password: bool = True


class UserUpdateRequest(BaseModel):
    role: Optional[str] = None
    is_active: Optional[bool] = None
    must_change_password: Optional[bool] = None


class PasswordResetRequest(BaseModel):
    password: str = Field(min_length=8, max_length=256)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _public_user(user: dict) -> dict:
    return {
        "username": user["username"],
        "role": user.get("role", "operator"),
        "is_active": bool(user.get("is_active", False)),
        "must_change_password": bool(user.get("must_change_password", False)),
        "created_at": user.get("created_at", ""),
        "updated_at": user.get("updated_at", ""),
    }


@router.get("/")
async def users_list(
    admin: TokenData = Depends(require_admin),
) -> list[dict]:
    """List users for the admin panel."""
    return [_public_user(u) for u in list_users()]


@router.post("/")
async def users_create(
    payload: UserCreateRequest,
    request: Request,
    admin: TokenData = Depends(require_admin),
) -> dict:
    """Create a user with a temporary password."""
    try:
        user = create_user(
            username=payload.username,
            password=payload.password,
            role=payload.role,
            must_change_password=payload.must_change_password,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit_log(
        action=AuditAction.USER_CREATE,
        username=admin.username,
        ip_address=_client_ip(request),
        resource_type="user",
        resource_id=payload.username,
        details={"role": payload.role},
    )
    return _public_user(user)


@router.put("/{username}")
async def users_update(
    username: str,
    payload: UserUpdateRequest,
    request: Request,
    admin: TokenData = Depends(require_admin),
) -> dict:
    """Update role/active flags."""
    try:
        user = update_user(
            username,
            role=payload.role,
            is_active=payload.is_active,
            must_change_password=payload.must_change_password,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit_log(
        action=AuditAction.USER_UPDATE,
        username=admin.username,
        ip_address=_client_ip(request),
        resource_type="user",
        resource_id=username,
        details=payload.model_dump(exclude_none=True),
    )
    return _public_user(user)


@router.post("/{username}/reset-password")
async def users_reset_password(
    username: str,
    payload: PasswordResetRequest,
    request: Request,
    admin: TokenData = Depends(require_admin),
) -> dict:
    """Set a temporary password and revoke existing sessions."""
    try:
        user = reset_user_password(username, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    audit_log(
        action=AuditAction.USER_PASSWORD_RESET,
        username=admin.username,
        ip_address=_client_ip(request),
        resource_type="user",
        resource_id=username,
    )
    return _public_user(user)


@router.delete("/{username}")
async def users_delete(
    username: str,
    request: Request,
    admin: TokenData = Depends(require_admin),
) -> dict:
    """Delete a user account and revoke its sessions."""
    if username == admin.username:
        raise HTTPException(status_code=400, detail="cannot delete yourself")
    try:
        delete_user(username)
    except ValueError as exc:
        status = 400 if "last active admin" in str(exc) else 404
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    audit_log(
        action=AuditAction.USER_DELETE,
        username=admin.username,
        ip_address=_client_ip(request),
        resource_type="user",
        resource_id=username,
    )
    return {"status": "deleted", "username": username}
