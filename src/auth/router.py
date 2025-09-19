"""
Authentication router for user registration, login, and token management.
"""

from fastapi import APIRouter, HTTPException, status, Depends, Request
from src.core.database import SessionDep
from src.auth.service import AuthService, get_auth_service
from src.auth.schemas import (
    UserCreate, UserLogin, UserResponse, TokenResponse, 
    TokenRefresh, PasswordChange, UserUpdate
)
from src.auth.dependencies import get_current_user, get_current_superuser
from src.auth.models import User
from src.core.utils import log_request, log_response, logger
from src.middleware.rate_limiting import auth_rate_limit, api_rate_limit, admin_rate_limit

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@auth_rate_limit()
async def register_user(
    user_data: UserCreate,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Register a new user."""
    log_request(request, {"username": user_data.username, "email": user_data.email})
    
    try:
        user = await auth_service.create_user(user_data)
        response = UserResponse(**user.to_dict())
        log_response(request, response.dict())
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
@auth_rate_limit()
async def login_user(
    login_data: UserLogin,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Login user and return access/refresh tokens."""
    log_request(request, {"username": login_data.username})
    
    try:
        # Authenticate user
        user = await auth_service.authenticate_user(login_data.username, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create tokens
        tokens = await auth_service.create_tokens(user)
        
        response = TokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type="bearer",
            expires_in=3600  # 1 hour
        )
        
        log_response(request, {"user_id": user.id, "username": user.username})
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=TokenResponse)
@auth_rate_limit()
async def refresh_token(
    token_data: TokenRefresh,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Refresh access token using refresh token."""
    log_request(request, {"token_type": "refresh"})
    
    try:
        # Verify refresh token
        user = await auth_service.verify_refresh_token(token_data.refresh_token)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create new tokens
        tokens = await auth_service.create_tokens(user)
        
        response = TokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type="bearer",
            expires_in=3600  # 1 hour
        )
        
        log_response(request, {"user_id": user.id, "username": user.username})
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout_user(
    token_data: TokenRefresh,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Logout user by revoking refresh token."""
    log_request(request, {"token_type": "logout"})
    
    try:
        await auth_service.revoke_refresh_token(token_data.refresh_token)
        
        response = {"message": "Successfully logged out"}
        log_response(request, response)
        return response
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information."""
    return UserResponse(**current_user.to_dict())


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    request: Request,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Update current user profile."""
    log_request(request, {"user_id": current_user.id})
    
    try:
        # Update user fields
        update_data = user_update.dict(exclude_unset=True)
        if not update_data:
            return UserResponse(**current_user.to_dict())
        
        # Update user in database
        from sqlalchemy import update
        await auth_service.session.execute(
            update(User)
            .where(User.id == current_user.id)
            .values(**update_data)
        )
        await auth_service.session.commit()
        
        # Get updated user
        updated_user = await auth_service.get_user_by_id(current_user.id)
        response = UserResponse(**updated_user.to_dict())
        
        log_response(request, response.dict())
        return response
        
    except Exception as e:
        logger.error(f"User update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User update failed"
        )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    request: Request,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Change user password."""
    log_request(request, {"user_id": current_user.id})
    
    try:
        # Verify current password
        if not auth_service.verify_password(password_data.current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password
        new_hashed_password = auth_service.get_password_hash(password_data.new_password)
        from sqlalchemy import update
        await auth_service.session.execute(
            update(User)
            .where(User.id == current_user.id)
            .values(hashed_password=new_hashed_password)
        )
        await auth_service.session.commit()
        
        # Revoke all user tokens
        await auth_service.revoke_all_user_tokens(current_user.id)
        
        response = {"message": "Password changed successfully"}
        log_response(request, response)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.get("/users", response_model=list[UserResponse])
@admin_rate_limit()
async def list_users(
    request: Request,
    current_user: User = Depends(get_current_superuser),
    auth_service: AuthService = Depends(get_auth_service)
):
    """List all users (superuser only)."""
    log_request(request, {"user_id": current_user.id})
    
    try:
        from sqlalchemy import select
        result = await auth_service.session.execute(select(User))
        users = result.scalars().all()
        
        response = [UserResponse(**user.to_dict()) for user in users]
        log_response(request, {"count": len(response)})
        return response
        
    except Exception as e:
        logger.error(f"List users error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    request: Request,
    current_user: User = Depends(get_current_superuser),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Delete a user (superuser only)."""
    log_request(request, {"user_id": current_user.id, "target_user_id": user_id})
    
    try:
        # Prevent self-deletion
        if user_id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        # Get target user
        target_user = await auth_service.get_user_by_id(user_id)
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Delete user
        from sqlalchemy import delete
        await auth_service.session.execute(
            delete(User).where(User.id == user_id)
        )
        await auth_service.session.commit()
        
        response = {"message": f"User {target_user.username} deleted successfully"}
        log_response(request, response)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )
