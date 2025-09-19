"""
Authentication service layer for user management and token operations.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from sqlalchemy import select, update, delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status, Depends
from src.core.database import SessionDep
from src.auth.models import User, RefreshToken
from src.auth.schemas import UserCreate, UserLogin, TokenRefresh
from src.auth.security import (
    verify_password, get_password_hash, create_access_token, 
    create_refresh_token, verify_token, generate_token_hash
)
from src.core.utils import logger
from src.core.exceptions import DatabaseError


class AuthService:
    """Authentication service for user management and token operations."""
    
    def __init__(self, session: SessionDep):
        """Initialize auth service with session dependency."""
        self.session = session
    
    # ========== USER MANAGEMENT METHODS ==========
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user."""
        try:
            # Check if user already exists
            existing_user = await self.get_user_by_username(user_data.username)
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already registered"
                )
            
            existing_email = await self.get_user_by_email(user_data.email)
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            # Create new user
            hashed_password = get_password_hash(user_data.password)
            user = User(
                username=user_data.username,
                email=user_data.email,
                hashed_password=hashed_password,
                is_active=True,
                is_superuser=False
            )
            
            self.session.add(user)
            await self.session.commit()
            await self.session.refresh(user)
            
            logger.info(f"User created: {user.username}")
            return user
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Database error creating user: {e}")
            raise DatabaseError(f"Failed to create user: {str(e)}")
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            result = await self.session.execute(
                select(User).where(User.username == username)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user by username: {e}")
            raise DatabaseError(f"Failed to get user: {str(e)}")
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            result = await self.session.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user by email: {e}")
            raise DatabaseError(f"Failed to get user: {str(e)}")
    
    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        try:
            result = await self.session.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user by ID: {e}")
            raise DatabaseError(f"Failed to get user: {str(e)}")
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username/email and password."""
        try:
            # Try username first, then email
            user = await self.get_user_by_username(username)
            if not user:
                user = await self.get_user_by_email(username)
            
            if not user:
                return None
            
            if not verify_password(password, user.hashed_password):
                return None
            
            if not user.is_active:
                return None
            
            # Update last login
            user.last_login = datetime.utcnow()
            await self.session.commit()
            
            return user
            
        except SQLAlchemyError as e:
            logger.error(f"Database error authenticating user: {e}")
            raise DatabaseError(f"Failed to authenticate user: {str(e)}")
    
    async def update_user_last_login(self, user_id: int) -> None:
        """Update user's last login timestamp."""
        try:
            await self.session.execute(
                update(User)
                .where(User.id == user_id)
                .values(last_login=datetime.utcnow())
            )
            await self.session.commit()
        except SQLAlchemyError as e:
            logger.error(f"Database error updating last login: {e}")
            raise DatabaseError(f"Failed to update last login: {str(e)}")
    
    # ========== TOKEN MANAGEMENT METHODS ==========
    
    async def create_tokens(self, user: User) -> Dict[str, str]:
        """Create access and refresh tokens for a user."""
        try:
            # Create access token
            access_token_data = {
                "sub": str(user.id),
                "username": user.username,
                "email": user.email,
                "is_superuser": user.is_superuser
            }
            access_token = create_access_token(access_token_data)
            
            # Create refresh token
            refresh_token_data = {
                "sub": str(user.id),
                "username": user.username
            }
            refresh_token = create_refresh_token(refresh_token_data)
            
            # Store refresh token in database
            await self.store_refresh_token(user.id, refresh_token)
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token
            }
            
        except Exception as e:
            logger.error(f"Error creating tokens: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create tokens"
            )
    
    async def store_refresh_token(self, user_id: int, refresh_token: str) -> None:
        """Store refresh token in database."""
        try:
            token_hash = generate_token_hash(refresh_token)
            expires_at = datetime.utcnow() + timedelta(days=30)  # 30 days
            
            refresh_token_record = RefreshToken(
                user_id=user_id,
                token_hash=token_hash,
                expires_at=expires_at
            )
            
            self.session.add(refresh_token_record)
            await self.session.commit()
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Database error storing refresh token: {e}")
            raise DatabaseError(f"Failed to store refresh token: {str(e)}")
    
    async def verify_refresh_token(self, refresh_token: str) -> Optional[User]:
        """Verify refresh token and return user."""
        try:
            # Verify JWT token
            payload = verify_token(refresh_token, "refresh")
            user_id = int(payload.get("sub"))
            
            # Check if token exists in database
            token_hash = generate_token_hash(refresh_token)
            result = await self.session.execute(
                select(RefreshToken).where(
                    RefreshToken.token_hash == token_hash,
                    RefreshToken.user_id == user_id,
                    RefreshToken.is_revoked == False,
                    RefreshToken.expires_at > datetime.utcnow()
                )
            )
            token_record = result.scalar_one_or_none()
            
            if not token_record:
                return None
            
            # Get user
            user = await self.get_user_by_id(user_id)
            if not user or not user.is_active:
                return None
            
            return user
            
        except Exception as e:
            logger.error(f"Error verifying refresh token: {e}")
            return None
    
    async def revoke_refresh_token(self, refresh_token: str) -> None:
        """Revoke a refresh token."""
        try:
            token_hash = generate_token_hash(refresh_token)
            await self.session.execute(
                update(RefreshToken)
                .where(RefreshToken.token_hash == token_hash)
                .values(is_revoked=True, revoked_at=datetime.utcnow())
            )
            await self.session.commit()
            
        except SQLAlchemyError as e:
            logger.error(f"Database error revoking refresh token: {e}")
            raise DatabaseError(f"Failed to revoke refresh token: {str(e)}")
    
    async def revoke_all_user_tokens(self, user_id: int) -> None:
        """Revoke all refresh tokens for a user."""
        try:
            await self.session.execute(
                update(RefreshToken)
                .where(RefreshToken.user_id == user_id)
                .values(is_revoked=True, revoked_at=datetime.utcnow())
            )
            await self.session.commit()
            
        except SQLAlchemyError as e:
            logger.error(f"Database error revoking user tokens: {e}")
            raise DatabaseError(f"Failed to revoke user tokens: {str(e)}")
    
    async def cleanup_expired_tokens(self) -> None:
        """Clean up expired refresh tokens."""
        try:
            await self.session.execute(
                delete(RefreshToken).where(
                    RefreshToken.expires_at < datetime.utcnow()
                )
            )
            await self.session.commit()
            
        except SQLAlchemyError as e:
            logger.error(f"Database error cleaning up tokens: {e}")
            raise DatabaseError(f"Failed to cleanup tokens: {str(e)}")


def get_auth_service(session: SessionDep) -> AuthService:
    """Get authentication service with session dependency."""
    return AuthService(session)
