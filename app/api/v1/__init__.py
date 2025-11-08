"""
API v1 module initialization.

This module exports API v1 routers and endpoints.
"""

from fastapi import APIRouter
from app.api.v1 import users, interactions, recommendations, health

# Create main API router (no prefix here - will be added in main.py)
api_router = APIRouter()

# Include sub-routers
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(interactions.router, prefix="/interactions", tags=["interactions"])
api_router.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
api_router.include_router(health.router, prefix="/health", tags=["health"])
