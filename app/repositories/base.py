"""
Base repository pattern implementation.

This module defines the abstract base repository and common patterns.
Design Rationale:
- Abstract base class defines the interface contract
- Generic type hints for type safety
- Separation of concerns between domain and persistence
- Repository pattern abstracts data access for testability
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=SQLModel)


class BaseRepository(Generic[T], ABC):
    """
    Abstract base repository implementing common CRUD operations.
    
    Design Rationale:
    - Generic repository pattern for type-safe data access
    - Abstract methods force concrete implementations
    - Async support for high-performance applications
    - Logging integration for debugging and monitoring
    
    Type Parameters:
        T: The SQLModel type this repository manages
    """
    
    def __init__(self, session: AsyncSession, model_class: type[T]):
        """
        Initialize the repository.
        
        Args:
            session: Async database session
            model_class: The SQLModel class this repository manages
        """
        self.session = session
        self.model_class = model_class
        self.logger = get_logger(f"{__name__}.{model_class.__name__}")
    
    @abstractmethod
    async def get_by_id(self, id: int) -> Optional[T]:
        """Get entity by primary key."""
        pass
    
    @abstractmethod
    async def get_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """Get all entities with pagination."""
        pass
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create a new entity."""
        pass
    
    @abstractmethod
    async def update(self, id: int, entity_data: Dict[str, Any]) -> Optional[T]:
        """Update an existing entity."""
        pass
    
    @abstractmethod
    async def delete(self, id: int) -> bool:
        """Delete an entity."""
        pass
    
    @abstractmethod
    async def count(self, **filters: Any) -> int:
        """Count entities matching filters."""
        pass


class SQLModelRepository(BaseRepository[T]):
    """
    Concrete implementation of BaseRepository using SQLModel.
    
    Design Rationale:
    - Implements common CRUD operations with SQLModel
    - Async session support for performance
    - Eager loading support for relationship optimization
    - Comprehensive error handling and logging
    """
    
    async def get_by_id(self, id: int, load_relationships: bool = True) -> Optional[T]:
        """
        Get entity by primary key with optional relationship loading.
        
        Args:
            id: Primary key value
            load_relationships: Whether to eagerly load relationships
            
        Returns:
            Entity instance or None if not found
        """
        try:
            query = select(self.model_class).where(self.model_class.id == id)
            
            if load_relationships:
                # Dynamically add selectinload for all relationships
                for relationship in self.model_class.__sqlmodel_relationships__.values():
                    query = query.options(selectinload(relationship.key))
            
            result = await self.session.execute(query)
            entity = result.scalar_one_or_none()
            
            if entity:
                self.logger.debug("Entity retrieved", entity_id=id, entity_type=self.model_class.__name__)
            else:
                self.logger.warning("Entity not found", entity_id=id, entity_type=self.model_class.__name__)
            
            return entity
            
        except Exception as e:
            self.logger.error(
                "Error retrieving entity",
                entity_id=id,
                entity_type=self.model_class.__name__,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        load_relationships: bool = True,
        **filters: Any
    ) -> List[T]:
        """
        Get all entities with pagination and optional filters.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            load_relationships: Whether to eagerly load relationships
            **filters: Additional filter criteria
            
        Returns:
            List of entity instances
        """
        try:
            query = select(self.model_class)
            
            # Apply filters
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    query = query.where(getattr(self.model_class, key) == value)
            
            if load_relationships:
                # Dynamically add selectinload for all relationships
                for relationship in self.model_class.__sqlmodel_relationships__.values():
                    query = query.options(selectinload(relationship.key))
            
            query = query.offset(skip).limit(limit)
            
            result = await self.session.execute(query)
            entities = result.scalars().all()
            
            self.logger.debug(
                "Entities retrieved",
                count=len(entities),
                skip=skip,
                limit=limit,
                entity_type=self.model_class.__name__
            )
            
            return list(entities)
            
        except Exception as e:
            self.logger.error(
                "Error retrieving entities",
                skip=skip,
                limit=limit,
                entity_type=self.model_class.__name__,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def create(self, entity: T) -> T:
        """
        Create a new entity.
        
        Args:
            entity: Entity instance to create
            
        Returns:
            Created entity with generated ID
        """
        try:
            self.session.add(entity)
            await self.session.commit()
            await self.session.refresh(entity)
            
            self.logger.info(
                "Entity created",
                entity_id=getattr(entity, 'id', None),
                entity_type=self.model_class.__name__
            )
            
            return entity
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(
                "Error creating entity",
                entity_type=self.model_class.__name__,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def update(self, id: int, entity_data: Dict[str, Any]) -> Optional[T]:
        """
        Update an existing entity.
        
        Args:
            id: Primary key of entity to update
            entity_data: Dictionary of fields to update
            
        Returns:
            Updated entity or None if not found
        """
        try:
            # Get the entity first
            entity = await self.get_by_id(id, load_relationships=False)
            if not entity:
                return None
            
            # Update fields
            for key, value in entity_data.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
            
            # Update timestamp if the model has updated_at field
            if hasattr(entity, 'updated_at'):
                entity.updated_at = datetime.utcnow()
            
            await self.session.commit()
            await self.session.refresh(entity)
            
            self.logger.info(
                "Entity updated",
                entity_id=id,
                entity_type=self.model_class.__name__,
                updated_fields=list(entity_data.keys())
            )
            
            return entity
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(
                "Error updating entity",
                entity_id=id,
                entity_type=self.model_class.__name__,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def delete(self, id: int) -> bool:
        """
        Delete an entity.
        
        Args:
            id: Primary key of entity to delete
            
        Returns:
            True if entity was deleted, False if not found
        """
        try:
            query = delete(self.model_class).where(self.model_class.id == id)
            result = await self.session.execute(query)
            await self.session.commit()
            
            deleted = result.rowcount > 0
            
            self.logger.info(
                "Entity deleted",
                entity_id=id,
                entity_type=self.model_class.__name__,
                deleted=deleted
            )
            
            return deleted
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(
                "Error deleting entity",
                entity_id=id,
                entity_type=self.model_class.__name__,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def count(self, **filters: Any) -> int:
        """
        Count entities matching filters.
        
        Args:
            **filters: Filter criteria
            
        Returns:
            Number of matching entities
        """
        try:
            query = select(func.count()).select_from(self.model_class)
            
            # Apply filters
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    query = query.where(getattr(self.model_class, key) == value)
            
            result = await self.session.execute(query)
            count = result.scalar()
            
            self.logger.debug(
                "Entity count",
                count=count,
                filters=filters,
                entity_type=self.model_class.__name__
            )
            
            return count
            
        except Exception as e:
            self.logger.error(
                "Error counting entities",
                filters=filters,
                entity_type=self.model_class.__name__,
                error=str(e),
                exc_info=True
            )
            raise
        