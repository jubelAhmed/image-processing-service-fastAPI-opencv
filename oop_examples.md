
This document showcases **Object-Oriented Programming practices** from this FastAPI application.

## 📋 Table of Contents

1. [Strategy Pattern](#1-strategy-pattern)
2. [Abstract Base Class](#2-abstract-base-class)
3. [Factory Pattern](#3-factory-pattern)
4. [Dependency Injection](#4-dependency-injection)
5. [Exception Hierarchy](#5-exception-hierarchy)
6. [Template Method Pattern](#6-template-method-pattern)
7. [Service Layer Pattern](#7-service-layer-pattern)

---

## 1. Strategy Pattern

**File:** `src/facial/style_config.py`

```python
"""
Strategy Pattern - Multiple implementations of the same interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class RegionStyle:
    """Data class for region styling information."""
    stroke: str
    fill: str
    stroke_width: int = 2
    stroke_dasharray: Optional[str] = None
    font_size: int = 26
    text_color: str = "white"

class StyleConfig(ABC):
    """Abstract base class for style configuration strategies."""
    
    @abstractmethod
    def get_region_style(self, region_id: int) -> RegionStyle:
        """Get style configuration for a specific region."""
        pass
    
    @abstractmethod
    def get_default_style(self) -> RegionStyle:
        """Get default style configuration."""
        pass

class DefaultStyleConfig(StyleConfig):
    """Default style configuration with purple theme."""
    
    def __init__(self):
        self.region_styles = {
            1: RegionStyle(stroke="#9D57A7", fill="rgba(161, 106, 169, 0.5)"),
            2: RegionStyle(stroke="#A16AA9", fill="rgba(161, 106, 169, 0.5)"),
            # ... more regions
        }
        self.default_style = RegionStyle(stroke="#000000", fill="rgba(0,0,0,0.2)")
    
    def get_region_style(self, region_id: int) -> RegionStyle:
        return self.region_styles.get(region_id, self.default_style)
    
    def get_default_style(self) -> RegionStyle:
        return self.default_style

class ColorfulStyleConfig(StyleConfig):
    """Colorful style configuration with different colors for each region."""
    
    def __init__(self):
        self.region_styles = {
            1: RegionStyle(stroke="#FF6B6B", fill="rgba(255, 107, 107, 0.3)"),  # Red
            2: RegionStyle(stroke="#4ECDC4", fill="rgba(78, 205, 196, 0.3)"),   # Teal
            # ... more regions
        }
        self.default_style = RegionStyle(stroke="#95A5A6", fill="rgba(149, 165, 166, 0.3)")
    
    def get_region_style(self, region_id: int) -> RegionStyle:
        return self.region_styles.get(region_id, self.default_style)
    
    def get_default_style(self) -> RegionStyle:
        return self.default_style

class StyleConfigFactory:
    """Factory for creating style configurations."""
    
    @staticmethod
    def create_style_config(style_type: str = "default") -> StyleConfig:
        """Create a style configuration based on type."""
        configs = {
            "default": DefaultStyleConfig,
            "colorful": ColorfulStyleConfig,
            "minimal": MinimalStyleConfig
        }
        
        config_class = configs.get(style_type.lower())
        if not config_class:
            raise ValueError(f"Unknown style type: {style_type}")
        
        return config_class()
```

**OOP Principles Demonstrated:**
- ✅ **Strategy Pattern** - Multiple style implementations
- ✅ **Abstract Base Class** - Common interface
- ✅ **Factory Pattern** - Creation of style configs
- ✅ **Open/Closed Principle** - Easy to add new styles
- ✅ **Data Classes** - Modern Python feature

---

## 2. Abstract Base Class

**File:** `src/facial/generators/output_generator.py`

```python
"""
Abstract Base Class - Template Method Pattern
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from src.facial.face_schema import MaskContours

class OutputGenerator(ABC):
    """Abstract base class for different output formats."""
    
    @abstractmethod
    def generate(self, image_shape: Tuple[int, int], regions: MaskContours, 
                 processed_image: Optional[np.ndarray] = None) -> str:
        """
        Generate output in the specific format.
        
        Args:
            image_shape: Tuple of (height, width)
            regions: Dictionary of region contours
            processed_image: Optional processed image array
            
        Returns:
            Base64-encoded output string
        """
        pass
```

**OOP Principles Demonstrated:**
- ✅ **Abstract Base Class** - Cannot be instantiated
- ✅ **Template Method** - Defines the interface
- ✅ **Polymorphism** - Different implementations
- ✅ **Type Hints** - Modern Python practices

---

## 3. Factory Pattern

**File:** `src/facial/generator_factory.py`

```python
"""
Factory Pattern - Creating objects without specifying their exact class
"""

from typing import Dict, Type, Optional
from src.facial.generators.output_generator import OutputGenerator
from src.facial.generators.svg_generator import SVGGenerator
from src.facial.generators.png_generator import PNGGenerator
from src.facial.generators.json_generator import JSONGenerator
from src.facial.style_config import StyleConfig, StyleConfigFactory

class GeneratorFactory:
    """Factory for creating output generators with different configurations."""
    
    _generators: Dict[str, Type[OutputGenerator]] = {
        'svg': SVGGenerator,
        'png': PNGGenerator,
        'json': JSONGenerator
    }
    
    @classmethod
    def create_generator(cls, generator_type: str, style_config: Optional[StyleConfig] = None) -> OutputGenerator:
        """
        Create a generator instance by type with optional style configuration.
        
        Args:
            generator_type: Type of generator ('svg', 'png', 'json')
            style_config: Optional style configuration
            
        Returns:
            Generator instance
            
        Raises:
            InvalidInputError: If generator type is not supported
        """
        generator_class = cls._generators.get(generator_type.lower())
        
        if not generator_class:
            raise InvalidInputError(f"Unsupported generator type: {generator_type}. "
                                  f"Supported types: {list(cls._generators.keys())}")
        
        # Only SVG generator uses style configuration
        if generator_type.lower() == 'svg' and style_config:
            return generator_class(style_config)
        else:
            return generator_class()
    
    @classmethod
    def get_available_generators(cls) -> Dict[str, Type[OutputGenerator]]:
        """Get dictionary of available generator types."""
        return cls._generators.copy()
    
    @classmethod
    def register_generator(cls, name: str, generator_class: Type[OutputGenerator]) -> None:
        """
        Register a new generator type.
        
        Args:
            name: Name of the generator type
            generator_class: Generator class to register
        """
        if not issubclass(generator_class, OutputGenerator):
            raise InvalidInputError(f"Generator class must inherit from OutputGenerator")
        
        cls._generators[name.lower()] = generator_class
```

**OOP Principles Demonstrated:**
- ✅ **Factory Pattern** - Centralized object creation
- ✅ **Class Methods** - Factory methods
- ✅ **Registry Pattern** - Dynamic registration
- ✅ **Type Safety** - Runtime type checking
- ✅ **Extensibility** - Easy to add new generators

---

## 4. Dependency Injection

**File:** `src/facial/facial_segmentation_processor.py`

```python
"""
Dependency Injection - Constructor injection pattern
"""

from typing import Dict, List, Tuple, Optional
from src.facial.facial_processing.face_segmentation_config import SegmentationConfig
from src.facial.image_generator import ImageGenerator
from src.facial.face_schema import LandmarkPoint
from src.core.exceptions import NoFaceDetectedError, InvalidImageError, ProcessingError

class FacialSegmentationProcessor:
    """Advanced face region subdivider with dependency injection."""
    
    def __init__(self, config: SegmentationConfig = None, image_generator: ImageGenerator = None):
        """
        Initialize processor with dependency injection.
        
        Args:
            config: Segmentation configuration (defaults to SegmentationConfig)
            image_generator: Image generator instance (defaults to new ImageGenerator)
        """
        self.config = config or SegmentationConfig()
        self.image_generator = image_generator or ImageGenerator()
    
    async def process_image(self, image_base64: str, segmentation_map_base64: str, 
                          landmarks: List[LandmarkPoint]) -> Tuple[str, Dict]:
        """
        Process image with improved error handling and dependency injection.
        
        Args:
            image_base64: Base64 encoded image
            segmentation_map_base64: Base64 encoded segmentation map
            landmarks: List of facial landmark points
            
        Returns:
            Tuple of (generated_output, contours)
            
        Raises:
            NoFaceDetectedError: If no face is detected
            InvalidImageError: If image data is invalid
            ProcessingError: If processing fails
        """
        try:
            # Decode input data
            image = decode_image(image_base64)
            segmentation_map = decode_segmentation_map(segmentation_map_base64)
            
            # Validate landmarks
            if not landmarks or len(landmarks) == 0:
                raise NoFaceDetectedError("No face detected in the image")
        
            # Process face regions
            image_shape, contours, processed_image = self.process_face_regions(
                image, segmentation_map, landmarks
            )
            
            # Generate output using injected generator
            output_base64 = self.image_generator.create(image_shape, contours, processed_image)
            
            return output_base64, contours
            
        except (NoFaceDetectedError, InvalidImageError):
            raise
        except Exception as e:
            raise ProcessingError(f"Image processing failed: {str(e)}") from e
```

**OOP Principles Demonstrated:**
- ✅ **Dependency Injection** - Constructor injection
- ✅ **Interface Segregation** - Clean dependencies
- ✅ **Error Handling** - Custom exception hierarchy
- ✅ **Async/Await** - Modern Python patterns
- ✅ **Type Hints** - Full type safety

---

## 5. Exception Hierarchy

**File:** `src/core/exceptions.py`

```python
"""
Exception Hierarchy - Structured error handling
"""

from fastapi import HTTPException, status

class BaseAPIException(HTTPException):
    """Base API exception with error code support."""
    
    def __init__(self, detail: str, error_code: str = None, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(
            status_code=status_code,
            detail={
                "detail": detail,
                "error_code": error_code
            }
        )

class ValidationException(BaseAPIException):
    """Raised when input validation fails."""
    
    def __init__(self, detail: str = "Validation error"):
        super().__init__(detail, "VALIDATION_ERROR")

class NotFoundException(BaseAPIException):
    """Raised when a resource is not found."""
    
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(detail, "NOT_FOUND", status.HTTP_404_NOT_FOUND)

class UnauthorizedException(BaseAPIException):
    """Raised when authentication is required."""
    
    def __init__(self, detail: str = "Unauthorized"):
        super().__init__(detail, "UNAUTHORIZED", status.HTTP_401_UNAUTHORIZED)

class InternalServerException(BaseAPIException):
    """Raised when an internal server error occurs."""
    
    def __init__(self, detail: str = "Internal server error"):
        super().__init__(detail, "INTERNAL_SERVER_ERROR", status.HTTP_500_INTERNAL_SERVER_ERROR)

class DatabaseError(BaseAPIException):
    """Raised when a database error occurs."""
    
    def __init__(self, detail: str = "Database error"):
        super().__init__(detail, "DATABASE_ERROR", status.HTTP_500_INTERNAL_SERVER_ERROR)
```

**OOP Principles Demonstrated:**
- ✅ **Inheritance** - Base exception class
- ✅ **Polymorphism** - Same interface, different behavior
- ✅ **Consistent Interface** - All exceptions follow same pattern
- ✅ **HTTP Integration** - Properly integrated with FastAPI
- ✅ **Error Codes** - Structured error responses

---

## 6. Template Method Pattern

**File:** `src/facial/generators/svg_generator.py`

```python
"""
Template Method Pattern - Concrete implementation of abstract base class
"""

import base64
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
from src.facial.generators.output_generator import OutputGenerator
from src.facial.style_config import StyleConfig, DefaultStyleConfig
from src.facial.face_schema import MaskContours
from src.facial.exceptions import ProcessingError

class SVGGenerator(OutputGenerator):
    """SVG output generator for contours with style configuration."""

    def __init__(self, style_config: StyleConfig = None):
        """
        Initialize SVG generator with style configuration.
        
        Args:
            style_config: Style configuration instance (defaults to DefaultStyleConfig)
        """
        self.style_config = style_config or DefaultStyleConfig()

    def generate(self, image_shape: Tuple[int, int], contours: MaskContours, 
                 processed_image: Optional[np.ndarray] = None) -> str:
        """
        Generate SVG from image shape and contours list.
        
        Args:
            image_shape: Tuple of (height, width)
            contours: Dict where each key contains contour points for that region ID
            processed_image: Optional numpy array of the processed image to use as background
        
        Returns:
            Base64-encoded SVG string
            
        Raises:
            ProcessingError: If SVG generation fails
        """
        try:
            svg_root = self._create_svg_root(image_shape)
            
            # Add background image if provided
            if processed_image is not None:
                self._add_background_image(svg_root, processed_image, image_shape)
            
            self._add_regions_to_svg(svg_root, contours)
            return self._encode_svg(svg_root)
            
        except Exception as e:
            raise ProcessingError(f"Failed to generate SVG: {str(e)}") from e

    def _create_svg_root(self, image_shape: Tuple[int, int]) -> ET.Element:
        """Create the root SVG element with proper dimensions."""
        return ET.Element("svg", {
            "width": str(image_shape[1]),
            "height": str(image_shape[0]),
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"0 0 {image_shape[1]} {image_shape[0]}"
        })

    def _add_background_image(self, svg_root: ET.Element, processed_image: np.ndarray, 
                            image_shape: Tuple[int, int]) -> None:
        """Add the processed image as background to the SVG."""
        # Encode the processed image to PNG format
        _, img_encoded = cv2.imencode('.png', processed_image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # Create image element and add it as the first child (background)
        ET.SubElement(svg_root, "image", {
            "href": f"data:image/png;base64,{img_base64}",
            "x": "0",
            "y": "0",
            "width": str(image_shape[1]),
            "height": str(image_shape[0]),
            "class": "background-image"
        })

    def _add_regions_to_svg(self, svg_root: ET.Element, contours: MaskContours) -> None:
        """Add all contours with their labels to the SVG."""
        for region_id, contour in contours.items():
            if not contour or len(contour) < 3:
                continue
                
            # Get style for this region
            style = self.style_config.get_region_style(region_id)
            
            # Create path element
            path_element = self._create_path_element(contour, style)
            svg_root.append(path_element)
            
            # Add region label
            self._add_region_label(svg_root, contour, region_id, style)

    def _create_path_element(self, contour: List[List[int]], style: RegionStyle) -> ET.Element:
        """Create SVG path element from contour points."""
        # Convert contour to SVG path string
        path_data = f"M {contour[0][0]} {contour[0][1]}"
        for point in contour[1:]:
            path_data += f" L {point[0]} {point[1]}"
        path_data += " Z"
        
        return ET.Element("path", {
            "d": path_data,
            "stroke": style.stroke,
            "fill": style.fill,
            "stroke-width": str(style.stroke_width),
            "stroke-dasharray": style.stroke_dasharray or "none"
        })

    def _add_region_label(self, svg_root: ET.Element, contour: List[List[int]], 
                         region_id: int, style: RegionStyle) -> None:
        """Add region number label at the centroid of the contour."""
        try:
            # Calculate centroid
            x_coords = [point[0] for point in contour]
            y_coords = [point[1] for point in contour]
            cx = sum(x_coords) / len(x_coords)
            cy = sum(y_coords) / len(y_coords)
            
            # Create text element
            text_element = ET.Element("text", {
                "x": str(int(cx)),
                "y": str(int(cy)),
                "text-anchor": "middle",
                "dominant-baseline": "middle",
                "font-size": str(style.font_size),
                "fill": style.text_color,
                "font-family": "Arial, sans-serif",
                "font-weight": "bold"
            })
            text_element.text = str(region_id)
            
            svg_root.append(text_element)
            
        except Exception:
            # If label drawing fails, continue without it
            pass

    def _encode_svg(self, svg_root: ET.Element) -> str:
        """Encode SVG element to base64 string."""
        svg_string = ET.tostring(svg_root, encoding='unicode')
        return base64.b64encode(svg_string.encode("utf-8")).decode("utf-8")
```

**OOP Principles Demonstrated:**
- ✅ **Template Method** - Implements abstract base class
- ✅ **Dependency Injection** - StyleConfig injection
- ✅ **Private Methods** - Encapsulation with `_` prefix
- ✅ **Error Handling** - Custom exceptions
- ✅ **Single Responsibility** - Only handles SVG generation

---

## 7. Service Layer Pattern

**File:** `src/facial/service.py`

```python
"""
Service Layer Pattern - Repository pattern for data access
"""

from typing import Dict, Any, Optional, List
import json
import hashlib
from datetime import datetime, timedelta
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import SessionDep
from src.facial.models import Cache, Job, ProcessingMetrics
from src.facial.schemas import LandmarkPoint
from src.core.utils import logger
from src.facial.exceptions import DatabaseException

class DatabaseService:
    """Database service layer with modern session dependency pattern."""
    
    def __init__(self, session: SessionDep):
        """Initialize database service with session dependency."""
        self.session = session
    
    async def store_job_status(
        self, 
        job_id: str, 
        status: str, 
        cache_id: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Store or update job status in database."""
        try:
            # Check if job exists
            result = await self.session.execute(
                select(Job).where(Job.job_id == job_id)
            )
            job = result.scalar_one_or_none()
            
            if job:
                # Update existing job
                await self.session.execute(
                    update(Job)
                    .where(Job.job_id == job_id)
                    .values(
                        status=status,
                        cache_id=cache_id,
                        error_message=error_message,
                        updated_at=datetime.utcnow()
                    )
                )
            else:
                # Create new job
                new_job = Job(
                    job_id=job_id,
                    status=status,
                    cache_id=cache_id,
                    error_message=error_message
                )
                self.session.add(new_job)
            
            await self.session.commit()
            logger.info(f"Job status updated: {job_id} -> {status}")
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Database error storing job status: {e}")
            raise DatabaseException(f"Failed to store job status: {str(e)}")
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status from database."""
        try:
            result = await self.session.execute(
                select(Job).where(Job.job_id == job_id)
            )
            job = result.scalar_one_or_none()
            
            if not job:
                return None
            
            return {
                "job_id": job.job_id,
                "status": job.status,
                "cache_id": job.cache_id,
                "error_message": job.error_message,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "updated_at": job.updated_at.isoformat() if job.updated_at else None
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting job status: {e}")
            raise DatabaseException(f"Failed to get job status: {str(e)}")
    
    async def store_processing_metrics(
        self,
        job_id: str,
        processing_time: float,
        image_size: tuple,
        contour_count: int
    ) -> None:
        """Store processing metrics for analysis."""
        try:
            metrics = ProcessingMetrics(
                job_id=job_id,
                processing_time_ms=processing_time * 1000,  # Convert to milliseconds
                image_width=image_size[1],
                image_height=image_size[0],
                contour_count=contour_count
            )
            
            self.session.add(metrics)
            await self.session.commit()
            
            logger.info(f"Processing metrics stored for job: {job_id}")
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Database error storing metrics: {e}")
            raise DatabaseException(f"Failed to store processing metrics: {str(e)}")
```

**OOP Principles Demonstrated:**
- ✅ **Service Layer** - Clean data access layer
- ✅ **Dependency Injection** - Session injection
- ✅ **Async/Await** - Modern async patterns
- ✅ **Error Handling** - Database exception handling
- ✅ **Type Hints** - Full type safety
- ✅ **Context Managers** - Proper session management

---

## 🎯 Summary of OOP Principles

### **SOLID Principles Demonstrated:**

1. **S - Single Responsibility Principle**
   - Each class has one reason to change
   - SVGGenerator only handles SVG creation
   - DatabaseService only handles data access

2. **O - Open/Closed Principle**
   - Easy to add new style configs without modifying existing code
   - New generators can be registered without changing factory

3. **L - Liskov Substitution Principle**
   - All generators can be used interchangeably
   - All style configs implement the same interface

4. **I - Interface Segregation Principle**
   - Clean, focused interfaces
   - No forced dependencies on unused methods

5. **D - Dependency Inversion Principle**
   - Depends on abstractions (StyleConfig, OutputGenerator)
   - Not on concrete implementations

### **Design Patterns Used:**

- ✅ **Strategy Pattern** - Multiple style implementations
- ✅ **Factory Pattern** - Object creation
- ✅ **Template Method** - Abstract base class
- ✅ **Dependency Injection** - Constructor injection
- ✅ **Repository Pattern** - Data access layer
- ✅ **Exception Hierarchy** - Structured error handling

### **Modern Python Features:**

- ✅ **Type Hints** - Full type safety
- ✅ **Data Classes** - Clean data structures
- ✅ **Async/Await** - Asynchronous programming
- ✅ **Abstract Base Classes** - Interface definition
- ✅ **Context Managers** - Resource management


