# Python OOP Examples - Core Concepts & Implementation

This document explains **Object-Oriented Programming fundamentals** from this FastAPI facial processing application.

**Created by:** Jubel
**Project:** FastAPI Facial Processing Application  

## Table of Contents

1. [Core OOP Concepts](#1-core-oop-concepts)
2. [OOP Implementation Examples](#2-oop-implementation-examples)
3. [Design Patterns in OOP](#3-design-patterns-in-oop)
4. [Key OOP Concepts Explained](#4-key-oop-concepts-explained)

---

## 1. Core OOP Concepts

### **1.1 Classes and Objects**

**Class:** A blueprint for creating objects
**Object:** An instance of a class

```python
# Class Definition
class Person:
    def __init__(self, name, age):
        self.name = name  # Instance variable
        self.age = age    # Instance variable
    
    def greet(self):      # Instance method
        return f"Hello, I'm {self.name}"

# Object Creation
person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

print(person1.greet())  # "Hello, I'm Alice"
print(person2.greet())  # "Hello, I'm Bob"
```

### **1.2 Inheritance**

**Inheritance:** A class can inherit properties and methods from another class

```python
# Parent Class (Base Class)
class BaseAPIException(HTTPException):
    def __init__(self, detail: str, error_code: str = None, 
                 status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(
            status_code=status_code,
            detail={
                "detail": detail,
                "error_code": error_code
            }
        )

# Child Class (Derived Class)
class ValidationException(BaseAPIException):
    def __init__(self, detail: str = "Validation error"):
        super().__init__(detail, "VALIDATION_ERROR")

# Another Child Class
class NotFoundException(BaseAPIException):
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(detail, "NOT_FOUND", status.HTTP_404_NOT_FOUND)

# Usage
try:
    raise ValidationException("Invalid image format")
except ValidationException as e:
    print(e.detail)  # {"detail": "Invalid image format", "error_code": "VALIDATION_ERROR"}
```

### **1.3 Polymorphism**

**Polymorphism:** Same interface, different implementations

```python
# Different classes with same method name
class DefaultStyleConfig(StyleConfig):
    def get_region_style(self, region_id: int) -> RegionStyle:
        return RegionStyle(stroke="#9D57A7", fill="rgba(161, 106, 169, 0.5)")

class ColorfulStyleConfig(StyleConfig):
    def get_region_style(self, region_id: int) -> RegionStyle:
        return RegionStyle(stroke="#FF6B6B", fill="rgba(255, 107, 107, 0.3)")

class MinimalStyleConfig(StyleConfig):
    def get_region_style(self, region_id: int) -> RegionStyle:
        return RegionStyle(stroke="#000000", fill="rgba(0,0,0,0.1)")

# Polymorphic behavior
style_configs = [
    DefaultStyleConfig(),
    ColorfulStyleConfig(), 
    MinimalStyleConfig()
]

for config in style_configs:
    style = config.get_region_style(1)  # Same method call
    print(f"Style: {style.stroke}")  # Different colors, same interface
```

### **1.4 Encapsulation**

**Encapsulation:** Hiding internal details and controlling access

```python
class FacialSegmentationProcessor:
    def __init__(self, config: SegmentationConfig = None):
        # Private attributes - Internal implementation details
        self._config = config or SegmentationConfig()
        self._image_generator = ImageGenerator()
        self.__processing_cache = {}  # Double underscore = very private
    
    def process_image(self, image_base64: str, landmarks: List[LandmarkPoint]) -> str:
        """Public method - External interface"""
        try:
            # Use private method internally
            validated_landmarks = self._validate_landmarks(landmarks)
            return self._process_face_regions(image_base64, validated_landmarks)
        except Exception as e:
            raise ProcessingError(f"Processing failed: {str(e)}")
    
    def _validate_landmarks(self, landmarks: List[LandmarkPoint]) -> List[LandmarkPoint]:
        """Private method - Internal implementation"""
        if not landmarks or len(landmarks) == 0:
            raise NoFaceDetectedError("No face detected")
        return landmarks
    
    def __clear_cache(self):  # Very private method
        """Very private method - Internal cleanup"""
        self.__processing_cache.clear()

# Usage
processor = FacialSegmentationProcessor()
result = processor.process_image(image_data, landmarks)  # Public method
# processor._validate_landmarks()  # Error! Private method
# processor.__processing_cache  # Error! Very private attribute
```

### **1.5 Abstraction**

**Abstraction:** Hiding complex implementation details

```python
from abc import ABC, abstractmethod

# Abstract Base Class
class OutputGenerator(ABC):
    """Abstract interface for different output formats"""
    
    @abstractmethod
    def generate(self, image_shape: Tuple[int, int], regions: Dict, 
                 processed_image: Optional[np.ndarray] = None) -> str:
        """Abstract method - Must be implemented by subclasses"""
        pass

# Concrete Implementation
class SVGGenerator(OutputGenerator):
    def __init__(self, style_config: StyleConfig = None):
        self.style_config = style_config or DefaultStyleConfig()
    
    def generate(self, image_shape: Tuple[int, int], regions: Dict, 
                 processed_image: Optional[np.ndarray] = None) -> str:
        """Concrete implementation - SVG generation logic"""
        # Complex SVG generation implementation...
        return "base64_encoded_svg"

# Another Concrete Implementation
class PNGGenerator(OutputGenerator):
    def generate(self, image_shape: Tuple[int, int], regions: Dict, 
                 processed_image: Optional[np.ndarray] = None) -> str:
        """Concrete implementation - PNG generation logic"""
        # Complex PNG generation implementation...
        return "base64_encoded_png"

# Usage
generators = [SVGGenerator(), PNGGenerator()]
for generator in generators:
    output = generator.generate((100, 100), regions)  # Same interface
    print(f"Generated: {type(generator).__name__}")  # Different implementations
```

---

## 2. OOP Implementation Examples

### **2.1 Strategy Pattern with OOP**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class RegionStyle:
    """Data class - Modern Python feature for clean data structures"""
    stroke: str
    fill: str
    stroke_width: int = 2

class StyleConfig(ABC):
    """Abstract Base Class - Defines interface"""
    
    @abstractmethod
    def get_region_style(self, region_id: int) -> RegionStyle:
        """Abstract method - Must be implemented by subclasses"""
        pass

class DefaultStyleConfig(StyleConfig):
    """Concrete Class - Implements the interface"""
    
    def __init__(self):
        # Encapsulation - Private data
        self._region_styles = {
            1: RegionStyle(stroke="#9D57A7", fill="rgba(161, 106, 169, 0.5)"),
            2: RegionStyle(stroke="#A16AA9", fill="rgba(161, 106, 169, 0.5)"),
        }
    
    def get_region_style(self, region_id: int) -> RegionStyle:
        """Polymorphism - Same method, different behavior"""
        return self._region_styles.get(region_id, self._get_default_style())
    
    def _get_default_style(self) -> RegionStyle:
        """Private method - Encapsulation"""
        return RegionStyle(stroke="#000000", fill="rgba(0,0,0,0.2)")

class ColorfulStyleConfig(StyleConfig):
    """Another Concrete Class - Different implementation"""
    
    def __init__(self):
        self._region_styles = {
            1: RegionStyle(stroke="#FF6B6B", fill="rgba(255, 107, 107, 0.3)"),
            2: RegionStyle(stroke="#4ECDC4", fill="rgba(78, 205, 196, 0.3)"),
        }
    
    def get_region_style(self, region_id: int) -> RegionStyle:
        """Polymorphism - Same method, different behavior"""
        return self._region_styles.get(region_id, self._get_default_style())
    
    def _get_default_style(self) -> RegionStyle:
        """Private method - Encapsulation"""
        return RegionStyle(stroke="#95A5A6", fill="rgba(149, 165, 166, 0.3)")

# Factory Pattern - Object creation
class StyleConfigFactory:
    """Factory Class - Creates objects without specifying exact class"""
    
    @staticmethod
    def create_style_config(style_type: str = "default") -> StyleConfig:
        """Static method - Class method, no instance needed"""
        configs = {
            "default": DefaultStyleConfig,
            "colorful": ColorfulStyleConfig,
        }
        
        config_class = configs.get(style_type.lower())
        if not config_class:
            raise ValueError(f"Unknown style type: {style_type}")
        
        return config_class()  # Polymorphism - Returns different types

# Usage - Demonstrates OOP principles
def demonstrate_oop():
    # Polymorphism - Same interface, different implementations
    default_config = StyleConfigFactory.create_style_config("default")
    colorful_config = StyleConfigFactory.create_style_config("colorful")
    
    # Both objects have same method but different behavior
    style1 = default_config.get_region_style(1)
    style2 = colorful_config.get_region_style(1)
    
    print(f"Default style: {style1.stroke}")  # "#9D57A7"
    print(f"Colorful style: {style2.stroke}")  # "#FF6B6B"
```

### **2.2 Factory Pattern with OOP**

```python
from typing import Dict, Type, Optional
from src.facial.generators.output_generator import OutputGenerator

class GeneratorFactory:
    """Factory Class - Centralized object creation"""
    
    # Class variable - Shared by all instances
    _generators: Dict[str, Type[OutputGenerator]] = {
        'svg': SVGGenerator,
        'png': PNGGenerator,
        'json': JSONGenerator
    }
    
    @classmethod
    def create_generator(cls, generator_type: str) -> OutputGenerator:
        """Class method - Can access class variables"""
        generator_class = cls._generators.get(generator_type.lower())
        
        if not generator_class:
            raise ValueError(f"Unsupported generator type: {generator_type}")
        
        return generator_class()  # Polymorphism - Returns different types
    
    @classmethod
    def register_generator(cls, name: str, generator_class: Type[OutputGenerator]) -> None:
        """Class method - Modifies class state"""
        if not issubclass(generator_class, OutputGenerator):
            raise ValueError(f"Generator class must inherit from OutputGenerator")
        
        cls._generators[name.lower()] = generator_class

# Usage - Demonstrates OOP principles
def demonstrate_factory():
    # Polymorphism - Same method, different return types
    svg_gen = GeneratorFactory.create_generator("svg")
    png_gen = GeneratorFactory.create_generator("png")
    
    # Both objects implement same interface
    print(type(svg_gen))  # <class 'SVGGenerator'>
    print(type(png_gen))  # <class 'PNGGenerator'>
    
    # Both have same methods (polymorphism)
    # svg_gen.generate(...) and png_gen.generate(...) work the same way
```

### **2.3 Abstract Base Class with OOP**

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional

class OutputGenerator(ABC):
    """Abstract Base Class - Cannot be instantiated directly"""
    
    @abstractmethod
    def generate(self, image_shape: Tuple[int, int], regions: Dict, 
                 processed_image: Optional[np.ndarray] = None) -> str:
        """Abstract method - Must be implemented by subclasses"""
        pass

class SVGGenerator(OutputGenerator):
    """Concrete Class - Implements abstract methods"""
    
    def __init__(self, style_config: StyleConfig = None):
        # Composition - Has-a relationship
        self.style_config = style_config or DefaultStyleConfig()
    
    def generate(self, image_shape: Tuple[int, int], regions: Dict, 
                 processed_image: Optional[np.ndarray] = None) -> str:
        """Concrete implementation - Must implement abstract method"""
        # Implementation details...
        return "svg_output"

class PNGGenerator(OutputGenerator):
    """Another Concrete Class - Different implementation"""
    
    def generate(self, image_shape: Tuple[int, int], regions: Dict, 
                 processed_image: Optional[np.ndarray] = None) -> str:
        """Concrete implementation - Different behavior"""
        # Implementation details...
        return "png_output"

# Usage - Demonstrates OOP principles
def demonstrate_abc():
    # Cannot instantiate abstract class
    # generator = OutputGenerator()  # Error!
    
    # Can instantiate concrete classes
    svg_gen = SVGGenerator()
    png_gen = PNGGenerator()
    
    # Polymorphism - Same interface, different implementations
    generators = [svg_gen, png_gen]
    for gen in generators:
        result = gen.generate((100, 100), {})  # Same method call
        print(f"Generated: {result}")
```

### **2.4 Dependency Injection with OOP**

```python
class FacialSegmentationProcessor:
    """Class with dependency injection"""
    
    def __init__(self, config: SegmentationConfig = None, 
                 image_generator: ImageGenerator = None):
        """
        Constructor injection - Dependencies passed in constructor
        
        Args:
            config: Segmentation configuration (dependency)
            image_generator: Image generator (dependency)
        """
        # Composition - Has-a relationship
        self.config = config or SegmentationConfig()
        self.image_generator = image_generator or ImageGenerator()
    
    async def process_image(self, image_base64: str, 
                          segmentation_map_base64: str, 
                          landmarks: List[LandmarkPoint]) -> Tuple[str, Dict]:
        """Method that uses injected dependencies"""
        try:
            # Use injected config
            if not self.config.validate_landmarks(landmarks):
                raise NoFaceDetectedError("Invalid landmarks")
            
            # Process image
            image_shape, contours, processed_image = self.process_face_regions(
                image_base64, segmentation_map_base64, landmarks
            )
            
            # Use injected image generator
            output_base64 = self.image_generator.create(
                image_shape, contours, processed_image
            )
            
            return output_base64, contours
            
        except Exception as e:
            raise ProcessingError(f"Processing failed: {str(e)}") from e

# Usage - Demonstrates OOP principles
def demonstrate_dependency_injection():
    # Create dependencies
    config = SegmentationConfig()
    image_gen = ImageGenerator()
    
    # Inject dependencies into processor
    processor = FacialSegmentationProcessor(config, image_gen)
    
    # Processor uses injected dependencies
    # This makes testing easier and follows dependency inversion principle
```

### **2.5 Exception Hierarchy with OOP**

```python
from fastapi import HTTPException, status

class BaseAPIException(HTTPException):
    """Base Exception Class - Inheritance"""
    
    def __init__(self, detail: str, error_code: str = None, 
                 status_code: int = status.HTTP_400_BAD_REQUEST):
        # Call parent constructor
        super().__init__(
            status_code=status_code,
            detail={
                "detail": detail,
                "error_code": error_code
            }
        )

class ValidationException(BaseAPIException):
    """Child Exception Class - Inherits from BaseAPIException"""
    
    def __init__(self, detail: str = "Validation error"):
        # Call parent constructor with specific values
        super().__init__(detail, "VALIDATION_ERROR")

class NotFoundException(BaseAPIException):
    """Another Child Exception Class"""
    
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(detail, "NOT_FOUND", status.HTTP_404_NOT_FOUND)

class UnauthorizedException(BaseAPIException):
    """Another Child Exception Class"""
    
    def __init__(self, detail: str = "Unauthorized"):
        super().__init__(detail, "UNAUTHORIZED", status.HTTP_401_UNAUTHORIZED)

# Usage - Demonstrates OOP principles
def demonstrate_exception_hierarchy():
    try:
        # All exceptions have same interface (polymorphism)
        raise ValidationException("Invalid input")
    except ValidationException as e:
        print(f"Validation error: {e.detail}")
    
    try:
        raise NotFoundException("User not found")
    except NotFoundException as e:
        print(f"Not found: {e.detail}")
    
    # Can catch base class to handle all child exceptions
    try:
        raise UnauthorizedException("Access denied")
    except BaseAPIException as e:  # Polymorphism
        print(f"API error: {e.detail}")
```

---

## 3. Design Patterns in OOP

### **3.1 Strategy Pattern**
- **OOP Concept:** Polymorphism + Inheritance
- **Purpose:** Multiple implementations of same interface
- **Example:** Different style configurations

### **3.2 Factory Pattern**
- **OOP Concept:** Encapsulation + Polymorphism
- **Purpose:** Centralized object creation
- **Example:** Creating different generators

### **3.3 Template Method Pattern**
- **OOP Concept:** Inheritance + Abstraction
- **Purpose:** Define algorithm structure, let subclasses implement details
- **Example:** Abstract base class with concrete implementations

### **3.4 Dependency Injection**
- **OOP Concept:** Composition + Dependency Inversion
- **Purpose:** Loose coupling between classes
- **Example:** Injecting dependencies into processor

---

## 4. Key OOP Concepts Explained

### **The Four Pillars of Object-Oriented Programming**

1. **Encapsulation** - Hiding internal implementation details and controlling access to data
   - Example: Private attributes (`__processing_cache`) and methods (`_validate_landmarks`) in `FacialSegmentationProcessor`

2. **Inheritance** - Creating new classes based on existing classes, inheriting their properties and methods
   - Example: `ValidationException` inherits from `BaseAPIException`, reusing common error handling logic

3. **Polymorphism** - Same interface, different implementations
   - Example: `get_region_style()` method behaves differently in `DefaultStyleConfig`, `ColorfulStyleConfig`, and `MinimalStyleConfig`

4. **Abstraction** - Hiding complex implementation details behind simple interfaces
   - Example: `OutputGenerator` abstract class defines the interface, while `SVGGenerator` and `PNGGenerator` provide concrete implementations

### **Composition vs Inheritance**

**Inheritance** represents an "Is-a" relationship where a child class is a specialized version of the parent class.
- Example: `ValidationException` IS-A `BaseAPIException`

**Composition** represents a "Has-a" relationship where a class contains instances of other classes.
- Example: `FacialSegmentationProcessor` HAS-A `SegmentationConfig`

```python
# Inheritance - "Is-a" relationship
class ValidationException(BaseAPIException):
    pass

# Composition - "Has-a" relationship
class FacialSegmentationProcessor:
    def __init__(self, config: SegmentationConfig):
        self.config = config
```

### **Abstract Base Classes**

Abstract base classes provide several key benefits:

- **Interface Enforcement** - Ensures subclasses implement required abstract methods
- **Prevent Instantiation** - Cannot create objects of abstract classes directly
- **Documentation** - Clearly defines the contract that subclasses must follow
- **Type Safety** - Guarantees all subclasses have the same interface

### **Dependency Injection**

Dependency injection is a design pattern where dependencies are provided from outside rather than created inside the class.

**Benefits:**
- **Testability** - Easy to mock dependencies for unit testing
- **Flexibility** - Can swap implementations without modifying the class
- **Loose Coupling** - Classes don't depend on concrete implementations

```python
# Without Dependency Injection (tight coupling)
class FacialSegmentationProcessor:
    def __init__(self):
        self.config = SegmentationConfig()  # Hard-coded dependency
        self.image_generator = ImageGenerator()  # Hard-coded dependency

# With Dependency Injection (loose coupling)
class FacialSegmentationProcessor:
    def __init__(self, config: SegmentationConfig = None, 
                 image_generator: ImageGenerator = None):
        self.config = config or SegmentationConfig()
        self.image_generator = image_generator or ImageGenerator()
```

### **Polymorphism Implementation**

Polymorphism in Python can be implemented through several approaches:

- **Method Overriding** - Child classes override parent methods with different implementations
- **Duck Typing** - Objects with the same methods can be used interchangeably
- **Abstract Base Classes** - Enforce consistent interfaces across different implementations

```python
# Method Overriding
class BaseAPIException(HTTPException):
    def __init__(self, detail: str, error_code: str = None):
        super().__init__(status_code=400, detail={"detail": detail, "error_code": error_code})

class ValidationException(BaseAPIException):
    def __init__(self, detail: str = "Validation error"):
        super().__init__(detail, "VALIDATION_ERROR")  # Override with specific values

# Duck Typing
def process_with_any_generator(generator):
    return generator.generate((100, 100), {})  # Works with any object that has generate() method

# Works with SVGGenerator, PNGGenerator, JSONGenerator, etc.
```

---

## **Summary: OOP in the Codebase**

### **Core OOP Concepts Demonstrated:**

1. **Classes & Objects** ✅
   - `StyleConfig`, `OutputGenerator`, `FacialSegmentationProcessor`

2. **Inheritance** ✅
   - `BaseAPIException` → `ValidationException`, `NotFoundException`
   - `StyleConfig` → `DefaultStyleConfig`, `ColorfulStyleConfig`

3. **Polymorphism** ✅
   - Same `generate()` method, different implementations
   - Same `get_region_style()` method, different behaviors

4. **Encapsulation** ✅
   - Private methods with `_` prefix
   - Private attributes with `__` prefix
   - Public interfaces for external access

5. **Abstraction** ✅
   - Abstract base classes (`OutputGenerator`, `StyleConfig`)
   - Abstract methods that must be implemented

### **Design Patterns Using OOP:**

- ✅ **Strategy Pattern** - Multiple style implementations
- ✅ **Factory Pattern** - Object creation
- ✅ **Template Method** - Abstract base class
- ✅ **Dependency Injection** - Constructor injection
- ✅ **Exception Hierarchy** - Inheritance-based error handling

**This codebase demonstrates modern Python OOP principles in practice!**
