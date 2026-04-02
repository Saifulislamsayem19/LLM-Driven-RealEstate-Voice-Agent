"""
# Contributing Guide

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment
3. Install development dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  
```

## Code Standards

### Python Style Guide

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions small and focused

### Formatting

```bash
# Format code
black src/

# Check linting
flake8 src/

# Type checking
mypy src/
```

### Example Code Style

```python
from typing import List, Optional

def search_properties(
    budget: float,
    location: str,
    bedrooms: Optional[int] = None
) -> List[Dict[str, Any]]:
    \"\"\"
    Search for properties matching criteria.
    
    Args:
        budget: Maximum price
        location: Desired location
        bedrooms: Number of bedrooms (optional)
        
    Returns:
        List of matching properties
        
    Raises:
        ValueError: If budget is negative
    \"\"\"
    if budget < 0:
        raise ValueError("Budget must be positive")
    
    # Implementation
    return results
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_services.py

# Run specific test
pytest tests/test_services.py::TestDataService::test_load_data
```

### Writing Tests

```python
import pytest
from src.services.data_service import DataService

class TestDataService:
    def test_load_data_success(self):
        \"\"\"Test successful data loading.\"\"\"
        service = DataService()
        df, _, _ = service.load_and_preprocess()
        
        assert df is not None
        assert len(df) > 0
    
    def test_load_data_missing_file(self):
        \"\"\"Test handling of missing file.\"\"\"
        with pytest.raises(DataLoadError):
            DataService.load_and_preprocess()
```

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructure
- `test`: Tests
- `chore`: Maintenance

### Examples

```bash
feat(agent): add property detail extraction

Implement property ID and number extraction from user queries
to support "show me property 1" requests.

Closes #123
```

## Pull Request Process

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and commit
3. Write/update tests
4. Update documentation
5. Push and create PR
6. Address review comments
7. Merge after approval

## Project Structure

```
src/
├── core/          # Configuration and exceptions
├── models/        # Data models
├── services/      # Business logic
├── api/           # API routes
└── utils/         # Utilities

tests/             # Test files
docs/              # Documentation
```

## Adding New Features

### 1. Service Implementation

```python
# src/services/new_service.py
class NewService:
    def __init__(self):
        pass
    
    def perform_action(self) -> Result:
        \"\"\"Perform the action.\"\"\"
        pass
```

### 2. API Route

```python
# src/api/routes/new_route.py
from fastapi import APIRouter

router = APIRouter()

@router.post("/endpoint")
async def new_endpoint():
    \"\"\"New endpoint.\"\"\"
    pass
```

### 3. Tests

```python
# tests/test_new_service.py
def test_new_service():
    service = NewService()
    result = service.perform_action()
    assert result.success
```

### 4. Documentation

Update relevant markdown files:
- README.md
- ARCHITECTURE.md
- API documentation

## Questions?

- Open an issue
- Email: saifulislamsayem19@gmail.com
- Slack: #real-estate-dev
"""
"""
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Environment Variables
.env
.env.local

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
logs/
*.log

# Database
vector_db/
user_data/
*.db
*.sqlite

# Temporary files
temp/
*.tmp

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Jupyter
.ipynb_checkpoints/
"""
