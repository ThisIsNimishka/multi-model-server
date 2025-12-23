# Contributing to Multi-Model AI Server

First off, thank you for considering contributing to Multi-Model AI Server! It's people like you that make this project better.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct: be respectful, inclusive, and constructive.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

**Bug Report Template:**

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Run command '....'
3. See error

**Expected behavior**
A clear description of what you expected to happen.

**Environment:**
 - OS: [e.g., Windows 11]
 - Python version: [e.g., 3.11]
 - CUDA version: [e.g., 12.8]
 - GPU(s): [e.g., RTX 3080]
 - vLLM version: [e.g., 0.6.0]

**Logs**
If applicable, add logs to help explain your problem.
```

### Suggesting Features

Feature suggestions are welcome! Please provide:

- A clear description of the feature
- Why it would be useful
- Possible implementation approach (optional)

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code, add tests if applicable
3. Ensure your code follows the existing style
4. Update documentation if needed
5. Make sure your code passes linting

## Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/multi-model-server.git
cd multi-model-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Style Guidelines

### Python

- Follow PEP 8
- Use type hints where possible
- Maximum line length: 100 characters
- Use descriptive variable names

```python
# Good
def start_model_server(model_config: ModelConfig, port: int) -> bool:
    """Start a model server with the given configuration."""
    ...

# Bad
def start(c, p):
    ...
```

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests when relevant

```
feat: add support for Llama models

- Add LlamaConfig class
- Update router to handle Llama endpoints
- Add tests for Llama integration

Fixes #123
```

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring

## Testing

```bash
# Run all tests
python test_client.py test

# Run with verbose output
python -m pytest -v

# Run specific test
python -m pytest tests/test_router.py -k "test_health"
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions and classes
- Update API documentation if endpoints change

## Release Process

1. Update version in relevant files
2. Update CHANGELOG.md
3. Create a pull request to `main`
4. After merge, create a GitHub release with tag

## Questions?

Feel free to open an issue with the "question" label or reach out to the maintainers.

Thank you for contributing! 🎉
