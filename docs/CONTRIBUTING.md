# Contributing to Embodied Intelligence Platform

Thank you for your interest in contributing to the Embodied Intelligence Platform (EIP)! This document provides guidelines for contributing to the project.

## 🚀 Quick Start

1. **Fork and clone** the repository
2. **Set up development environment**:
   ```bash
   ./scripts/setup_dev_env.sh
   ```
3. **Install pre-commit hooks**:
   ```bash
   pip install pre-commit
   pre-commit install
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📋 Development Workflow

### 1. Code Quality Standards

- **Python**: Follow PEP 8 with 100-character line limit
- **ROS 2**: Follow ROS 2 coding standards
- **Documentation**: Use Google-style docstrings
- **Tests**: Maintain >90% test coverage

### 2. Pre-commit Checks

The following checks run automatically on commit:

- **Code formatting**: Black, isort
- **Linting**: flake8, mypy
- **Security**: bandit, detect-secrets
- **Safety**: pip-audit, safety check

### 3. Testing Requirements

- **Unit tests**: Required for all new functionality
- **Integration tests**: Required for system components
- **Safety benchmarks**: Must pass for safety-critical code
- **Performance tests**: Required for optimization changes

### 4. Pull Request Process

1. **Create feature branch** from `develop`
2. **Write tests** for new functionality
3. **Update documentation** as needed
4. **Run full test suite**:
   ```bash
   colcon test --packages-select your_package
   python -m pytest benchmarks/safety_benchmarks/
   ```
5. **Submit PR** with detailed description
6. **Address review comments**
7. **Merge** after approval

## 🏗️ Architecture Guidelines

### Package Structure

```
package_name/
├── package_name/
│   ├── __init__.py
│   ├── main_node.py
│   └── utilities.py
├── launch/
│   └── demo.launch.py
├── config/
│   └── params.yaml
├── tests/
│   └── test_main_node.py
├── package.xml
├── CMakeLists.txt
└── setup.py
```

### Safety-First Development

- **All components** must pass safety verification
- **Safety tests** must be written for new features
- **Emergency stops** must be implemented for physical systems
- **Input validation** is required for all external inputs

### ROS 2 Best Practices

- Use **QoS profiles** appropriate for message criticality
- Implement **parameter validation** in nodes
- Use **services** for request-response patterns
- Use **actions** for long-running tasks
- Implement **graceful shutdown** handlers

## 🧪 Testing Guidelines

### Unit Tests

```python
def test_feature_name():
    """Test description"""
    # Arrange
    component = Component()
    
    # Act
    result = component.method()
    
    # Assert
    assert result == expected_value
```

### Integration Tests

```python
class TestSystemIntegration:
    """Test system integration"""
    
    @pytest.fixture
    def ros_context(self):
        """Setup ROS 2 context"""
        rclpy.init()
        yield
        rclpy.shutdown()
    
    def test_end_to_end_workflow(self, ros_context):
        """Test complete workflow"""
        # Test implementation
        pass
```

### Safety Tests

```python
def test_safety_constraint():
    """Test safety constraint enforcement"""
    # Test that safety violations are detected
    # Test that emergency stops work
    # Test that unsafe actions are blocked
    pass
```

## 📚 Documentation Standards

### Code Documentation

```python
def complex_function(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When parameters are invalid
        
    Example:
        >>> result = complex_function("test", 42)
        >>> print(result)
        True
    """
    pass
```

### README Files

Each package should have a README.md with:

- **Purpose**: What the package does
- **Dependencies**: Required packages and versions
- **Usage**: How to use the package
- **Examples**: Code examples
- **Testing**: How to run tests

## 🔒 Security Guidelines

### Input Validation

- **Sanitize all inputs** from external sources
- **Validate data types** and ranges
- **Check for injection attacks** (SQL, XSS, etc.)
- **Use parameterized queries** for database operations

### Model Security

- **Verify model checksums** before loading
- **Scan for malicious code** in model files
- **Use trusted model sources** only
- **Implement model sandboxing** for untrusted models

### Secrets Management

- **Never commit secrets** to version control
- **Use environment variables** for sensitive data
- **Rotate API keys** regularly
- **Use secure storage** for production secrets

## 🚀 Performance Guidelines

### Optimization

- **Profile before optimizing** - measure first
- **Use appropriate data structures** for the task
- **Implement caching** for expensive operations
- **Optimize critical paths** only

### Memory Management

- **Monitor memory usage** in long-running processes
- **Implement cleanup** for large data structures
- **Use context managers** for resource management
- **Avoid memory leaks** in event handlers

## 🤝 Community Guidelines

### Communication

- **Be respectful** and inclusive
- **Ask questions** when unsure
- **Provide constructive feedback**
- **Help others** when possible

### Issue Reporting

When reporting issues, include:

- **Environment**: OS, Python version, ROS 2 version
- **Steps to reproduce**: Clear, numbered steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Logs**: Relevant error messages and logs

### Feature Requests

When requesting features, include:

- **Use case**: Why the feature is needed
- **Proposed solution**: How it could work
- **Alternatives considered**: Other approaches
- **Impact**: Who would benefit

## 📋 Checklist for Contributors

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] Tests are written and passing
- [ ] Documentation is updated
- [ ] Security considerations addressed
- [ ] Performance impact assessed
- [ ] Pre-commit checks pass
- [ ] Integration tests pass
- [ ] Safety tests pass

## 🆘 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check existing docs first
- **Code Examples**: Look at existing implementations

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the Apache 2.0 License.

---

Thank you for contributing to the future of intelligent robotics! 🤖 