# Test Suite for Discord AI Agent

This directory contains the comprehensive test suite for the Discord AI Agent project, focusing on RSS functionality, search capabilities, and vector operations.

## Test Structure

### Core Test Files

- **`test_searchutil_sparse_vector.py`** - Tests for the search utility module
  - Vector search functionality
  - Text chunking with Chonkie
  - HTML to Markdown conversion
  - Search result formatting
  - Integration tests for full search pipeline

- **`test_rssutil.py`** - Tests for the RSS utility module
  - RSS feed handling and caching
  - Search integration with RSS feeds
  - Timestamp filtering
  - Error handling and edge cases
  - Integration tests for full RSS pipeline

- **`test_source_handlers.py`** - Tests for source handlers
  - RSS fetching and parsing
  - HTTP client interactions
  - Cache management
  - Error handling for network issues

### Configuration Files

- **`conftest.py`** - Shared test fixtures and configuration
  - Mock RSS entries and feeds
  - HTTP response mocks
  - Logging configuration
  - Common test utilities

- **`__init__.py`** - Package initialization

## Test Data

The test suite uses realistic RSS data based on the LocalLLaMA subreddit (`https://www.reddit.com/r/LocalLLaMA.rss`) to ensure tests reflect real-world usage patterns.

### Sample RSS Entries

Tests include mock RSS entries covering:
- Open source AI models discussion
- Fine-tuning techniques (LoRA, QLoRA)
- Hardware requirements for local LLMs
- Model comparisons
- Deployment strategies

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -e ".[test]"
```

### Basic Test Execution

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=pylib --cov-report=html
```

### Selective Test Execution

Run specific test files:
```bash
pytest test/test_searchutil_sparse_vector.py
pytest test/test_rssutil.py
pytest test/test_source_handlers.py
```

Run specific test classes:
```bash
pytest test/test_searchutil_sparse_vector.py::TestVectorSearchEngine
pytest test/test_rssutil.py::TestRSSAgent
```

Run specific test methods:
```bash
pytest test/test_searchutil_sparse_vector.py::TestVectorSearchEngine::test_html_to_markdown_with_html2text
```

### Test Categories

Run only unit tests (exclude integration):
```bash
pytest -m "not integration"
```

Run only integration tests:
```bash
pytest -m "integration"
```

Run tests excluding slow tests:
```bash
pytest -m "not slow"
```

## Test Coverage

The test suite provides comprehensive coverage for:

### Search Functionality
- ✅ HTML to Markdown conversion (with and without html2text)
- ✅ Text chunking (with and without Chonkie)
- ✅ Vector search with SentenceTransformer
- ✅ Keyword search fallback
- ✅ Search result formatting
- ✅ Error handling and edge cases

### RSS Functionality
- ✅ RSS feed fetching and parsing
- ✅ Cache management
- ✅ Timestamp filtering
- ✅ Search integration
- ✅ Error handling for network issues
- ✅ Malformed feed handling

### Integration Tests
- ✅ Full RSS pipeline from fetch to search results
- ✅ Caching behavior verification
- ✅ Real XML parsing with mock responses

## Mock Strategy

The test suite uses extensive mocking to:
- Avoid external network calls during testing
- Control test data and scenarios
- Test error conditions reliably
- Ensure fast test execution

### Key Mocks
- `httpx.AsyncClient` - HTTP requests
- `feedparser` - RSS parsing
- `SentenceTransformer` - Vector embeddings
- `SentenceChunker` - Text chunking
- `html2text` - HTML conversion

## Continuous Integration

The test suite is configured to run in CI environments with:
- Coverage reporting
- Integration test markers
- Async test support
- Structured logging

## Adding New Tests

When adding new functionality:

1. **Unit Tests**: Add tests for individual functions/methods
2. **Integration Tests**: Add tests for complete workflows
3. **Edge Cases**: Test error conditions and boundary cases
4. **Mock Data**: Use realistic test data based on actual RSS feeds
5. **Documentation**: Update this README with new test categories

### Test Naming Convention
- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<functionality>_<scenario>`

### Example Test Structure
```python
class TestNewFeature:
    """Test the new feature functionality."""
    
    def test_new_feature_success(self):
        """Test successful execution of new feature."""
        # Test implementation
        
    def test_new_feature_error_handling(self):
        """Test error handling in new feature."""
        # Test implementation
        
    @pytest.mark.integration
    def test_new_feature_integration(self):
        """Test new feature in full pipeline."""
        # Test implementation
``` 