Guide for contributing to Discord-AI-Agent

# QA/Testing

## Relevant repository structure

* `test/README.md` - Notes & docs for the test suite
* `test/conftest.py` - Shared fixtures and test configuration

Example test files:

* `test/test_searchutil_sparse_vector.py` - Comprehensive tests for vector search, chunking, and HTML conversion
* `test/test_rssutil.py` - Tests for RSS feed handling, caching, and search integration
* `test/test_source_handlers.py` - Tests for RSS fetching and parsing functionality

These files cover:

* HTML to Markdown conversion (with/without html2text)
* Text chunking (with/without Chonkie)
* Vector search with SentenceTransformer
* Keyword search fallback
* RSS feed fetching and caching
* Error handling (network failures, malformed data, missing dependencies) and edge cases (empty inputs, cache expiration, invalid timestamps)

Other notes:

* Proper Mocking: Extensive use of mocks to avoid external dependencies
* Async Support: Full async/await testing with pytest-asyncio
* Integration Tests: End-to-end pipeline testing (marked with `@pytest.mark.integration`)
* Test markers for integration and slow tests

## Running tests

Install test dependencies

```sh
uv pip install ".[test]"
```

Run all tests

```sh
pytest
```

More general info, but with less verbose tracebacks

```sh
pytest test/ -v --tb=short
```

Run with coverage

```sh
pytest --cov=pylib --cov-report=html
```

Run specific test files

```sh
pytest test/test_searchutil_sparse_vector.py
```

Run only unit tests (exclude integration)

```sh
pytest -m "not integration"
```
