.PHONY: help clean clean-pyc clean-build clean-test lint test test-all coverage docs release sdist

help:
	@echo "clean - remove auxiliary files/artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove testing artifacts"
	@echo "test - run tests quickly with the default Python"

clean: clean-build clean-py clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.eggs' -exec rm -rf {} +

clean-py:
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*.nbc' -exec rm -f {} +
	find . -name '*.nbi' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

clean-test:
	find . -name '.benchmarks' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +
	find . -name '.tox' -exec rm -rf {} +

test:
	pytest
