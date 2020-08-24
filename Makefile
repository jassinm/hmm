all: clean build install

clean:
	rm -rf build
	rm -rf mta.egg-info
	rm -rf dist
	rm -rf .pytest_cache
	rm -rf .tox
	find . -name '*.pyc' -print | xargs rm
	find . -name '*.so' -print | xargs rm
	find . -name '*.c' -print | xargs rm

build:
	python setup.py build
	python setup.py build_ext --inplace

install:
	python setup.py install
