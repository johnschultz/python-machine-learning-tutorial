ENVDIR=venv
OSTYPE=$(shell uname)
SHELL=/bin/bash

.PHONY: localenv
localenv:
	[ -d $(ENVDIR) ] || virtualenv --python=python3 $(ENVDIR)
	CC=gcc $(ENVDIR)/bin/pip install -r requirements.txt
