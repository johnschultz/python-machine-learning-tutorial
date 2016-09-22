ENVDIR=venv
OSTYPE=$(shell uname)
SHELL=/bin/bash

.PHONY: localenv
localenv:
	[ -d $(ENVDIR) ] || python3 -m venv $(ENVDIR)
	CC=gcc $(ENVDIR)/bin/pip install -r requirements.txt
