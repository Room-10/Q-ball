SHELL := /bin/bash

DOKUMENT := main

CHPT_DIR := chpt
CHAPTERS := $(CHPT_DIR)/pre-model.tex \
            $(CHPT_DIR)/pre-W1-TV.tex \
            $(CHPT_DIR)/W1-TV.tex
            
HEADER_DIR := .
HEADERS := $(HEADER_DIR)/header.tex


all: $(DOKUMENT).pdf

$(DOKUMENT).pdf: $(DOKUMENT).tex $(DOKUMENT).bbl
	pdflatex -halt-on-error $(DOKUMENT)
	pdflatex -halt-on-error $(DOKUMENT)

$(DOKUMENT).aux: $(DOKUMENT).tex $(CHAPTERS) $(HEADERS) $(FIGURES)
	pdflatex -halt-on-error $(DOKUMENT)

$(DOKUMENT).bbl: database.bib
	biber $(DOKUMENT)

database.bib: $(DOKUMENT).aux

clean:
	rm -rf $(DOKUMENT).{aux,log,pdf,bbl,blg,run.xml,toc,bcf,gnuplot}
	rm -rf database.bib

