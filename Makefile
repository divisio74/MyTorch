PYTHON ?= python3
SOURCE ?= my_torch_analyser.py
EXECUTABLE ?= my_torch_analyser

.PHONY: all install

all: $(EXECUTABLE)

install:
	$(PYTHON) -m pip install -r requirements.txt

$(EXECUTABLE): $(SOURCE)
	cp "$<" "$@"
	chmod +x "$@"
