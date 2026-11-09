PYTHON		?=	python3
SOURCE		?=	my_torch_analyzer.py
EXECUTABLE	?=	my_torch_analyzer

.PHONY: all install

all: $(EXECUTABLE)

re: fclean all

fclean:
	rm -f $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)

install:
	$(PYTHON) -m pip install -r requirements.txt

$(EXECUTABLE): $(SOURCE)
	cp "$<" "$@"
	chmod +x "$@"
