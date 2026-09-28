.PHONY: all train test app clean

all: train test

train:
	python src/train.py

test:
	pytest -v

app:
	streamlit run app/app.py

clean:
	rm -rf __pycache__ src/__pycache__ app/__pycache__ app/views/__pycache__ tests/__pycache__ .pytest_cache
