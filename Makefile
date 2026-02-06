.PHONY: infra-up infra-down dev test dashboard lint install

install:
	pip install -e ".[dev]"

infra-up:
	docker compose up -d

infra-down:
	docker compose down

dev:
	python -m src.main

test:
	pytest tests/ -v --tb=short

dashboard:
	streamlit run src/dashboard/app.py

lint:
	ruff check src/ tests/ config/
	ruff format --check src/ tests/ config/

format:
	ruff check --fix src/ tests/ config/
	ruff format src/ tests/ config/
