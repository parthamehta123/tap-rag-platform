.PHONY: install test lint eval ingest api ui docker-up docker-down fmt

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests

test:
	USE_MOCK_LLM=true USE_MOCK_EMBEDDINGS=true pytest tests/unit tests/eval -q

eval:
	USE_MOCK_LLM=true USE_MOCK_EMBEDDINGS=true python -m tap_rag.rag.ingest
	USE_MOCK_LLM=true USE_MOCK_EMBEDDINGS=true python -m tap_rag.eval.runner

ingest:
	USE_MOCK_EMBEDDINGS=true python -m tap_rag.rag.ingest

api:
	USE_MOCK_LLM=true USE_MOCK_EMBEDDINGS=true uvicorn tap_rag.api.main:app --reload --port 8000

ui:
	USE_MOCK_LLM=true USE_MOCK_EMBEDDINGS=true streamlit run src/tap_rag/ui/app.py

mcp:
	python -m tap_rag.mcp.server

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

tf-init:
	cd terraform && terraform init

tf-plan:
	cd terraform && terraform plan -var-file=environments/dev/terraform.tfvars.example
