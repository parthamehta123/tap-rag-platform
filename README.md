# TAP RAG Platform

Production-ready **RAG + Fine-Tuning** platform for TAP CloudOps / threat intel — aligned with the Cisco AI interview architecture (LangChain RAG, LangGraph guardrails, MCP reputation tools, LoRA anomaly classification, Bedrock, golden evals, CI/CD).

**Repo:** [github.com/parthamehta123/tap-rag-platform](https://github.com/parthamehta123/tap-rag-platform)

## Architecture

```
                    ┌──────────────┐
                    │   Cognito /  │
                    │   SSO JWT    │
                    └──────┬───────┘
                           ▼
              ALB → ECS / EKS  →  FastAPI + Streamlit
                           │
              ┌────────────┼────────────────────┐
              ▼            ▼                    ▼
        LangChain RAG   LangGraph Agent    LoRA Classifier
        (retrieve →     (read/write/       (SUSPICIOUS /
         re-rank →       execute +          BENIGN /
         Bedrock)        sandbox + PR)      INVESTIGATE)
              │            │                    │
              ▼            ▼                    ▼
           ChromaDB     MCP Tools          Training data
           (S3 sync)    (hash/IP)          + model card
                           │
                           ▼
                 Bedrock AgentCore (Runtime / Gateway /
                 Memory / Guardrails / OpenTelemetry)
```

| Layer | Choice | Why |
|-------|--------|-----|
| RAG chatbot | LangChain LCEL | Linear retrieve → generate |
| Security agent | LangGraph | Branching, sandbox, PR gates |
| Schemas | Pydantic v2 | Boundary validation + MCP schemas |
| LLM / Embed | Bedrock Claude + Titan (mockable) | IAM, VPC, Guardrails |
| Vector store | ChromaDB → S3 | Self-hosted, ECS-friendly |
| Tools | MCP | Agent-discoverable reputation APIs |
| Domain model | LoRA LLaMA | Data-sovereign anomaly classification |
| Data jobs | Databricks + PySpark | Prevalence aggregates at scale |
| IaC | Terraform | ECR, ECS, S3, Bedrock Guardrail, IAM |
| CI/CD | GitHub Actions | Lint, unit, golden eval, Docker, deploy |

## Quick start (local, no AWS)

```bash
git clone https://github.com/parthamehta123/tap-rag-platform.git
cd tap-rag-platform
cp .env.example .env
python -m venv .venv && source .venv/bin/activate
make install
make ingest
make test
make api    # http://localhost:8000/docs
# another terminal:
make ui     # http://localhost:8501
```

Or with Docker Compose:

```bash
docker compose up --build
```

Mocks are on by default (`USE_MOCK_LLM=true`, `USE_MOCK_EMBEDDINGS=true`) so CI and laptops work without Bedrock.

## API surface

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| POST | `/v1/rag/query` | Two-stage RAG |
| POST | `/v1/agent/run` | LangGraph security agent |
| POST | `/v1/lora/classify` | Anomaly classification |
| POST | `/v1/feedback` | Thumbs / ratings → RLHF seed |

## RAG design (from production patterns)

1. **Ingest** — parallel file load (`.md/.py/.tf/...`), chunk 600 / overlap 100, content-hash dedupe  
2. **Retrieve** — Chroma cosine top-k  
3. **Dedupe** — max 2 chunks per source  
4. **Re-rank** — cross-encoder (enabled when not in mock mode)  
5. **Generate** — grounded system prompt + citations  
6. **Feedback** — JSON to disk / S3 for eval + RLHF

## Golden eval pipeline

- Dataset: `data/golden/rag_golden.json`  
- Runner: `python -m tap_rag.eval.runner`  
- Gates (CI On-Demand Evaluations): answer hit-rate, citation recall, hallucination rate  
- LoRA holdout eval: `tests/eval/test_golden.py` using `data/training/tap_training_data.json`

```bash
make eval
```

## MCP reputation server

```bash
python -m tap_rag.mcp.server          # stdio MCP
python -m tap_rag.mcp.client          # demo client
```

Tools: `hash_reputation`, `ip_reputation`, `bulk_hash_reputation` — Pydantic-validated.

## LangGraph agents

- `tap_rag.agent.security_agent.SecurityAgent` — permission → sandbox → diff validation → PR / block  
- `tap_rag.agent.hybrid_graph` — routes to RAG / reputation / LoRA by intent  
- `tap_rag.agent.agentcore.AgentCoreRuntimeAdapter` — Bedrock AgentCore circuit breakers & wiring notes  

## Fine-tuning (LoRA)

Training examples live in `data/training/tap_training_data.json` (same schema as the Colab notebook).  
Local/CI uses a rule-based classifier; set `LORA_ADAPTER_DIR` + install `[lora]` extras for PEFT inference.

```bash
pip install -e ".[lora]"
```

## Databricks / PySpark

| Job | Path |
|-----|------|
| Weekly prevalence aggregate | `jobs/databricks/prevalence_weekly.py` |
| Doc chunking for RAG | `jobs/databricks/rag_chunk_job.py` |

## Terraform / K8s / ECS

```bash
cd terraform && terraform init
terraform plan -var-file=environments/dev/terraform.tfvars.example
```

Provisions: S3 artifacts, ECR (api/ui), ECS cluster, IAM (Bedrock+S3), Bedrock Guardrail, CloudWatch logs.  
EKS manifests: `k8s/api-deployment.yaml` (Deployment + Service + HPA).

## GitHub Actions

| Workflow | Trigger | What |
|----------|---------|------|
| `ci.yml` | PR / push | Ruff, unit tests, golden eval, Docker build |
| `deploy.yml` | main CI success / manual | OIDC → ECR push → ECS rollout |
| `reembed.yml` | `data/docs/**` changes | Rebuild Chroma artifact |

Configure secrets: `AWS_DEPLOY_ROLE_ARN`, `S3_ARTIFACTS_BUCKET`; vars: `ECS_CLUSTER`, `ECS_API_SERVICE`, `AWS_REGION`.

## Production checklist

1. Set `USE_MOCK_*=false` and IAM task role with Bedrock + S3  
2. Sync Chroma to S3 on ingest; pull on container start  
3. Attach Bedrock Guardrail ID from Terraform output  
4. Run golden eval in CI before deploy (already wired)  
5. Sample online conversations → S3 feedback → RLHF / DPO later  
6. Optional: move agent hosting to Bedrock AgentCore Runtime  

## Project layout

```
tap-rag-platform/
├── src/tap_rag/
│   ├── models/          # Pydantic schemas
│   ├── rag/             # ingest, retriever, pipeline, llm
│   ├── agent/           # LangGraph + AgentCore
│   ├── mcp/             # reputation MCP server/client
│   ├── lora/            # classify / eval / model card
│   ├── eval/            # golden-dataset runner
│   ├── api/             # FastAPI
│   └── ui/              # Streamlit
├── data/{docs,golden,training}/
├── jobs/databricks/
├── terraform/
├── k8s/
├── .github/workflows/
├── Dockerfile
└── docker-compose.yml
```

## License

MIT — internal portfolio / interview reference architecture.
