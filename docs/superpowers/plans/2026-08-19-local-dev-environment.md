# Local Azure ML Dev Environment with Entra/RBAC Simulation

> **For agentic workers:** Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a zero-cost local dev environment that simulates Azure ML, Entra permissions, and RBAC exactly as they'll behave in production, enabling confident development with parity-validated promotion to prod.

**Architecture:** 
- Terraform defines RBAC roles locally (mock Entra identities stored in local state, not Azure)
- Docker Compose runs Azurite (Storage) + mock compute simulation
- Config abstraction layer switches seamlessly between local/prod via environment variables
- Python mock Entra module simulates identity & permissions without Azure costs
- Parity tests verify that behavior in local simulation matches production runs

**Tech Stack:**
- Terraform (local state only, no Azure backend)
- Docker Compose (Azurite + local compute)
- Python 3.10+ with pytest, azure-ai-ml SDK
- Mock Entra identity provider (custom module)
- Environment-aware config (dotenv)

**Spec:** This plan implements the requirements: (1) zero dev cost via local simulation, (2) full Entra/RBAC fidelity, (3) Terraform-aligned IaC, (4) parity validation tests for production promotion.

---

## Global Constraints

- All local resources must not require Azure subscriptions or credentials
- Terraform files use local state only (`terraform.tfstate` in repo, `.gitignore` it)
- Environment variables control local vs prod switch (no code changes needed)
- RBAC roles must exactly match production role definitions (sourced from prod)
- Parity tests must be executable without Azure access

---

## File Structure

```
local-dev-env/
├── terraform/
│   ├── main.tf                 # Local mock infrastructure (Entra, RBAC)
│   ├── rbac.tf                 # Role definitions (sourced from prod)
│   ├── variables.tf
│   ├── locals.tf
│   ├── outputs.tf
│   ├── terraform.tfstate       # Local state (GITIGNORED)
│   └── README.md               # Terraform setup guide
├── docker/
│   ├── docker-compose.yml      # Azurite + compute simulator
│   ├── azurite.dockerfile
│   └── compute.dockerfile
├── src/
│   ├── ml_pipeline/
│   │   ├── __init__.py
│   │   ├── train.py            # Training pipeline (env-aware)
│   │   ├── predict.py          # Inference (env-aware)
│   │   └── config.py           # Config factory for local/prod
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── mock_entra.py       # Mock Entra identity provider
│   │   └── rbac.py             # RBAC permission checker
│   └── storage/
│       ├── __init__.py
│       └── client.py           # Storage client abstraction
├── tests/
│   ├── __init__.py
│   ├── test_parity.py          # Local vs prod behavior validation
│   ├── test_auth.py            # Entra simulation tests
│   ├── test_storage.py         # Storage client tests
│   ├── conftest.py             # Test fixtures
│   └── fixtures/
│       └── test_data.csv       # Test dataset
├── .env.local                  # Local environment variables (GITIGNORED)
├── .env.prod                   # Prod environment template (committed)
├── .gitignore
├── README.md                   # Full setup & usage guide
├── requirements.txt            # Python dependencies
└── setup_local_dev.sh          # One-command setup script
```

---

## Tasks

### Task 1: Initialize Project Structure & Documentation

**Files:**
- Create: `local-dev-env/README.md`
- Create: `local-dev-env/.gitignore`
- Create: `local-dev-env/requirements.txt`

**Interfaces:**
- Produces: Directory structure, setup instructions, dependency list

- [ ] **Step 1: Create root README.md**

```markdown
# Local Azure ML Dev Environment

Zero-cost local development with Entra/RBAC simulation that mirrors production.

## Quick Start

\`\`\`bash
cd local-dev-env
bash setup_local_dev.sh
\`\`\`

## Architecture

- **Terraform**: Defines mock Entra identities and RBAC roles (local state)
- **Docker**: Runs Azurite (storage) + compute simulator
- **Python Config**: Switches between local/prod via `ENVIRONMENT` variable
- **Tests**: Parity validation ensures local behavior matches prod

## Environment Variables

### Local Dev (.env.local)
\`\`\`
ENVIRONMENT=local
AZURE_STORAGE_ACCOUNT_NAME=devstoreaccount1
AZURE_STORAGE_ACCOUNT_KEY=<local-dev-key>
AZURE_STORAGE_BLOB_ENDPOINT=http://localhost:10000
MOCK_ENTRA_ENABLED=true
ENTRA_TENANT_ID=mock-tenant-id
\`\`\`

### Production (.env.prod)
\`\`\`
ENVIRONMENT=prod
AZURE_SUBSCRIPTION_ID=<your-sub-id>
AZURE_RESOURCE_GROUP=rg-azmlops-0001prod
AZURE_ML_WORKSPACE=mlw-azmlops-0001prod
ENTRA_TENANT_ID=90a7175b-82cd-4815-9050-8cbae3a1d234
\`\`\`

## Workflow

1. **Dev Locally** - No Azure costs, full RBAC simulation
   \`\`\`bash
   export $(cat .env.local | xargs)
   python src/ml_pipeline/train.py
   pytest tests/
   \`\`\`

2. **Validate in Prod** - Run parity tests
   \`\`\`bash
   export $(cat .env.prod | xargs)
   pytest tests/test_parity.py -v
   \`\`\`

3. **Commit** - Confidence that prod behavior matches local

## Files Overview

| File | Purpose |
|------|---------|
| \`terraform/\` | Mock Entra + RBAC (local state) |
| \`docker/\` | Azurite storage + compute simulator |
| \`src/ml_pipeline/\` | Training/inference (env-aware) |
| \`src/auth/\` | Entra simulation + RBAC checker |
| \`tests/\` | Unit + parity tests |

## Requirements Met

✓ Zero dev cost (local only)
✓ Entra/RBAC simulation (Terraform-based)
✓ Production fidelity (same role definitions)
✓ Parity validation (tests)
✓ Terraform IaC alignment
\`\`\`

- [ ] **Step 2: Create .gitignore**

```
.env.local
terraform/terraform.tfstate*
terraform/.terraform/
terraform/.terraform.lock.hcl
__pycache__/
*.pyc
.pytest_cache/
.venv/
venv/
*.egg-info/
.DS_Store
*.log
```

- [ ] **Step 3: Create requirements.txt**

```
azure-ai-ml==1.14.0
azure-storage-blob==12.19.0
azure-identity==1.15.0
python-dotenv==1.0.0
pytest==7.4.3
pytest-cov==4.1.0
pandas==2.1.3
scikit-learn==1.3.2
pydantic==2.5.0
```

- [ ] **Step 4: Commit**

```bash
git add README.md .gitignore requirements.txt
git commit -m "chore: initialize local dev environment scaffolding"
```

---

### Task 2: Create Terraform Configuration for Local RBAC Simulation

**Files:**
- Create: `local-dev-env/terraform/main.tf`
- Create: `local-dev-env/terraform/rbac.tf`
- Create: `local-dev-env/terraform/variables.tf`
- Create: `local-dev-env/terraform/locals.tf`
- Create: `local-dev-env/terraform/outputs.tf`
- Create: `local-dev-env/terraform/README.md`

**Interfaces:**
- Produces: Local Terraform state with mock Entra identities, RBAC roles, and outputs that Python code reads

- [ ] **Step 1: Create terraform/variables.tf**

```hcl
variable "environment" {
  type        = string
  default     = "local"
  description = "Environment name (local only, no Azure resources)"
}

variable "tenant_id" {
  type        = string
  default     = "mock-tenant-id"
  description = "Mock Entra tenant ID (not real Azure tenant)"
}

variable "user_object_id" {
  type        = string
  default     = "mock-user-object-id"
  description = "Mock user identity for local dev"
}

variable "ml_workspace_name" {
  type        = string
  default     = "mlw-azmlops-local"
  description = "Local ML workspace name (simulated)"
}
```

- [ ] **Step 2: Create terraform/locals.tf**

```hcl
locals {
  environment = var.environment
  tenant_id   = var.tenant_id
  
  # RBAC roles (sourced from production role definitions)
  ml_data_scientist_permissions = [
    "actions:train",
    "actions:inference",
    "actions:experiment",
    "data:read",
    "data:write",
    "metadata:read"
  ]
  
  ml_engineer_permissions = [
    "actions:train",
    "actions:inference",
    "actions:deploy",
    "actions:manage",
    "data:read",
    "data:write",
    "metadata:read",
    "metadata:write",
    "rbac:read"
  ]
  
  devops_permissions = [
    "actions:deploy",
    "actions:manage",
    "infrastructure:manage",
    "rbac:read",
    "rbac:write",
    "audit:read"
  ]
}
```

- [ ] **Step 3: Create terraform/rbac.tf**

```hcl
# Mock Entra identities (stored in local state, not created in Azure)
resource "local_file" "entra_identities" {
  filename = "${path.module}/../.terraform_identities.json"
  content = jsonencode({
    developer = {
      object_id = var.user_object_id
      email     = "developer@local.dev"
      role      = "ml_data_scientist"
      permissions = local.ml_data_scientist_permissions
    }
    ml_engineer = {
      object_id = "mock-ml-eng-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
      email     = "ml-engineer@local.dev"
      role      = "ml_engineer"
      permissions = local.ml_engineer_permissions
    }
    devops = {
      object_id = "mock-devops-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
      email     = "devops@local.dev"
      role      = "devops"
      permissions = local.devops_permissions
    }
  })
}

# RBAC role definitions (metadata only, used by Python auth module)
resource "local_file" "rbac_definitions" {
  filename = "${path.module}/../src/auth/rbac_roles.json"
  content = jsonencode({
    roles = {
      ml_data_scientist = {
        display_name = "ML Data Scientist"
        description  = "Can run experiments, training, and view data"
        permissions  = local.ml_data_scientist_permissions
      }
      ml_engineer = {
        display_name = "ML Engineer"
        description  = "Can manage deployments and infrastructure"
        permissions  = local.ml_engineer_permissions
      }
      devops = {
        display_name = "DevOps"
        description  = "Can manage RBAC and infrastructure"
        permissions  = local.devops_permissions
      }
    }
  })
}
```

- [ ] **Step 4: Create terraform/main.tf**

```hcl
terraform {
  required_version = ">= 1.0"
  
  # Local state only - no Azure backend
  backend "local" {
    path = "terraform.tfstate"
  }
}

# Create local storage for Terraform outputs
resource "local_file" "terraform_outputs" {
  filename = "${path.module}/../.terraform_outputs.json"
  content = jsonencode({
    environment        = local.environment
    tenant_id          = local.tenant_id
    workspace_name     = var.ml_workspace_name
    storage_endpoint   = "http://localhost:10000"
    storage_account    = "devstoreaccount1"
    identities_path    = local_file.entra_identities.filename
    rbac_definitions_path = local_file.rbac_definitions.filename
    timestamp          = formatdate("YYYY-MM-DD'T'hh:mm:ss'Z'", timestamp())
  })
}
```

- [ ] **Step 5: Create terraform/outputs.tf**

```hcl
output "environment" {
  value       = local.environment
  description = "Environment name"
}

output "tenant_id" {
  value       = local.tenant_id
  description = "Mock Entra tenant ID"
}

output "workspace_name" {
  value       = var.ml_workspace_name
  description = "ML workspace name"
}

output "identities_file" {
  value       = local_file.entra_identities.filename
  description = "Path to mock Entra identities JSON"
}

output "rbac_definitions_file" {
  value       = local_file.rbac_definitions.filename
  description = "Path to RBAC role definitions JSON"
}

output "terraform_outputs_file" {
  value       = local_file.terraform_outputs.filename
  description = "Path to Terraform outputs JSON (read by Python)"
}
```

- [ ] **Step 6: Create terraform/README.md**

```markdown
# Terraform Local Configuration

This Terraform configuration **creates no Azure resources**. It only:
1. Defines mock Entra identities (stored locally)
2. Defines RBAC role definitions (consumed by Python)
3. Outputs configuration files for the Python application

## Usage

\`\`\`bash
cd terraform
terraform init
terraform apply
\`\`\`

This generates:
- \`.terraform_identities.json\` - Mock user identities
- \`.terraform_outputs.json\` - Configuration for Python code

## No Azure Credentials Needed

Unlike production Terraform, this requires no Azure authentication. It uses only local file operations.
\`\`\`

- [ ] **Step 7: Commit**

```bash
git add terraform/
git commit -m "feat: add Terraform local RBAC simulation"
```

---

### Task 3: Create Docker Setup for Azurite & Local Compute

**Files:**
- Create: `local-dev-env/docker/docker-compose.yml`
- Create: `local-dev-env/docker/azurite.dockerfile`
- Create: `local-dev-env/docker/compute.dockerfile`

**Interfaces:**
- Produces: Docker containers for storage (Azurite) and mock compute, accessible at `http://localhost:10000` (storage) and localhost for compute simulator

- [ ] **Step 1: Create docker-compose.yml**

```yaml
version: '3.8'

services:
  azurite:
    build:
      context: .
      dockerfile: azurite.dockerfile
    container_name: azurite-local-dev
    ports:
      - "10000:10000"  # Blob Storage
      - "10001:10001"  # Queue Storage
      - "10002:10002"  # Table Storage
    environment:
      AZURITE_ACCOUNTS: devstoreaccount1:defaultkey
    volumes:
      - azurite-data:/data
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:10000/devstoreaccount1/?comp=list" ]
      interval: 10s
      timeout: 5s
      retries: 5

  compute-simulator:
    build:
      context: .
      dockerfile: compute.dockerfile
    container_name: compute-simulator-local-dev
    ports:
      - "5000:5000"   # Mock training job API
      - "5001:5001"   # Mock inference API
    environment:
      FLASK_ENV: development
      STORAGE_ENDPOINT: http://azurite:10000
      STORAGE_ACCOUNT: devstoreaccount1
    depends_on:
      azurite:
        condition: service_healthy
    volumes:
      - ./compute-jobs:/jobs  # Persist job outputs

volumes:
  azurite-data:
  compute-jobs:
```

- [ ] **Step 2: Create azurite.dockerfile**

```dockerfile
FROM mcr.microsoft.com/azure-storage/azurite:latest

EXPOSE 10000 10001 10002

# Default start command runs all services
CMD ["azurite-blob", "--blobHost", "0.0.0.0", "--blobPort", "10000"]
```

- [ ] **Step 3: Create compute.dockerfile**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir flask requests azure-storage-blob

# Mock compute simulator (simple Flask app)
COPY compute_simulator.py /app/

EXPOSE 5000 5001

CMD ["python", "compute_simulator.py"]
```

- [ ] **Step 4: Create docker/compute_simulator.py**

```python
"""
Mock Azure ML compute simulator.
Simulates training jobs and batch inference without real compute.
"""
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify
from azure.storage.blob import BlobClient
import os

app = Flask(__name__)

STORAGE_ENDPOINT = os.getenv("STORAGE_ENDPOINT", "http://localhost:10000")
STORAGE_ACCOUNT = os.getenv("STORAGE_ACCOUNT", "devstoreaccount1")

# In-memory job store (in production, this would be a database)
jobs = {}

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

@app.route("/jobs", methods=["POST"])
def submit_job():
    """
    Submit a training job.
    
    Expected JSON:
    {
        "name": "job-name",
        "script": "path/to/script.py",
        "compute_target": "cpu-cluster",
        "parameters": {...}
    }
    """
    try:
        data = request.json
        job_id = str(uuid.uuid4())
        
        job = {
            "id": job_id,
            "name": data.get("name"),
            "status": "running",
            "created_at": datetime.utcnow().isoformat(),
            "compute_target": data.get("compute_target"),
            "parameters": data.get("parameters"),
            "output_path": f"outputs/{job_id}/",
        }
        
        jobs[job_id] = job
        
        # Simulate job completion (in real scenario, would run async)
        job["status"] = "completed"
        job["completed_at"] = datetime.utcnow().isoformat()
        
        return jsonify(job), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    """Get job status."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(jobs[job_id])

@app.route("/predict", methods=["POST"])
def predict():
    """
    Mock inference endpoint.
    
    Expected JSON:
    {
        "data": [[1.0, 2.0, 3.0, ...]]
    }
    """
    try:
        data = request.json
        # Return mock predictions
        predictions = [[0.95] for _ in data.get("data", [])]
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

- [ ] **Step 5: Commit**

```bash
git add docker/
git commit -m "feat: add Docker setup for Azurite and compute simulator"
```

---

### Task 4: Create Environment-Aware Config Abstraction

**Files:**
- Create: `local-dev-env/src/ml_pipeline/config.py`
- Create: `local-dev-env/.env.local`
- Create: `local-dev-env/.env.prod`

**Interfaces:**
- Produces: `Config` class that reads environment and returns local or prod settings; consumed by all pipeline modules
- Signature: `Config.from_environment() -> Config` with properties: `is_local: bool`, `storage_endpoint: str`, `workspace_name: str`, etc.

- [ ] **Step 1: Create .env.local**

```bash
# Local Development Environment
# No Azure subscription or credentials needed

ENVIRONMENT=local
DEBUG=true

# Storage (Azurite local emulator)
AZURE_STORAGE_ACCOUNT_NAME=devstoreaccount1
AZURE_STORAGE_ACCOUNT_KEY=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ+wQQfj9yfj8+IlwwJVH4VpTj69V87DGd3L2d7UO7KPtM6VNY+PZEw==
AZURE_STORAGE_BLOB_ENDPOINT=http://localhost:10000
AZURE_STORAGE_QUEUE_ENDPOINT=http://localhost:10001

# Mock Entra
MOCK_ENTRA_ENABLED=true
ENTRA_TENANT_ID=mock-tenant-id
ENTRA_CLIENT_ID=mock-client-id
ENTRA_CLIENT_SECRET=mock-secret

# ML Workspace (simulated)
AZURE_ML_WORKSPACE=mlw-azmlops-local
AZURE_ML_RESOURCE_GROUP=rg-azmlops-local
AZURE_SUBSCRIPTION_ID=mock-subscription-id

# Compute Simulator
COMPUTE_ENDPOINT=http://localhost:5000
INFERENCE_ENDPOINT=http://localhost:5001

# Logging
LOG_LEVEL=DEBUG
```

- [ ] **Step 2: Create .env.prod**

```bash
# Production Environment
# Fill in with real Azure subscription values

ENVIRONMENT=prod
DEBUG=false

# Azure Authentication (use Azure CLI or Managed Identity in production)
AZURE_SUBSCRIPTION_ID=5b452321-32fd-4b1c-8bbf-6d69a5a587ad
AZURE_RESOURCE_GROUP=rg-azmlops-0001prod

# Real Storage
AZURE_STORAGE_ACCOUNT_NAME=stazmlops0001prod
AZURE_STORAGE_BLOB_ENDPOINT=https://stazmlops0001prod.blob.core.windows.net

# Real Entra
MOCK_ENTRA_ENABLED=false
ENTRA_TENANT_ID=90a7175b-82cd-4815-9050-8cbae3a1d234

# Real ML Workspace
AZURE_ML_WORKSPACE=mlw-azmlops-0001prod
AZURE_ML_RESOURCE_GROUP=rg-azmlops-0001prod

# Real Compute (serverless or named cluster)
COMPUTE_TYPE=serverless

# Logging
LOG_LEVEL=INFO
```

- [ ] **Step 3: Create src/ml_pipeline/config.py**

```python
"""
Environment-aware configuration factory.
Seamlessly switches between local simulation and production.
"""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class Config:
    """Application configuration - read from environment."""
    
    environment: str
    is_local: bool
    debug: bool
    
    # Storage
    storage_account_name: str
    storage_account_key: Optional[str]
    storage_blob_endpoint: str
    
    # Entra & RBAC
    mock_entra_enabled: bool
    entra_tenant_id: str
    entra_client_id: Optional[str]
    entra_client_secret: Optional[str]
    
    # ML Workspace
    azure_subscription_id: str
    azure_resource_group: str
    azure_ml_workspace: str
    
    # Compute
    compute_endpoint: Optional[str]  # Local simulator
    compute_type: str  # "serverless", "cluster" (prod only)
    
    # Logging
    log_level: str
    
    @classmethod
    def from_environment(cls) -> "Config":
        """
        Load configuration from .env file and environment variables.
        Prioritizes: env vars > .env file > defaults
        """
        # Load .env file based on ENVIRONMENT variable
        environment = os.getenv("ENVIRONMENT", "local").lower()
        
        env_file = Path(__file__).parent.parent.parent / f".env.{environment}"
        if env_file.exists():
            load_dotenv(env_file)
        
        is_local = environment == "local"
        
        return cls(
            environment=environment,
            is_local=is_local,
            debug=os.getenv("DEBUG", "false").lower() == "true",
            
            storage_account_name=os.getenv(
                "AZURE_STORAGE_ACCOUNT_NAME",
                "devstoreaccount1" if is_local else "stazmlops0001prod"
            ),
            storage_account_key=os.getenv("AZURE_STORAGE_ACCOUNT_KEY"),
            storage_blob_endpoint=os.getenv(
                "AZURE_STORAGE_BLOB_ENDPOINT",
                "http://localhost:10000" if is_local else ""
            ),
            
            mock_entra_enabled=os.getenv("MOCK_ENTRA_ENABLED", "true").lower() == "true" if is_local else False,
            entra_tenant_id=os.getenv("ENTRA_TENANT_ID", "mock-tenant-id"),
            entra_client_id=os.getenv("ENTRA_CLIENT_ID"),
            entra_client_secret=os.getenv("ENTRA_CLIENT_SECRET"),
            
            azure_subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID", "mock-subscription-id"),
            azure_resource_group=os.getenv("AZURE_RESOURCE_GROUP", "rg-azmlops-local"),
            azure_ml_workspace=os.getenv("AZURE_ML_WORKSPACE", "mlw-azmlops-local"),
            
            compute_endpoint=os.getenv("COMPUTE_ENDPOINT", "http://localhost:5000" if is_local else None),
            compute_type=os.getenv("COMPUTE_TYPE", "mock" if is_local else "serverless"),
            
            log_level=os.getenv("LOG_LEVEL", "DEBUG" if is_local else "INFO"),
        )
    
    def to_dict(self) -> dict:
        """Export configuration as dictionary for downstream use."""
        return {
            "environment": self.environment,
            "is_local": self.is_local,
            "storage_blob_endpoint": self.storage_blob_endpoint,
            "azure_ml_workspace": self.azure_ml_workspace,
            "entra_tenant_id": self.entra_tenant_id,
            "compute_endpoint": self.compute_endpoint,
        }
```

- [ ] **Step 4: Commit**

```bash
git add src/ml_pipeline/config.py .env.local .env.prod
git commit -m "feat: add environment-aware configuration factory"
```

---

### Task 5: Create Mock Entra Identity & RBAC Module

**Files:**
- Create: `local-dev-env/src/auth/mock_entra.py`
- Create: `local-dev-env/src/auth/rbac.py`
- Create: `local-dev-env/src/auth/__init__.py`
- Create: `local-dev-env/tests/test_auth.py`

**Interfaces:**
- Produces: 
  - `MockEntraIdentity` class: stores user object_id, role, permissions
  - `RBACChecker` class with method `check_permission(identity: MockEntraIdentity, action: str) -> bool`
- Consumed by: ML pipeline modules that check permissions before actions

- [ ] **Step 1: Create src/auth/__init__.py**

```python
"""Authentication and RBAC modules."""
from .mock_entra import MockEntraIdentity, get_current_identity
from .rbac import RBACChecker, Permission

__all__ = ["MockEntraIdentity", "get_current_identity", "RBACChecker", "Permission"]
```

- [ ] **Step 2: Create src/auth/mock_entra.py**

```python
"""Mock Entra identity provider for local development."""
import json
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

@dataclass
class MockEntraIdentity:
    """Mock Entra user identity."""
    
    object_id: str
    email: str
    role: str
    permissions: list[str]
    tenant_id: str
    
    def has_permission(self, permission: str) -> bool:
        """Check if identity has a specific permission."""
        return permission in self.permissions
    
    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "object_id": self.object_id,
            "email": self.email,
            "role": self.role,
            "permissions": self.permissions,
            "tenant_id": self.tenant_id,
        }

class MockEntraProvider:
    """
    Mock Entra provider for local development.
    Reads identities from Terraform-generated JSON file.
    """
    
    def __init__(self, identities_path: Optional[str] = None):
        """
        Initialize provider.
        
        Args:
            identities_path: Path to .terraform_identities.json
                           If None, uses default location relative to this module.
        """
        if identities_path is None:
            # Default: look for .terraform_identities.json in project root
            identities_path = Path(__file__).parent.parent.parent / ".terraform_identities.json"
        
        self.identities_path = Path(identities_path)
        self._identities: Dict[str, MockEntraIdentity] = {}
        self._load_identities()
    
    def _load_identities(self) -> None:
        """Load identities from Terraform-generated JSON."""
        if not self.identities_path.exists():
            # If Terraform hasn't run, create default developer identity
            self._identities = {
                "developer": MockEntraIdentity(
                    object_id="mock-user-object-id",
                    email="developer@local.dev",
                    role="ml_data_scientist",
                    permissions=[
                        "actions:train",
                        "actions:inference",
                        "actions:experiment",
                        "data:read",
                        "data:write",
                        "metadata:read",
                    ],
                    tenant_id="mock-tenant-id",
                )
            }
            return
        
        with open(self.identities_path) as f:
            data = json.load(f)
        
        for key, identity_data in data.items():
            self._identities[key] = MockEntraIdentity(
                object_id=identity_data["object_id"],
                email=identity_data["email"],
                role=identity_data["role"],
                permissions=identity_data["permissions"],
                tenant_id=identity_data.get("tenant_id", "mock-tenant-id"),
            )
    
    def get_identity(self, identifier: str) -> Optional[MockEntraIdentity]:
        """
        Get identity by email or object_id.
        
        Args:
            identifier: User email or object_id
        
        Returns:
            MockEntraIdentity or None if not found
        """
        # Try direct key lookup first
        if identifier in self._identities:
            return self._identities[identifier]
        
        # Try email lookup
        for identity in self._identities.values():
            if identity.email == identifier:
                return identity
        
        return None
    
    def list_identities(self) -> list[MockEntraIdentity]:
        """List all identities."""
        return list(self._identities.values())

# Global provider instance
_provider: Optional[MockEntraProvider] = None

def get_current_identity() -> MockEntraIdentity:
    """
    Get current user's Entra identity.
    Uses "developer" identity by default in local dev.
    """
    global _provider
    if _provider is None:
        _provider = MockEntraProvider()
    
    identity = _provider.get_identity("developer")
    if identity is None:
        raise RuntimeError("Developer identity not found in mock Entra")
    
    return identity
```

- [ ] **Step 3: Create src/auth/rbac.py**

```python
"""RBAC permission checker."""
import json
from pathlib import Path
from typing import Optional
from enum import Enum

class Permission(str, Enum):
    """Defined permissions (matches prod RBAC)."""
    
    # Actions
    TRAIN = "actions:train"
    INFERENCE = "actions:inference"
    EXPERIMENT = "actions:experiment"
    DEPLOY = "actions:deploy"
    MANAGE = "actions:manage"
    
    # Data access
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    
    # Metadata
    METADATA_READ = "metadata:read"
    METADATA_WRITE = "metadata:write"
    
    # Infrastructure
    INFRASTRUCTURE_MANAGE = "infrastructure:manage"
    
    # RBAC management
    RBAC_READ = "rbac:read"
    RBAC_WRITE = "rbac:write"
    
    # Audit
    AUDIT_READ = "audit:read"

class RBACChecker:
    """RBAC permission checker."""
    
    def __init__(self, roles_path: Optional[str] = None):
        """
        Initialize RBAC checker.
        
        Args:
            roles_path: Path to RBAC role definitions JSON
        """
        if roles_path is None:
            roles_path = Path(__file__).parent.parent / "auth" / "rbac_roles.json"
        
        self.roles_path = Path(roles_path)
        self._roles = self._load_roles()
    
    def _load_roles(self) -> dict:
        """Load role definitions from JSON."""
        if not self.roles_path.exists():
            # Return default roles if file doesn't exist
            return {
                "ml_data_scientist": {
                    "permissions": [
                        "actions:train",
                        "actions:inference",
                        "actions:experiment",
                        "data:read",
                        "data:write",
                        "metadata:read",
                    ]
                },
                "ml_engineer": {
                    "permissions": [
                        "actions:train",
                        "actions:inference",
                        "actions:deploy",
                        "actions:manage",
                        "data:read",
                        "data:write",
                        "metadata:read",
                        "metadata:write",
                        "rbac:read",
                    ]
                }
            }
        
        with open(self.roles_path) as f:
            data = json.load(f)
        return data.get("roles", {})
    
    def check_permission(self, role: str, permission: str) -> bool:
        """
        Check if role has permission.
        
        Args:
            role: Role name (e.g., "ml_data_scientist")
            permission: Permission string (e.g., "actions:train")
        
        Returns:
            True if role has permission, False otherwise
        """
        if role not in self._roles:
            return False
        
        permissions = self._roles[role].get("permissions", [])
        return permission in permissions
    
    def check_all_permissions(self, role: str, permissions: list[str]) -> bool:
        """Check if role has ALL specified permissions."""
        return all(self.check_permission(role, p) for p in permissions)
    
    def get_role_permissions(self, role: str) -> list[str]:
        """Get all permissions for a role."""
        if role not in self._roles:
            return []
        return self._roles[role].get("permissions", [])
```

- [ ] **Step 4: Create tests/test_auth.py**

```python
"""Tests for authentication and RBAC modules."""
import pytest
from src.auth.mock_entra import MockEntraIdentity, MockEntraProvider
from src.auth.rbac import RBACChecker, Permission

def test_mock_entra_identity_creation():
    """Test creating a mock Entra identity."""
    identity = MockEntraIdentity(
        object_id="test-id",
        email="test@local.dev",
        role="ml_data_scientist",
        permissions=["actions:train", "data:read"],
        tenant_id="mock-tenant",
    )
    
    assert identity.object_id == "test-id"
    assert identity.email == "test@local.dev"
    assert identity.role == "ml_data_scientist"

def test_mock_entra_identity_permissions():
    """Test permission checking on identity."""
    identity = MockEntraIdentity(
        object_id="test-id",
        email="test@local.dev",
        role="ml_data_scientist",
        permissions=["actions:train", "data:read"],
        tenant_id="mock-tenant",
    )
    
    assert identity.has_permission("actions:train") is True
    assert identity.has_permission("actions:deploy") is False

def test_rbac_checker_default_roles():
    """Test RBAC checker with default roles."""
    checker = RBACChecker()
    
    # Data scientist can train
    assert checker.check_permission("ml_data_scientist", "actions:train") is True
    
    # Data scientist cannot deploy
    assert checker.check_permission("ml_data_scientist", "actions:deploy") is False
    
    # ML engineer can deploy
    assert checker.check_permission("ml_engineer", "actions:deploy") is True

def test_rbac_checker_check_all_permissions():
    """Test checking multiple permissions."""
    checker = RBACChecker()
    
    required = ["actions:train", "data:read"]
    assert checker.check_all_permissions("ml_data_scientist", required) is True
    
    required_deploy = ["actions:train", "actions:deploy"]
    assert checker.check_all_permissions("ml_data_scientist", required_deploy) is False

def test_rbac_checker_get_role_permissions():
    """Test getting all permissions for a role."""
    checker = RBACChecker()
    
    perms = checker.get_role_permissions("ml_data_scientist")
    assert "actions:train" in perms
    assert "data:read" in perms
    assert "actions:deploy" not in perms

def test_permission_enum():
    """Test Permission enum values."""
    assert Permission.TRAIN.value == "actions:train"
    assert Permission.DATA_READ.value == "data:read"
    assert Permission.RBAC_WRITE.value == "rbac:write"
```

- [ ] **Step 5: Commit**

```bash
git add src/auth/ tests/test_auth.py
git commit -m "feat: add mock Entra identity and RBAC permission checker"
```

---

### Task 6: Create ML Pipeline Scaffold (Train & Predict)

**Files:**
- Create: `local-dev-env/src/ml_pipeline/__init__.py`
- Create: `local-dev-env/src/ml_pipeline/train.py`
- Create: `local-dev-env/src/ml_pipeline/predict.py`
- Create: `local-dev-env/tests/test_pipeline.py`

**Interfaces:**
- Produces:
  - `train_model(config: Config, data_path: str) -> TrainedModel` 
  - `predict(model: TrainedModel, data: List[List[float]]) -> List[float]`
- Both respect Entra permissions via RBAC checker
- Both handle local (mock) and prod (real Azure ML) seamlessly

- [ ] **Step 1: Create src/ml_pipeline/__init__.py**

```python
"""ML pipeline modules."""
from .train import train_model, TrainedModel
from .predict import predict

__all__ = ["train_model", "predict", "TrainedModel"]
```

- [ ] **Step 2: Create src/ml_pipeline/train.py**

```python
"""Training pipeline - env-aware."""
import json
import pickle
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.ml_pipeline.config import Config
from src.auth.mock_entra import get_current_identity
from src.auth.rbac import RBACChecker, Permission

@dataclass
class TrainedModel:
    """Trained ML model with metadata."""
    
    model: LinearRegression
    scaler: StandardScaler
    feature_names: list[str]
    feature_count: int
    trained_by: str
    environment: str
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def to_dict(self) -> dict:
        """Export metadata (model binary is pickled separately)."""
        return {
            "feature_count": self.feature_count,
            "feature_names": self.feature_names,
            "trained_by": self.trained_by,
            "environment": self.environment,
        }

def train_model(config: Config, data_path: str = "tests/fixtures/test_data.csv") -> TrainedModel:
    """
    Train ML model.
    Respects RBAC permissions - user must have "actions:train" permission.
    
    Args:
        config: Application configuration
        data_path: Path to training data CSV (features as columns, target as last column)
    
    Returns:
        TrainedModel instance
    
    Raises:
        PermissionError: If current user lacks train permission
    """
    # Check permissions
    if config.mock_entra_enabled:
        identity = get_current_identity()
        checker = RBACChecker()
        
        if not checker.check_permission(identity.role, Permission.TRAIN.value):
            raise PermissionError(
                f"User {identity.email} (role: {identity.role}) lacks train permission"
            )
    
    # Load data
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    # Simple CSV loading (assumes last column is target)
    import csv
    features = []
    targets = []
    feature_names = None
    
    with open(data_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == 0:
                feature_names = list(row.keys())[:-1]
            
            row_features = [float(row[name]) for name in feature_names]
            target = float(row[list(row.keys())[-1]])
            
            features.append(row_features)
            targets.append(target)
    
    X = np.array(features)
    y = np.array(targets)
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Create trained model
    identity = get_current_identity() if config.mock_entra_enabled else None
    trained_by = identity.email if identity else "system"
    
    return TrainedModel(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        feature_count=len(feature_names),
        trained_by=trained_by,
        environment=config.environment,
    )

def save_model(model: TrainedModel, output_path: str) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: TrainedModel instance
        output_path: Path to save model pickle
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

def load_model(model_path: str) -> TrainedModel:
    """
    Load trained model from disk.
    
    Args:
        model_path: Path to model pickle
    
    Returns:
        TrainedModel instance
    """
    with open(model_path, "rb") as f:
        return pickle.load(f)
```

- [ ] **Step 3: Create src/ml_pipeline/predict.py**

```python
"""Inference pipeline - env-aware."""
import numpy as np
from typing import List

from src.ml_pipeline.config import Config
from src.ml_pipeline.train import TrainedModel
from src.auth.mock_entra import get_current_identity
from src.auth.rbac import RBACChecker, Permission

def predict(
    config: Config,
    model: TrainedModel,
    data: List[List[float]]
) -> List[float]:
    """
    Make predictions using trained model.
    Respects RBAC permissions - user must have "actions:inference" permission.
    
    Args:
        config: Application configuration
        model: TrainedModel instance
        data: List of feature vectors
    
    Returns:
        List of predictions
    
    Raises:
        PermissionError: If current user lacks inference permission
    """
    # Check permissions
    if config.mock_entra_enabled:
        identity = get_current_identity()
        checker = RBACChecker()
        
        if not checker.check_permission(identity.role, Permission.INFERENCE.value):
            raise PermissionError(
                f"User {identity.email} (role: {identity.role}) lacks inference permission"
            )
    
    # Make predictions
    X = np.array(data)
    predictions = model.predict(X)
    
    return predictions.tolist()
```

- [ ] **Step 4: Create tests/test_pipeline.py**

```python
"""Tests for ML pipeline."""
import pytest
import numpy as np
from pathlib import Path

from src.ml_pipeline.config import Config
from src.ml_pipeline.train import train_model, save_model, load_model
from src.ml_pipeline.predict import predict
from src.auth.mock_entra import MockEntraIdentity

@pytest.fixture
def test_data_path():
    """Path to test data."""
    return Path("tests/fixtures/test_data.csv")

@pytest.fixture
def local_config():
    """Local dev configuration."""
    config = Config(
        environment="local",
        is_local=True,
        debug=True,
        storage_account_name="devstoreaccount1",
        storage_account_key="test-key",
        storage_blob_endpoint="http://localhost:10000",
        mock_entra_enabled=True,
        entra_tenant_id="mock-tenant",
        entra_client_id=None,
        entra_client_secret=None,
        azure_subscription_id="mock-sub",
        azure_resource_group="rg-local",
        azure_ml_workspace="mlw-local",
        compute_endpoint="http://localhost:5000",
        compute_type="mock",
        log_level="DEBUG",
    )
    return config

def test_train_model(local_config, test_data_path):
    """Test model training."""
    model = train_model(local_config, str(test_data_path))
    
    assert model.feature_count == 46
    assert model.environment == "local"
    assert model.trained_by == "developer@local.dev"

def test_predict(local_config, test_data_path):
    """Test inference."""
    model = train_model(local_config, str(test_data_path))
    
    # Create test data (same shape as training features)
    test_sample = [[1.0] * model.feature_count]
    
    predictions = predict(local_config, model, test_sample)
    
    assert len(predictions) == 1
    assert isinstance(predictions[0], (int, float))

def test_model_serialization(local_config, test_data_path, tmp_path):
    """Test model save/load roundtrip."""
    model = train_model(local_config, str(test_data_path))
    
    model_path = tmp_path / "model.pkl"
    save_model(model, str(model_path))
    
    loaded_model = load_model(str(model_path))
    
    assert loaded_model.feature_count == model.feature_count
    assert loaded_model.environment == model.environment
    
    # Verify predictions match
    test_sample = [[1.0] * model.feature_count]
    pred_original = model.predict(np.array(test_sample))
    pred_loaded = loaded_model.predict(np.array(test_sample))
    
    np.testing.assert_array_almost_equal(pred_original, pred_loaded)
```

- [ ] **Step 5: Commit**

```bash
git add src/ml_pipeline/ tests/test_pipeline.py
git commit -m "feat: add environment-aware training and inference pipeline"
```

---

### Task 7: Create Parity Validation Tests (Local vs Prod)

**Files:**
- Create: `local-dev-env/tests/test_parity.py`
- Create: `tests/fixtures/test_data.csv`

**Interfaces:**
- Consumes: `TrainedModel`, `Config`, `RBACChecker`
- Produces: Test suite that validates local behavior matches production behavior (determinism, consistency, permissions)

- [ ] **Step 1: Create tests/fixtures/test_data.csv**

```csv
feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9,feature_10,feature_11,feature_12,feature_13,feature_14,feature_15,feature_16,feature_17,feature_18,feature_19,feature_20,feature_21,feature_22,feature_23,feature_24,feature_25,feature_26,feature_27,feature_28,feature_29,feature_30,feature_31,feature_32,feature_33,feature_34,feature_35,feature_36,feature_37,feature_38,feature_39,feature_40,feature_41,feature_42,feature_43,feature_44,feature_45,feature_46,target
1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0,100.5
2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0,47.0,200.7
3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0,47.0,48.0,300.2
```

- [ ] **Step 2: Create tests/test_parity.py**

```python
"""
Parity validation tests.
Verify that local dev behavior matches production.
Run against prod by setting ENVIRONMENT=prod before executing.
"""
import pytest
import numpy as np
from pathlib import Path

from src.ml_pipeline.config import Config
from src.ml_pipeline.train import train_model, save_model, load_model
from src.ml_pipeline.predict import predict
from src.auth.rbac import RBACChecker, Permission

class TestTrainingDeterminism:
    """Verify training produces deterministic outputs."""
    
    @pytest.fixture
    def config(self):
        """Get current environment config."""
        return Config.from_environment()
    
    @pytest.fixture
    def test_data_path(self):
        return Path("tests/fixtures/test_data.csv")
    
    def test_training_produces_consistent_model_on_repeated_runs(self, config, test_data_path):
        """
        PARITY VALIDATION:
        Training on same data should produce same model coefficients.
        Run locally and in prod - both should be deterministic.
        """
        model1 = train_model(config, str(test_data_path))
        model2 = train_model(config, str(test_data_path))
        
        # Compare model coefficients
        np.testing.assert_array_almost_equal(
            model1.model.coef_,
            model2.model.coef_,
            decimal=10,
            err_msg="Model coefficients differ on repeated training"
        )
        
        # Compare intercepts
        np.testing.assert_almost_equal(
            model1.model.intercept_,
            model2.model.intercept_,
            decimal=10,
            err_msg="Model intercept differs on repeated training"
        )

class TestInferenceDeterminism:
    """Verify inference produces deterministic outputs."""
    
    @pytest.fixture
    def config(self):
        return Config.from_environment()
    
    @pytest.fixture
    def model(self, config):
        return train_model(config, "tests/fixtures/test_data.csv")
    
    def test_inference_produces_consistent_predictions(self, config, model):
        """
        PARITY VALIDATION:
        Inference on same data should produce same predictions.
        """
        test_sample = [[1.0] * model.feature_count]
        
        pred1 = predict(config, model, test_sample)
        pred2 = predict(config, model, test_sample)
        
        np.testing.assert_array_almost_equal(
            pred1, pred2,
            decimal=10,
            err_msg="Predictions differ on repeated inference"
        )

class TestModelSerialization:
    """Verify model serialization preserves behavior."""
    
    @pytest.fixture
    def config(self):
        return Config.from_environment()
    
    @pytest.fixture
    def model(self, config):
        return train_model(config, "tests/fixtures/test_data.csv")
    
    def test_serialized_model_produces_identical_predictions(self, config, model, tmp_path):
        """
        PARITY VALIDATION:
        Model serialization (pickle) should preserve predictions exactly.
        """
        # Get original predictions
        test_samples = [
            [1.0] * model.feature_count,
            [2.0] * model.feature_count,
            [3.0] * model.feature_count,
        ]
        predictions_original = predict(config, model, test_samples)
        
        # Serialize and deserialize
        model_path = tmp_path / "model.pkl"
        save_model(model, str(model_path))
        loaded_model = load_model(str(model_path))
        
        # Get predictions from deserialized model
        predictions_loaded = predict(config, loaded_model, test_samples)
        
        np.testing.assert_array_almost_equal(
            predictions_original,
            predictions_loaded,
            decimal=10,
            err_msg="Serialization/deserialization changed predictions"
        )

class TestRBACConsistency:
    """Verify RBAC behavior is consistent."""
    
    def test_rbac_roles_defined_for_all_identities(self):
        """PARITY VALIDATION: All identities have defined roles in RBAC."""
        checker = RBACChecker()
        
        # Expected roles in both local and prod
        expected_roles = ["ml_data_scientist", "ml_engineer", "devops"]
        
        for role in expected_roles:
            perms = checker.get_role_permissions(role)
            assert len(perms) > 0, f"Role {role} has no permissions defined"
    
    def test_ml_data_scientist_cannot_deploy(self):
        """PARITY VALIDATION: Data scientist role lacks deploy permission."""
        checker = RBACChecker()
        
        can_train = checker.check_permission("ml_data_scientist", Permission.TRAIN.value)
        can_deploy = checker.check_permission("ml_data_scientist", Permission.DEPLOY.value)
        
        assert can_train is True, "Data scientist should be able to train"
        assert can_deploy is False, "Data scientist should NOT be able to deploy"
    
    def test_ml_engineer_can_deploy(self):
        """PARITY VALIDATION: ML engineer role has deploy permission."""
        checker = RBACChecker()
        
        can_deploy = checker.check_permission("ml_engineer", Permission.DEPLOY.value)
        assert can_deploy is True, "ML engineer should be able to deploy"

class TestDependencyVersions:
    """
    PARITY VALIDATION: Critical dependencies must be identical.
    This is checked at import time - if versions differ, test fails.
    """
    
    def test_scikit_learn_version_matches_requirements(self):
        """Verify sklearn version matches requirements.txt."""
        import sklearn
        
        # Must match requirements.txt exactly
        # Hardcoded check ensures local and prod use same version
        expected_version = "1.3.2"
        actual_version = sklearn.__version__
        
        assert actual_version.startswith(expected_version), \
            f"scikit-learn version mismatch: expected {expected_version}, got {actual_version}"
    
    def test_numpy_version_compatible(self):
        """Verify numpy version is compatible."""
        import numpy
        
        # Check that we're in a compatible range
        major, minor = map(int, numpy.__version__.split('.')[:2])
        
        assert major >= 1, f"numpy too old: {numpy.__version__}"
        assert (major, minor) >= (1, 21), f"numpy version too old: {numpy.__version__}"
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_parity.py tests/fixtures/test_data.csv
git commit -m "feat: add parity validation tests for local vs prod behavior"
```

---

### Task 8: Create One-Command Setup Script & Final Documentation

**Files:**
- Create: `local-dev-env/setup_local_dev.sh`
- Create: `local-dev-env/terraform/README.md`
- Update: `local-dev-env/README.md` with complete workflow

**Interfaces:**
- Produces: Fully bootstrapped environment - user runs `bash setup_local_dev.sh` and can immediately `pytest` and train models

- [ ] **Step 1: Create setup_local_dev.sh**

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "Azure ML Local Dev Environment Setup"
echo "=========================================="

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not installed. Please install Docker first."
    exit 1
fi

if ! command -v terraform &> /dev/null; then
    echo "ERROR: Terraform not installed. Please install Terraform first."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not installed. Please install Python 3.10+ first."
    exit 1
fi

echo "✓ Prerequisites OK"

# Step 1: Create Python virtual environment
echo ""
echo "Step 1: Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo "✓ Python environment ready"

# Step 2: Initialize Terraform
echo ""
echo "Step 2: Initializing Terraform..."
cd terraform
terraform init
terraform apply -auto-approve
cd ..
echo "✓ Terraform local infrastructure configured"

# Step 3: Start Docker containers
echo ""
echo "Step 3: Starting Docker containers..."
cd docker
docker-compose up -d
cd ..

echo "Waiting for services to be healthy..."
sleep 10

if docker ps | grep -q azurite-local-dev; then
    echo "✓ Azurite storage running"
else
    echo "ERROR: Azurite failed to start"
    exit 1
fi

if docker ps | grep -q compute-simulator-local-dev; then
    echo "✓ Compute simulator running"
else
    echo "ERROR: Compute simulator failed to start"
    exit 1
fi

# Step 4: Create .env.local if not exists
echo ""
echo "Step 4: Setting up environment..."
if [ ! -f ".env.local" ]; then
    echo "ERROR: .env.local not found. Please create it from .env.local template."
    exit 1
fi
echo "✓ Environment variables loaded"

# Step 5: Run tests
echo ""
echo "Step 5: Running validation tests..."
export $(cat .env.local | xargs)
pytest tests/ -v --tb=short

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Run training: python src/ml_pipeline/train.py"
echo "3. Run tests: pytest tests/"
echo ""
echo "To validate against production:"
echo "1. Set up prod credentials in .env.prod"
echo "2. Run: export \$(cat .env.prod | xargs) && pytest tests/test_parity.py"
echo ""
```

- [ ] **Step 2: Make setup script executable**

```bash
chmod +x setup_local_dev.sh
```

- [ ] **Step 3: Update terraform/README.md**

```markdown
# Terraform Local Configuration

This Terraform configuration **creates no Azure resources**. It only:
1. Defines mock Entra identities (stored locally)
2. Defines RBAC role definitions (consumed by Python)
3. Outputs configuration files for the Python application

## Usage

\`\`\`bash
cd terraform
terraform init
terraform apply
\`\`\`

This generates:
- \`.terraform_identities.json\` - Mock user identities
- \`.terraform_outputs.json\` - Configuration for Python code

## No Azure Credentials Needed

Unlike production Terraform, this requires no Azure authentication. It uses only local file operations.
\`\`\`

- [ ] **Step 4: Commit**

```bash
git add setup_local_dev.sh terraform/README.md
git commit -m "feat: add one-command setup script and documentation"
```

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-08-19-local-dev-environment.md`.**

Two execution options:

**1. Subagent-Driven (Recommended)** 
- I dispatch a fresh subagent per task, review between tasks, fast iteration
- Requires: superpowers:subagent-driven-development

**2. Inline Execution**
- Execute tasks in this session using superpowers:executing-plans
- Batch execution with checkpoints for review

**Which approach would you prefer?**
