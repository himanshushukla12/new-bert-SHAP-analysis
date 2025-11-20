# Dual Domain ML Framework: Mental Health Apps & Quick Commerce

This project implements an explainable ML pipeline to predict user satisfaction determinants from unstructured text reviews. It supports two domains: Mental Health Apps (MHA) and Quick Commerce.

## 0. Environment Setup (MANDATORY - USING UV)

This project uses `uv` for fast package management.

### Install uv (if not installed)
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Create virtual environment
```bash
uv venv .venv
```

### Activate environment

**MacOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\activate
```

### Install dependencies using uv
```bash
uv pip install -r requirements.txt
```

### Freeze dependencies
```bash
uv pip freeze > requirements.txt
```

## 1. Running the Pipeline

### Launch Streamlit App
The easiest way to interact with the project is via the Streamlit dashboard.

```bash
streamlit run app/streamlit_app.py
```

### Run Tests
```bash
pytest tests/
```

## 2. Project Structure
- `data/`: Place your CSV files here.
- `src/`: Core source code.
- `app/`: Streamlit application.
- `docs/`: Detailed documentation.
