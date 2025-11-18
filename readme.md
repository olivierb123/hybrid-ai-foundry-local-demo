# Hybrid AI Symptom Checker – Azure AI Foundry + Foundry Local

This repository contains a minimal example demonstrating how to build a **hybrid AI workflow** using:

- **Azure AI Foundry** (cloud intelligence)
- **Agent Framework** (tool routing and orchestration)
- **Foundry Local** (local GPU inference)

The example shows how a cloud-hosted agent can safely call a **local tool** to process sensitive data—such as medical lab reports—**without sending any raw PHI/PII to the cloud**. Only a structured JSON summary produced locally on the user’s machine is passed to the cloud agent.

---

## Project Structure

```
src/
  hybrid_symptom_checker.py   # Main hybrid agent demo
  requirements.txt            # Python dependencies
README.md
```

---

## Prerequisites

- Python 3.10+
- Git
- Azure CLI (`az`)
- Access to **Azure AI Foundry**
- **Foundry Local** installed and the service running
- A deployed cloud model (e.g., **GPT-4o**) in your Azure AI Foundry project
- A local model available in Foundry Local (e.g., **Phi-4-mini-instruct-cuda-gpu:5**)

---

## Setup

```bash
# Install dependencies
pip install -r src/requirements.txt

# Login to Azure
az login
```

Verify Foundry Local is running:

```bash
foundry service status
foundry model list
```

---

## Run the Hybrid Demo

```bash
python src/hybrid_symptom_checker.py
```

The script will:

1. Send the user’s case and raw lab report to the cloud agent  
2. The agent detects lab data and calls the **local summarization tool**  
3. Foundry Local runs inference on your GPU  
4. The cloud agent uses the structured JSON summary to generate final guidance  

---

## How It Works

This project demonstrates:

- **Local privacy:** raw lab report text stays completely on-device  
- **Cloud reasoning:** GPT-4o handles triage logic and instruction adherence  
- **Tool abstraction:** a Python function decorated with `@ai_function` becomes a tool callable by the agent  
- **Unified developer experience:** the agent routes between cloud and local seamlessly  

---

## License

MIT License
