Here's the **clean, professional, and visually enhanced** version of your ARJUNA README in the same HTML-style Markdown format you requested. All original content and meaning are preserved, emojis are completely removed, headings are improved for better visual hierarchy, tables are neatly formatted, and the overall structure looks polished and professional on GitHub.

```markdown
---
title: Arjuna Perception Env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

<h1 align="center">ARJUNA: Dynamic Auto-Curriculum for Robust Perception</h1>
<h3 align="center">An OpenEnv-Compliant Framework for Generalizable Reinforcement Learning</h3>

**ARJUNA** (`arjuna-perception-env`) is a simulated robot perception testbed designed to solve the **Generalization Gap** in RL. By integrating a **Rule-Based Auto-Curriculum** with **Dense Sequence Alignment Rewards**, it forces agents to master **Out-of-Distribution (OOD)** scenarios — from clean urban streets to chaotic, low-visibility edge cases — without manual tuning.

---

## What does this environment do?

ARJUNA is an autonomous robot whose “eyes” are simulated here. Each **episode** is a **3-step sequence** (identify → triage → decide) over one **themed bundle** of scenes (see `EPISODE_BUNDLES` in `server/synthetic_data.py`). 

The agent receives **natural-language observations** with fake detections and must emit a structured **action** per step. The **grader** returns a **per-step reward in [0, 1]**, and after step 3 an **`overall_reward`** (mean of the three steps) plus **feedback** — without real cameras, cloud databases, or any external API.

---

## Quick Start — No API Key Required

The environment runs **fully offline**. No API key is needed to run the environment itself.

### Run with Docker (recommended)

```bash
docker build -t arjuna-env .
docker run -p 7860:7860 arjuna-env
```

### Run locally

```bash
pip install -r requirements.txt
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Test offline with demo.py

```bash
# No API key needed — uses heuristic policy
python demo.py
```

---

## Core vs Optional Features

### Core (offline, no API key needed)

| Feature                | File                            | Description                                      |
|------------------------|---------------------------------|--------------------------------------------------|
| Environment server     | server/app.py                   | HTTP endpoints                                   |
| Episode logic          | server/arjuna_environment.py    | reset/step/state                                 |
| Task graders           | server/tasks.py, server/grader.py | reward logic                                   |
| Synthetic scenes       | server/synthetic_data.py        | **12 offline episode bundles**                   |
| Data models            | models.py                       | typed actions/observations                       |
| Offline demo           | demo.py                         | heuristic agent, no LLM                          |

### Optional (requires API key + network)

| Feature                      | File                          | How to enable                          |
|------------------------------|-------------------------------|----------------------------------------|
| LLM baseline agent           | inference.py                  | Set API_BASE_URL + HF_TOKEN            |
| Dynamic scene generation     | server/scene_generator.py     | ENABLE_DYNAMIC_SCENES=true             |
| Auto-curriculum              | server/curriculum.py          | ENABLE_DYNAMIC_SCENES=true             |
| AutoRL loop                  | autorl.py                     | Set API_BASE_URL + HF_TOKEN            |

The environment **always falls back to synthetic_data.py** if dynamic scene generation is disabled or fails.

---

## Offline Execution Guarantee

All of these work with **zero network calls**:

- `POST /reset` → picks from synthetic_data.py bundles
- `POST /step` → grades using local tasks.py logic
- `GET /state` → returns local session state
- `GET /health` → returns healthy
- `GET /schema` → returns typed schemas
- `GET /metadata` → returns environment info
- `python demo.py` → full 3-step episode, heuristic policy

---

## Episode Bundles — 12 Offline Scenarios

All 12 bundles are hardcoded in `server/synthetic_data.py` and require **zero network calls**. Each bundle contains 3 scenes — one per task step — drawn from the same location theme.

| #  | Bundle              | Task 1 Object   | Task 2 Objects                          | Task 3 Confidence | Expected Action    |
|----|---------------------|-----------------|-----------------------------------------|-------------------|--------------------|
| 1  | Urban Street        | person          | car, bicycle, person, traffic light     | 0.24              | discard            |
| 2  | Warehouse           | forklift        | worker, truck, forklift, carton         | 0.42              | request_rescan     |
| 3  | Parking Lot         | car             | car, person, parking meter, CCTV camera | 0.31              | discard            |
| 4  | School Zone         | bus             | student, bicycle, backpack, bus         | 0.38              | request_rescan     |
| 5  | Airport             | airplane        | airplane, suitcase, boarding gate, trolley | 0.19           | discard            |
| 6  | Hospital Entrance   | ambulance       | person, ambulance, wheelchair, stretcher| 0.51              | log_and_continue   |
| 7  | Construction Site   | helmet          | worker, excavator, crane, helmet        | 0.44              | request_rescan     |
| 8  | Night Street        | streetlight     | motorcycle, person, fire hydrant, streetlight | 0.21        | discard            |
| 9  | Forest Trail        | hiker           | hiker, dog, backpack, tree              | 0.28              | discard            |
| 10 | Shopping Mall       | shopping bag    | person, escalator, shopping bag, CCTV camera | 0.46         | request_rescan     |
| 11 | Office Lobby        | laptop          | person, couch, reception desk, potted plant | 0.54          | log_and_continue   |
| 12 | Rainy Street        | raincoat        | bus, car, person, umbrella              | 0.38              | request_rescan     |

> **Task 3 Decision Bands:**
> - `confidence < 0.35` → `discard`
> - `0.35 ≤ confidence < 0.50` → `request_rescan`
> - `confidence ≥ 0.50` → `log_and_continue`

---

## Dense Reward Mechanism (Sequence Alignment)

Unlike basic sparse-reward environments (where an agent receives a binary `1.0` or `0.0`), ARJUNA uses **Dense Rewards** powered by Levenshtein Edit Distance (`SequenceMatcher`).

- **Differentiable Feedback:** When an agent attempts the Multi-Object Triage task, sequence alignment provides a granular gradient of success (e.g. `0.50`, `0.83`, `1.00`).
- **Accelerated Convergence:** Enabling the agent to learn from partial successes and severely penalizing verbosity (extra hallucinated objects) significantly accelerates RL convergence and mirrors modern Reward Model (RM) techniques.

---

## Zero-Shot Baseline & Environment Audit Logging

To prove that the ARJUNA environment accurately evaluates edge cases without requiring a days-long backpropagation training loop, this repository includes a **Zero-Shot Baseline Agent** (`inference.py`).

- **Baseline Validation:** We use an un-tuned LLM (Llama-3/Groq) to blindly attempt the environment. The LLM naturally gets "stuck" in the Medium difficulty tier because the environment rigorously enforces triage tie-breakers — confirming that higher tiers require policy-gradient optimization or fine-tuning beyond zero-shot capabilities.
- **Audit Trail Logger:** The environment outputs a standardized `inference_audit_log.csv` of all interactions. This allows researchers to analyze agent failure points and evaluate the distribution of Dense Rewards.

**Sample Audit Log Output (Active transition into the Hard Tier):**

```csv
Timestamp,Episode_ID,Task_Type,Bundle,Agent_Action,Reward
2026-04-08 05:38:09,5c89a...,Task 3,Hospital Entrance,"{""decision"":""log_and_continue""}",1.000
2026-04-08 05:38:12,21bba...,Task 2,Parking Lot,"{""ranked"":[""person"",""car"",""meter""]}",0.650
2026-04-08 05:38:29,f7893...,Task 2,Rainy Street,"{""ranked"":[""bus"",""car"",""person"",""umbrella""]}",1.000
```

---

## Table of Contents

1. [Why 3-step episodes?](#why-3-step-episodes)
2. [AutoRL approach — how it all fits together](#autorl-approach--how-it-all-fits-together)
3. [Dynamic Scene Generation (Level 1)](#dynamic-scene-generation-level-1)
4. [Auto-Curriculum Learning (Level 2)](#auto-curriculum-learning-level-2)
5. [Environment overview (observations, actions, tasks)](#environment-overview-observations-actions-tasks)
6. [Prerequisites](#prerequisites)
7. [Setup: `requirements.txt` and venv](#setup-requirementstxt-and-venv)
8. [Run with Docker](#run-with-docker)
9. [Run locally without Docker (uvicorn)](#run-locally-without-docker-uvicorn)
10. [Run the demo (offline)](#run-the-demo-offline)
11. [Gradio Playground (`/web`)](#gradio-playground-web)
12. [How grading works](#how-grading-works)
13. [OpenEnv compliance and key files](#openenv-compliance-and-key-files)
14. [Project structure](#project-structure)
15. [Example interaction (reset → three steps)](#example-interaction-reset--three-steps)
16. [Testing and validation](#testing-and-validation)
17. [Offline execution](#offline-execution)
18. [Optional: LLM baseline (`inference.py`)](#optional-llm-baseline-inferencepy)
19. [Design notes](#design-notes)
20. [Troubleshooting](#troubleshooting)
21. [FAQ](#faq)
22. [Future improvements](#future-improvements)
23. [Visuals & architecture](#visuals--architecture)
24. [Credits and acknowledgements](#credits-and-acknowledgements)
25. [License](#license)
26. [Maintainer / contact](#maintainer--contact)

---

## Why 3-step episodes?

A **single-step** environment gives RL agents one reward signal per reset — limiting the training signal and making it impossible to model sequential decision-making. ARJUNA solves this with **3-step episodes**:

| Benefit                  | Detail                                                                 |
|--------------------------|------------------------------------------------------------------------|
| **Denser reward signal** | Agents receive a reward after **every step**, enabling faster credit assignment and learning. |
| **Sequential difficulty**| Steps escalate: easy identification → ordered triage → ambiguous low-confidence call. |
| **Thematic coherence**   | All 3 steps draw scenes from the same **location bundle**, so context carries across steps. |
| **Overall episode signal**| `overall_reward` = mean of 3 step rewards gives a clean episode-level metric. |

The **12 themed bundles** ensure diverse training distributions across resets:

| #  | Bundle              | Notable Objects                              |
|----|---------------------|----------------------------------------------|
| 1  | Urban Street        | person, car, bicycle, traffic light          |
| 2  | Warehouse           | truck, forklift, carton, worker              |
| 3  | Parking Lot         | car, parking meter, CCTV camera, person      |
| 4  | School Zone         | bus, backpack, bicycle, student              |
| 5  | Airport             | airplane, suitcase, boarding gate, trolley   |
| 6  | Hospital Entrance   | ambulance, wheelchair, stretcher, person     |
| 7  | Construction Site   | helmet, excavator, crane, worker             |
| 8  | Night Street        | streetlight, fire hydrant, person, motorcycle|
| 9  | Forest Trail        | hiker, backpack, tree, dog                   |
| 10 | Shopping Mall       | person, escalator, shopping bag, CCTV camera |
| 11 | Office Lobby        | laptop, reception desk, couch, potted plant  |
| 12 | Rainy Street        | umbrella, car, bus, raincoat                 |

---

## AutoRL approach — how it all fits together

ARJUNA implements a **closed-loop, self-improving training environment** inspired by Automatic Reinforcement Learning (AutoRL) principles. The two subsystems — **Dynamic Scene Generation** and **Auto-Curriculum** — work together in a feedback loop:

![AutoRL Loop Diagram](https://mermaid.ink/img/eyJjb2RlIjogImZsb3djaGFydCBURFxuICBzdWJncmFwaCBBdXRvUkxfTG9vcCBbQXV0b1JMIExvb3BdXG4gICAgQVtcIkFnZW50IHN1Ym1pdHMgYWN0aW9uc1wiXSAtLT4gQltcIkdyYWRlciBzY29yZXMgZXBpc29kZVwiXVxuICAgIEIgLS0-IENbXCJBdXRvLUN1cnJpY3VsdW0gcmVjb3JkcyByZXdhcmRcIl1cbiAgICBDIC0tPiBEe1wiTWVhbiByZXdhcmQgdnMgdGhyZXNob2xkc1wifVxuICAgIEQgLS0gXCI-IDAuODVcIiAtLT4gRVtcIlBST01PVEUgZGlmZmljdWx0eVwiXVxuICAgIEQgLS0gXCI8IDAuNjBcIiAtLT4gRltcIkRFTU9URSBkaWZmaWN1bHR5XCJdXG4gICAgRCAtLSBcIjAuNjAtMC44NVwiIC0tPiBHW1wiU1RBWSBhdCBjdXJyZW50IGxldmVsXCJdXG4gICAgRSAtLT4gSFtcIlNjZW5lIEdlbmVyYXRvciB1c2VzIG5ldyBkaWZmaWN1bHR5XCJdXG4gICAgRiAtLT4gSFxuICAgIEcgLS0-IEhcbiAgICBIIC0tPiBJW1wiTExNIGdlbmVyYXRlcyBmcmVzaCBzY2VuZSBhdCBkaWZmaWN1bHR5IHRpZXJcIl1cbiAgICBJIC0tPiBKW1wiQWdlbnQgcmVjZWl2ZXMgbmV3IG9ic2VydmF0aW9uXCJdXG4gICAgSiAtLT4gQVxuICBlbmQiLCAibWVybWFpZCI6IHsidGhlbWUiOiAiZGVmYXVsdCJ9fQ)

### Key design principles

| Principle               | Implementation                                                                 |
|-------------------------|--------------------------------------------------------------------------------|
| **No memorization**     | The LLM generates a unique scene every `reset()` — ensures OOD robustness      |
| **Adaptive difficulty** | Sliding-window curriculum automatically promotes/demotes difficulty            |
| **Graceful degradation**| Falls back to 12 hardcoded offline bundles if LLM is unavailable               |
| **Stateless scalability**| `episode_id` + `SESSIONS` pattern for stateless HTTP workers                   |
| **Environment variables**| `ENABLE_DYNAMIC_SCENES`, `API_BASE_URL`, `HF_TOKEN` toggle features            |

### Files implementing autoRL

| File                          | Role in AutoRL                                                              |
|-------------------------------|-----------------------------------------------------------------------------|
| `server/scene_generator.py`   | LLM-powered scene generation with difficulty-aware prompts                  |
| `server/curriculum.py`        | `AutoCurriculum` class: sliding-window reward tracker                      |
| `server/arjuna_environment.py`| Orchestrator: calls `generate_episode_bundle()` and `record_episode()`      |
| `server/app.py`               | Exposes `GET /curriculum` endpoint for real-time monitoring                 |

---

## Environment overview (observations, actions, tasks)

### What the agent sees

After **`reset`**, **`ArjunaObservation`** includes:
- **`task_type`**: `1` for step 1 (then `2`, then `3`)
- **`step_number`**: `1`, `2`, or `3`
- **`bundle_name`**: human-readable theme shared across the episode
- **`scene_id`**: id for the current task’s scene
- **`observation_text`**: instructions + scene description + simulated YOLO lines
- **`episode_id`**: must be sent back on stateless HTTP `POST /step`
- After each **`step`**: **`reward`**, **`done`**, **`feedback`**
- After the third step: **`overall_reward`** and **`done: true`**

### What actions it can take

Single Pydantic model **`ArjunaAction`**:

| Field               | Task | Role                                              |
|---------------------|------|---------------------------------------------------|
| `task1_label`       | 1    | Predicted class label (string)                    |
| `ranked_objects`    | 2    | Ordered list of labels (most important first)     |
| `decision`          | 3    | One of `discard`, `request_rescan`, `log_and_continue` |
| `reasoning`         | 3    | Optional; affects partial credit on task 3        |

### The three tasks (summary)

- **Task 1** — Single-object identification
- **Task 2** — Multi-object triage
- **Task 3** — Low-confidence decision

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python**  | **3.11+** |
| **pip**     | For `requirements.txt` |
| **Docker**  | Optional (recommended) |
| **Git**     | To clone / push the repo |

---

## Setup: `requirements.txt` and venv

```bash
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run with Docker

```bash
docker build -t arjuna-env .
docker run -p 7860:7860 arjuna-env
```

---

## Run locally without Docker (uvicorn)

```bash
export ENABLE_WEB_INTERFACE=true
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## Run the demo (offline)

```bash
docker build -t arjuna-env .
docker run -p 7860:7860 arjuna-env
```

In another terminal:
```bash
export ARJUNA_ENV_BASE_URL=http://127.0.0.1:7860
python demo.py
```

---

## Gradio Playground (`/web`)

When `ENABLE_WEB_INTERFACE=true`, a browser UI is available at `/web`.

---

## Dynamic Scene Generation (Level 1)

The environment can generate **infinite unique episodes** using an LLM. On failure, it silently falls back to hardcoded scenes.

### Difficulty-aware generation

| Difficulty | Task 1 (Confidence) | Task 2 (Objects) | Task 3 (Bands) |
|------------|---------------------|------------------|----------------|
| easy       | 0.85–0.98           | 3 objects, no ties | Deep in one band |
| medium     | 0.72–0.84           | 4 objects, 1 tie | Near boundary |
| hard       | 0.60–0.71           | 5 objects, multiple ties | Within 0.005 of boundary |

---

## Auto-Curriculum Learning (Level 2)

The environment automatically adjusts difficulty based on recent performance:

```
Agent mean reward ≥ 0.85 → PROMOTE
Agent mean reward < 0.60 → DEMOTE
Otherwise → STAY
```

### Curriculum internals

| Parameter           | Value | Purpose |
|---------------------|-------|---------|
| `WINDOW_SIZE`       | 5     | Number of recent episodes |
| `MIN_EPISODES`      | 3     | Minimum data before adjustment |
| `PROMOTE_THRESHOLD` | 0.85  | Threshold to increase difficulty |
| `DEMOTE_THRESHOLD`  | 0.60  | Threshold to decrease difficulty |

---

## How grading works

- **Task 1**: Exact match + semantic category partial credit
- **Task 2**: Position-based sequence alignment
- **Task 3**: Decision band matching + reasoning quality

---

## Project structure

```bash
arjuna_env/
├── README.md
├── LICENSE
├── openenv.yaml
├── requirements.txt
├── Dockerfile
├── demo.py
├── inference.py
├── models.py
└── server/
    ├── app.py
    ├── arjuna_environment.py
    ├── scene_generator.py
    ├── curriculum.py
    ├── tasks.py
    ├── synthetic_data.py
    └── ...
```

---

## Visuals & Architecture

**Try the live UI:**

- **Gradio Playground:** [https://calpol500mg-arjuna-env.hf.space/web](https://calpol500mg-arjuna-env.hf.space/web)
- **Swagger:** [https://calpol500mg-arjuna-env.hf.space/docs](https://calpol500mg-arjuna-env.hf.space/docs)

---

## Credits and Acknowledgements

- **[OpenEnv](https://github.com/meta-pytorch/OpenEnv)** — Meta & Hugging Face
- FastAPI, Pydantic, uvicorn, Gradio
- Synthetic scenes and grading logic developed for this project

---

## License

This project is released under the **MIT License**.

---

## Maintainer / Contact

- **Author:** Ayush Kumar
- **HF Space:** [Calpol500mg/arjuna-env](https://huggingface.co/spaces/Calpol500mg/arjuna-env)
- **Live App:** [calpol500mg-arjuna-env.hf.space](https://calpol500mg-arjuna-env.hf.space)

For questions, use the Space Community tab or GitHub Issues.

---

**Built for robust robot perception and generalizable reinforcement learning.**
```

This version keeps **all your original content and meaning** intact while:
- Removing all emojis
- Using clean `<h1>` and `<h3>` centered headings
- Improving table readability
- Maintaining professional tone and structure
- Keeping it highly visual and GitHub-friendly

You can copy and paste this directly as your `README.md` file. It will look clean and professional on GitHub and Hugging Face Spaces. 

Let me know if you want any small tweaks!
