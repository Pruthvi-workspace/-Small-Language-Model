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
