---
title: Kavach API
emoji: 🛡️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: YOLOv8 detection backend for the Kavach console
---

# Kavach detection backend

FastAPI + YOLOv8n (CPU) backend for **Kavach**, a border and perimeter surveillance console. It runs detection, tracking, zone and tripwire analytics and number plate reading on a demo clip and streams frames and telemetry over a WebSocket.

- Operator console: https://kavach-surveillance.vercel.app
- Source: https://github.com/ritvik302005/KAVACH
- Health check: `/health`

This Space is built from `deploy/huggingface/` in the source repository by `push_space.py`.
