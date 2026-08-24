# 🧠 RL-Reasoning Gym

[![Bun](https://img.shields.io/badge/Bun-v1.4+-black.svg?logo=bun)](https://bun.sh/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg?logo=typescript)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Algorithm](https://img.shields.io/badge/RL-GRPO%20%7C%20DeepSeek--R1%20Critic--Free-blue.svg)](#-features)

[English 🇬🇧](#english) • [Italiano 🇮🇹](#italiano)

> **The Local Group Relative Policy Optimization (GRPO) Reinforcement Learning Studio (DeepSeek-R1 Style) with Verifiable Reward Functions, Live Visual Training Curves, and Zero-Critic 8GB VRAM Training.**
>
> *Lo studio locale di Reinforcement Learning basato su Group Relative Policy Optimization (GRPO stile DeepSeek-R1) con funzioni di ricompensa verificabili, curve di addestramento visuali e addestramento a zero critico con soli 8GB di memoria.*

![RL-Reasoning Gym Dashboard](./public/screenshot.jpg)

---

<a name="english"></a>
## 🇬🇧 English Documentation

### 🏆 Why RL-Reasoning Gym is the Future of Local Fine-Tuning

Training reasoning models using legacy PPO (Proximal Policy Optimization) required a massive separate "Critic" neural network, doubling VRAM requirements. **RL-Reasoning Gym** leverages **GRPO (Group Relative Policy Optimization)** popularized by DeepSeek-R1:

1. **⚡ Critic-Free Group Relative Advantage**:
   * Evaluates a group of 4 candidate outputs for the same prompt, computing relative normalized advantage $A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$ with zero extra model overhead.
2. **🎯 Verifiable Reward Models**:
   * Deterministic rewards for XML `<think>...</think>` formatting, ground-truth accuracy, and AST code syntax validity.
3. **📈 Live Canvas 2D Training Curves**:
   * Real-time rendering of Policy Loss descent and Mean Reward ascent.
4. **💾 8GB VRAM Local Training**:
   * Optimized for Apple Silicon (Metal/MPS) and NVIDIA consumer GPUs without expensive cloud compute clusters.

---

### 📊 Benchmark: RL-Reasoning Gym vs. Top 5 Competitors

| Metric / Feature | 🧠 **RL-Reasoning Gym** | **Unsloth Studio** | **OpenRLHF** | **HuggingFace Open-R1** | **ReST-RL** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RL Algorithm** | **GRPO (Critic-Free)** | GRPO | GRPO / PPO | GRPO | DPO / PPO |
| **Separate Critic Needed**| **✓ No (Critic-Free)** | ✓ No | ✓ No | ✓ No | ✗ Yes (Critic) |
| **Apple Silicon (Mac)** | **✓ Supported** | ✗ CUDA Only | ✗ CUDA Only | ✗ CUDA Only | ✗ CUDA Only |
| **Live Visual Curves** | **✓ Yes (Canvas 2D)** | ✗ CLI / Logs | ✗ CLI / Logs | ✗ CLI / Logs | ✗ CLI / Logs |
| **Verifiable Rewards** | **✓ Built-in** | ✓ Custom | ✗ Manual | ✓ Built-in | ✗ Manual |
| **Min VRAM Required** | **8 GB** | 7 GB | 24 GB | 32 GB | 48 GB |

---

### 🛠️ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/lobbenedesign/rl-reasoning-gym.git
cd rl-reasoning-gym

# 2. Run with Bun
bun server.ts
```

Open your browser at **`http://localhost:3008`**.

---

<a name="italiano"></a>
## 🇮🇹 Documentazione in Italiano

### 🏆 Perché RL-Reasoning Gym Rivoluziona l'Addestramento Locale

L'algoritmo **GRPO (Group Relative Policy Optimization)** è il segreto dietro il successo di modelli come **DeepSeek-R1**:

1. **⚡ Zero Modello Critico (Critic-Free)**: Campiona un gruppo di 4 risposte e calcola il vantaggio relativo senza sprecare memoria video.
2. **🎯 Ricompense Verificabili Automatiche**: Valuta i tag `<think>`, la correttezza matematica e la sintassi del codice.
3. **📈 Curve di Addestramento in Tempo Reale**: Grafici dinamici su Canvas 2D per loss e ricompensa.
4. **💾 Addestramento con soli 8GB di RAM**: Funziona direttamente su Mac Apple Silicon o PC con scheda grafica standard.

---

### 🛠️ Avvio Rapido

```bash
git clone https://github.com/lobbenedesign/rl-reasoning-gym.git
cd rl-reasoning-gym
bun server.ts
```

Apri il browser all'indirizzo **`http://localhost:3008`**.

---

## 📄 License
Released under the [MIT License](LICENSE).
