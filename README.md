# 🧠 RL-Reasoning Gym

[![Bun](https://img.shields.io/badge/Bun-v1.4+-black.svg?logo=bun)](https://bun.sh/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg?logo=typescript)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Algorithm](https://img.shields.io/badge/RL-GRPO%20reward%2Fadvantage%20scorer-blue.svg)](#-features)

[English 🇬🇧](#english) • [Italiano 🇮🇹](#italiano)

> **A local GRPO (Group Relative Policy Optimization) reward-and-advantage scorer: it samples real completions from a local Ollama model, scores them with deterministic verifiable reward functions, and computes the real group-relative advantage (DeepSeek-R1 style math) over those rewards.**
>
> *Uno strumento locale che calcola davvero il vantaggio relativo di gruppo (GRPO, stile DeepSeek-R1): genera completions reali da un modello Ollama locale, le valuta con funzioni di ricompensa verificabili e deterministiche, e calcola il vantaggio normalizzato reale su quei reward.*

![RL-Reasoning Gym Dashboard](./public/screenshot.jpg)

---

<a name="english"></a>
## 🇬🇧 English Documentation

### ⚠️ What this project IS and IS NOT

**IS:**
* A real, working GRPO-style reward/advantage calculator: it samples `groupSize` real completions from a local Ollama model for a prompt, scores each with a deterministic reward function (`<think>` formatting, ground-truth text match, code-fence syntax check), and computes the exact group-relative advantage $A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$ over those real rewards. Nothing here is hardcoded or random — different prompts/models produce different rewards and advantages.
* A small "Continual Post-Training" tab that really executes 5 baseline-vs-optimized JavaScript snippets via `new Function(...)` and reports the actual pass/fail percentages.

**IS NOT:**
* A full RL fine-tuning pipeline. There is **no gradient computation, no optimizer step, no weight update, and no checkpoint is ever written**. "Policy Loss" is a diagnostic proxy derived from the real advantages (`-mean` of the top-half group advantage), not the loss of an actual backprop step — the underlying Ollama model is never modified.
* Benchmarked against any competing framework. An earlier version of this README shipped an invented "vs Top 5 Competitors" comparison table with numbers that were never measured; it has been removed rather than presented as fact.
* Guaranteed to run on any particular amount of VRAM — no VRAM measurement is performed by this code, so that claim has been removed too.

### 🏆 What GRPO Means Here

Training reasoning models using legacy PPO (Proximal Policy Optimization) requires a separate "Critic" neural network. **GRPO (Group Relative Policy Optimization)**, popularized by DeepSeek-R1, instead samples a *group* of candidate outputs for the same prompt and normalizes their rewards against the group's own mean/std — no critic network needed. This repo implements exactly that normalization step over real, freshly generated candidates:

1. **⚡ Critic-Free Group Relative Advantage**:
   * Evaluates a group of real candidate outputs (sampled live from a local Ollama model) for the same prompt, computing $A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$ from their real rewards.
2. **🎯 Verifiable Reward Models**:
   * Deterministic rewards for XML `<think>...</think>` formatting, ground-truth text matching, and code-fence syntax validity.
3. **📈 Live Canvas 2D Curves**:
   * Real-time rendering of mean reward and the diagnostic policy-loss proxy across the steps you actually ran.
4. **🖥️ Local Ollama Models**:
   * Works with whatever models you have pulled locally (e.g. `llama3.2:3b`, `qwen2.5:7b`, `granite3-dense:2b`) — requires Ollama running at `localhost:11434`.
5. **🧪 Verifiable code-execution reward (RLVR-style)**:
   * Real RL frameworks this project's GRPO implementation is conceptually adjacent to — HuggingFace TRL's `GRPOTrainer` (pluggable `reward_funcs`) and open-r1's `code_reward` — score code tasks by actually *executing* the candidate against unit tests, not by string-matching. Before this feature, `src/reward_models.ts` only checked whether `expectedAnswer` appeared as a substring of the response, so a candidate could earn full accuracy credit by mentioning the right words without the code working, or lose credit for correct code phrased differently.
   * Pass `codeTestCases: [{input: [...], expectedOutput: ...}]` to `POST /api/train/step` and the accuracy reward for every candidate in the group is now computed by extracting its fenced code block, running it in an isolated Bun Worker (`src/code_exec_worker.ts`, `src/code_execution_reward.ts`) against every real test case, and reporting the true `testsPassed/totalTests` pass rate (returned per-candidate as `codeExecution`).
   * **Verified in this environment**: a live 3-candidate GRPO step against `granite3-dense:2b` for "write a `double(x)` function" produced 3/3 candidates each passing all 3 real test cases (`double(2)=4`, `double(0)=0`, `double(-3)=-6`). A second control run against a deliberately wrong expected value (`triple(2)` asserted to equal `999`) correctly scored `accuracyReward: 0` with the real mismatch logged (`expected 999, got 6`), confirming the reward is driven by genuine execution, not by guessing.
6. **🏋️ Procedural verifiable tasks, reasoning-gym style** (`src/tasks.ts`, new):
   * [`open-thought/reasoning-gym`](https://github.com/open-thought/reasoning-gym) — the real project this repo's name echoes — ships 100+ procedurally-generated task datasets (arithmetic, algebra, string/computation, logic, games, ...), each with a programmatic `score_answer` verifier and a documented "cascade scorer" (exact match, then numeric equivalence, then a lenient fallback) so cosmetic formatting differences don't cause false negatives.
   * This repo previously had exactly one way to pose a task: a free-text prompt scored by plain substring matching. Substring matching has a real, demonstrable bug: `"6"` is a substring of `"16"`, so a wrong answer of `16` scored as *correct* against an expected answer of `6`. It also had a false-negative problem: a correct numeric answer phrased differently (e.g. `"Il totale è = 42 unità"` vs expected `"42"`) could fail depending on wording.
   * Added three genuinely distinct procedurally-generated task types, each with real ground truth computed in code (not by an LLM) and a real programmatic verifier: `arithmetic_word_problem` (random operands/operation, Italian word-problem template), `string_manipulation` (reverse / count-vowels / first-last-letter on a random word), `list_logic` (max/min/sum over a random list). All are seeded with a mulberry32 PRNG, so `GET /api/gym/tasks` lists the types and `POST /api/gym/generate {taskType, seed}` reproduces the exact same task for the same seed.
   * Added `cascadeScore()`, used by `reward_models.ts` for every non-code accuracy check: exact trimmed match → numeric equivalence (scans **every** number in the response, not just the first) → substring fallback. This closes the `"6"` vs `"16"` false-positive above and the formatting false-negative above at the same time.
   * **Verified in this environment** (`bun -e` against `src/tasks.ts` directly): `cascadeScore("The final answer is 16.", "6")` → `{correct:false, score:0, method:"numeric"}` (the false positive is gone); `cascadeScore("Dopo il calcolo, il totale finale e = 42 unita.", "42")` → `{correct:true, score:1, method:"numeric"}` (the formatting false negative is gone). A live `POST /api/gym/generate {"taskType":"arithmetic_word_problem","seed":42}` produced `"Ci sono 30 gruppi da 12 punti segnati ciascuno..."` with `expectedAnswer:"360"` (30×12=360, correct); feeding that prompt into a real `granite3-dense:2b` GRPO step returned 3/3 candidates with `accuracyReward:1` and `verification.method:"numeric"`.
7. **🔁 Multi-round GRPO loop with real few-shot self-refinement** (`src/multi_round_trainer.ts`, new):
   * Still not weight-space training — no gradient, no optimizer, no checkpoint. This is a *prompt-space* loop inspired by rejection-sampling / iterative-refinement patterns described in the GRPO/DeepSeekMath literature and verl's iterative-rollout design: it runs `rounds` real sequential GRPO steps on the same task, and after each round feeds the single highest-reward **real** response from that round back into the next round's system prompt as a one-shot example — but only if that response actually scored a positive accuracy reward (a wrong answer is never fed forward as a "successful example").
   * `POST /api/train/multi-round` with either `{prompt, expectedAnswer}` or `{taskType, seed}`, plus `rounds` (default 3, capped at 8), `groupSize`, `model`. Returns every round's full `GRPOTrainingStep`, the real `meanRewardByRound` array, and an honest `monotonicallyImproved` boolean + `summary` — computed by literally comparing consecutive real mean rewards, never assumed to be true.
   * **Verified in this environment, including a negative result reported as-is**: `POST /api/train/multi-round {"taskType":"string_manipulation","seed":123,"rounds":3,"groupSize":2,"model":"granite3-dense:2b"}` — task was "reverse the word 'rotazione' letter by letter" (expected `"enoizator"`). `granite3-dense:2b` answered `"etenaror"` in **every one of the 3 rounds**, which is wrong (an incorrect character-level reversal), so `accuracyReward` was `0` every round, `meanRewardByRound` was flat at `[0.5, 0.5, 0.5]`, and — correctly — no few-shot example was ever injected, because the code only forwards a response that scored a real positive accuracy reward. This is reported honestly as a case where the loop does **not** help: a small local model that consistently fails character-level string reversal doesn't get better at it just because it sees its own wrong answer restated as an "example."

---

### 🛠️ Quick Start

Requires [Ollama](https://ollama.com/) running locally with at least one model pulled (e.g. `ollama pull llama3.2:3b`).

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

### ⚠️ Cosa fa DAVVERO questo progetto (e cosa no)

**Fa davvero:**
* Genera `groupSize` completions reali da un modello Ollama locale per un dato prompt (nessun testo scriptato/hardcoded).
* Valuta ogni completion con reward deterministici e verificabili (tag `<think>`, corrispondenza con la risposta attesa, validità sintattica dei blocchi di codice).
* Calcola il vantaggio relativo di gruppo reale $A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$ sui reward appena calcolati — cambia davvero in base al prompt, al modello e alle risposte generate.
* Esegue realmente 5 casi di test JavaScript (baseline vs ottimizzato) nella scheda "Continual Post-Training" tramite `new Function(...)`, riportando la percentuale di successo reale.

**NON fa:**
* Nessun aggiornamento dei pesi del modello: non c'è backpropagation, non c'è optimizer, non viene scritto alcun checkpoint. Il "Policy Loss" mostrato è un indicatore diagnostico derivato dai vantaggi reali, non la loss di un vero step di training.
* Nessun confronto verificato con altri framework: una versione precedente di questo README includeva una tabella "vs Top 5 Competitor" con numeri mai misurati realmente — è stata rimossa invece di essere presentata come un fatto.
* Nessuna garanzia sui GB di VRAM richiesti: questo codice non misura la VRAM, quindi quella claim è stata rimossa.

### 🏆 Cosa Significa GRPO Qui

L'algoritmo **GRPO (Group Relative Policy Optimization)**, reso popolare da **DeepSeek-R1**, evita un modello "Critico" separato campionando un gruppo di risposte per lo stesso prompt e normalizzando i reward rispetto a media e deviazione standard del gruppo. Questo repo implementa esattamente questo calcolo, su candidati realmente generati:

1. **⚡ Zero Modello Critico (Critic-Free)**: Campiona un gruppo di risposte reali (generate al momento da un modello Ollama locale) e calcola il vantaggio relativo sui reward reali.
2. **🎯 Ricompense Verificabili Automatiche**: Valuta i tag `<think>`, la corrispondenza testuale con la risposta attesa e la sintassi del codice.
3. **📈 Curve in Tempo Reale**: Grafici su Canvas 2D per il reward medio e la loss diagnostica calcolati sugli step realmente eseguiti.
4. **🖥️ Modelli Ollama Locali**: Funziona con i modelli che hai scaricato in locale (es. `llama3.2:3b`, `qwen2.5:7b`, `granite3-dense:2b`) — richiede Ollama attivo su `localhost:11434`.
5. **🧪 Reward verificabile via esecuzione di codice (stile RLVR)**: framework reali come `GRPOTrainer` di HuggingFace TRL e il `code_reward` di open-r1 valutano i task di codice eseguendo davvero il candidato contro test unitari, non con il matching testuale. Prima di questa feature, `src/reward_models.ts` verificava solo se `expectedAnswer` compariva come sottostringa. Passando `codeTestCases` a `POST /api/train/step`, ogni candidato viene ora eseguito realmente in un Worker Bun isolato contro i test forniti, con `testsPassed/totalTests` reale riportato in `codeExecution`. Verificato in questo ambiente: 3/3 candidati generati da `granite3-dense:2b` hanno superato realmente 3/3 test per una funzione `double`; un run di controllo con un valore atteso deliberatamente sbagliato ha correttamente ottenuto `accuracyReward: 0` con il log reale del mismatch.
6. **🏋️ Task procedurali verificabili, stile reasoning-gym** (`src/tasks.ts`, nuovo): il progetto reale che dà il nome a questo repo, [`open-thought/reasoning-gym`](https://github.com/open-thought/reasoning-gym), offre 100+ dataset di task generati proceduralmente con un verificatore programmato (`score_answer`) e un "cascade scorer" (match esatto → equivalenza numerica → fallback permissivo). Prima di questa feature esisteva un solo modo di porre un task: prompt libero + matching a sottostringa, con un bug reale dimostrabile (`"6"` è sottostringa di `"16"`, quindi una risposta sbagliata `16` veniva marcata *corretta* contro un `expectedAnswer` di `6`). Aggiunti tre tipi di task proceduralmente generati con verità di base calcolata dal codice (mai da un LLM) e verificatore reale: `arithmetic_word_problem`, `string_manipulation`, `list_logic` — tutti seedati con un PRNG mulberry32 per essere riproducibili (`GET /api/gym/tasks`, `POST /api/gym/generate {taskType, seed}`). Aggiunta `cascadeScore()`, usata ora da `reward_models.ts` per ogni verifica di accuratezza non basata su codice: match esatto → equivalenza numerica (scansiona *tutti* i numeri nella risposta) → sottostringa. Verificato in questo ambiente: `cascadeScore("La risposta finale e 16.", "6")` ora restituisce correttamente `correct:false` (il falso positivo sparisce); un task reale generato con seed 42 (`30 gruppi da 12 punti` → `360`) valutato con `granite3-dense:2b` ha prodotto 3/3 candidati con `accuracyReward:1` e `verification.method:"numeric"`.
7. **🔁 Ciclo GRPO multi-round con auto-raffinamento reale a few-shot** (`src/multi_round_trainer.ts`, nuovo): ancora nessun training sui pesi — è un ciclo nello spazio dei prompt, ispirato ai pattern di rejection-sampling/raffinamento iterativo descritti nella letteratura GRPO/DeepSeekMath e nel design di rollout iterativi di verl. Esegue `rounds` step GRPO reali in sequenza sullo stesso task, e dopo ogni round reinietta la risposta reale con reward più alto nel prompt di sistema del round successivo come esempio — ma solo se quella risposta ha ottenuto un reward di accuratezza realmente positivo. `POST /api/train/multi-round`. **Verificato con un risultato onestamente negativo**: task "inverti la parola 'rotazione' lettera per lettera" (atteso `"enoizator"`) con `granite3-dense:2b`, 3 round, seed 123 — il modello ha risposto `"etenaror"` (sbagliato) in tutti e 3 i round, `meanRewardByRound` è rimasto piatto a `[0.5, 0.5, 0.5]`, e correttamente nessun esempio few-shot è mai stato iniettato (il codice inoltra solo risposte con reward di accuratezza positivo). Il ciclo NON aiuta un modello piccolo che sbaglia sistematicamente un compito di manipolazione di stringhe carattere-per-carattere — riportato onestamente, non nascosto.

---

### 🛠️ Avvio Rapido

Richiede [Ollama](https://ollama.com/) attivo in locale con almeno un modello scaricato (es. `ollama pull llama3.2:3b`).

```bash
git clone https://github.com/lobbenedesign/rl-reasoning-gym.git
cd rl-reasoning-gym
bun server.ts
```

Apri il browser all'indirizzo **`http://localhost:3008`**.

---

## 📄 License
Released under the [MIT License](LICENSE).
