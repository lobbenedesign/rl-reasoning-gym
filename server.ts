#!/usr/bin/env bun
/**
 * 🧠 RL-REASONING GYM SERVER (v1.0.0)
 * Local Group Relative Policy Optimization (GRPO) Reinforcement Learning Studio
 */

import { GRPOTrainer } from "./src/grpo_trainer";
import { ContinualPostTrainingEngine } from "./src/continual_post_training";
import { ollamaIsReachable } from "./src/ollama_client";
import { generateTask, TASK_TYPES, type TaskType } from "./src/tasks";
import { MultiRoundTrainer } from "./src/multi_round_trainer";
import { join } from "path";
import { existsSync } from "fs";

const PORT = Number(process.env.PORT) || 3008;

const trainer = new GRPOTrainer();
const postTrainer = new ContinualPostTrainingEngine();
const multiRoundTrainer = new MultiRoundTrainer();

console.log(`\n======================================================`);
console.log(`🧠 RL-REASONING GYM running on http://localhost:${PORT}`);
console.log(`⚡ GRPO group-relative reward/advantage scorer: Ready`);
console.log(`🎯 Verifiable reward functions (real, deterministic): Online`);
console.log(`⚙️  Rollouts are sampled from a real local Ollama model (no scripted text)`);
ollamaIsReachable().then(ok => {
  console.log(ok
    ? `✅ Ollama reachable at http://localhost:11434 — real generations enabled`
    : `⚠️  Ollama NOT reachable at http://localhost:11434 — /api/train/step will fail honestly until it is started`);
});
console.log(`======================================================\n`);

const server = Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);

    const headers = {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization"
    };

    if (req.method === "OPTIONS") return new Response(null, { headers });

    // Serve Static UI Assets
    if (url.pathname === "/" || url.pathname === "/index.html") {
      const p = join(__dirname, "public", "index.html");
      return new Response(Bun.file(p), { headers: { "Content-Type": "text/html" } });
    }
    if (url.pathname === "/app.js") {
      const p = join(__dirname, "public", "app.js");
      return new Response(Bun.file(p), { headers: { "Content-Type": "application/javascript" } });
    }
    if (url.pathname === "/style.css") {
      const p = join(__dirname, "public", "style.css");
      return new Response(Bun.file(p), { headers: { "Content-Type": "text/css" } });
    }
    if (url.pathname.startsWith("/public/")) {
      const p = join(__dirname, url.pathname);
      if (existsSync(p)) return new Response(Bun.file(p));
    }

    // 1. Status
    if (url.pathname === "/api/status" && req.method === "GET") {
      const ollamaUp = await ollamaIsReachable();
      return new Response(JSON.stringify({
        status: "online",
        version: "1.0.0-rlgym",
        algorithm: "GRPO (Group Relative Policy Optimization) — real reward/advantage math, no gradient update",
        criticFree: true,
        ollamaReachable: ollamaUp
      }), { headers });
    }

    // 2. Run a real GRPO rollout scoring step (samples from Ollama, no fabricated text)
    if (url.pathname === "/api/train/step" && req.method === "POST") {
      try {
        let body: any = {};
        try { body = await req.json(); } catch {}
        const prompt = body.prompt || "Prove that sum(1..n) = n*(n+1)/2.";
        const expected = body.expectedAnswer || "n*(n+1)/2";
        const groupSize = Number(body.groupSize) || 4;
        const model = body.model || "llama3.2:3b";
        // RLVR path: real unit-test cases {input:[], expectedOutput} to execute
        // each candidate's code against, instead of substring-matching text.
        const codeTestCases = Array.isArray(body.codeTestCases) ? body.codeTestCases : undefined;

        const stepResult = await trainer.runStep(prompt, expected, groupSize, model, codeTestCases);
        return new Response(JSON.stringify(stepResult), { headers });
      } catch (e: any) {
        // No fake "success" fallback: a failed rollout is reported as an error.
        return new Response(JSON.stringify({ error: e.message }), { status: 502, headers });
      }
    }

    // 3. Scaled Continual Post-Training Epoch (GLM-5.3 Style)
    if (url.pathname === "/api/train/post-training" && req.method === "POST") {
      try {
        let body: any = {};
        try { body = await req.json(); } catch {}
        const base = body.baseCheckpoint || "GLM-5.2-Base / Qwen-2.5-Coder-7B";
        const domain = body.domain || "Code & SWE";

        const epoch = postTrainer.runPostTrainingEpoch(base, domain);
        return new Response(JSON.stringify(epoch), { headers });
      } catch (e: any) {
        return new Response(JSON.stringify({ error: e.message }), { status: 500, headers });
      }
    }

    // 4. Training Telemetry History
    if (url.pathname === "/api/train/history" && req.method === "GET") {
      return new Response(JSON.stringify({
        grpo: trainer.getHistory(),
        postTraining: postTrainer.getEpochs()
      }), { headers });
    }

    // 5. Reasoning-gym-style procedural task types (real generators + real verifiers, no LLM grading)
    if (url.pathname === "/api/gym/tasks" && req.method === "GET") {
      return new Response(JSON.stringify({ taskTypes: TASK_TYPES }), { headers });
    }

    // 6. Generate one procedural task instance (real ground truth computed here, not by an LLM)
    if (url.pathname === "/api/gym/generate" && req.method === "POST") {
      try {
        let body: any = {};
        try { body = await req.json(); } catch {}
        const taskType = (body.taskType || "arithmetic_word_problem") as TaskType;
        const seed = body.seed !== undefined ? Number(body.seed) : undefined;
        const task = generateTask(taskType, seed);
        // Don't ship the verify() closure over the wire — return the data the
        // client needs (prompt + expectedAnswer); verification itself always
        // runs server-side against real Ollama output in /api/train/step.
        return new Response(JSON.stringify({
          taskType: task.taskType,
          seed: task.seed,
          prompt: task.prompt,
          expectedAnswer: task.expectedAnswer
        }), { headers });
      } catch (e: any) {
        return new Response(JSON.stringify({ error: e.message }), { status: 400, headers });
      }
    }

    // 7. Multi-round GRPO loop: N real sequential rounds on the same task,
    //    with the real best response from round i fed back as a few-shot
    //    example for round i+1. Reports the honest per-round reward curve —
    //    including a run where reward does NOT improve every round.
    if (url.pathname === "/api/train/multi-round" && req.method === "POST") {
      try {
        let body: any = {};
        try { body = await req.json(); } catch {}
        let prompt = body.prompt;
        let expected = body.expectedAnswer;
        const taskType = body.taskType as TaskType | undefined;
        if (taskType) {
          const task = generateTask(taskType, body.seed !== undefined ? Number(body.seed) : undefined);
          prompt = task.prompt;
          expected = task.expectedAnswer;
        }
        if (!prompt || !expected) {
          return new Response(JSON.stringify({ error: "Provide either {prompt, expectedAnswer} or {taskType}" }), { status: 400, headers });
        }
        const rounds = Number(body.rounds) || 3;
        const groupSize = Number(body.groupSize) || 3;
        const model = body.model || "llama3.2:3b";
        const codeTestCases = Array.isArray(body.codeTestCases) ? body.codeTestCases : undefined;

        const result = await multiRoundTrainer.runMultiRound(prompt, expected, rounds, groupSize, model, codeTestCases);
        return new Response(JSON.stringify(result), { headers });
      } catch (e: any) {
        return new Response(JSON.stringify({ error: e.message }), { status: 502, headers });
      }
    }

    return new Response("Not Found", { status: 404, headers });
  }
});
