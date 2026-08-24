#!/usr/bin/env bun
/**
 * 🧠 RL-REASONING GYM SERVER (v1.0.0)
 * Local Group Relative Policy Optimization (GRPO) Reinforcement Learning Studio
 */

import { GRPOTrainer } from "./src/grpo_trainer";
import { RLCompetitorBenchmark } from "./src/competitor_benchmark";
import { join } from "path";
import { existsSync } from "fs";

const PORT = Number(process.env.PORT) || 3008;

const trainer = new GRPOTrainer();
const benchmark = new RLCompetitorBenchmark();

console.log(`\n======================================================`);
console.log(`🧠 RL-REASONING GYM running on http://localhost:${PORT}`);
console.log(`⚡ GRPO Critic-Free Policy Optimization: Ready`);
console.log(`🎯 Multi-Task Verifiable Reward Functions: Online`);
console.log(`📊 5-Competitor Benchmark Matrix: Active`);
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
      return new Response(JSON.stringify({
        status: "online",
        version: "1.0.0-rlgym",
        algorithm: "GRPO (Group Relative Policy Optimization)",
        criticFree: true,
        supportedHardware: "Apple Silicon (MPS) & NVIDIA CUDA"
      }), { headers });
    }

    // 2. Run GRPO Training Step
    if (url.pathname === "/api/train/step" && req.method === "POST") {
      try {
        let body: any = {};
        try { body = await req.json(); } catch {}
        const prompt = body.prompt || "Prove that sum(1..n) = n*(n+1)/2.";
        const expected = body.expectedAnswer || "n*(n+1)/2";
        const groupSize = Number(body.groupSize) || 4;

        const stepResult = await trainer.runStep(prompt, expected, groupSize);
        return new Response(JSON.stringify(stepResult), { headers });
      } catch (e: any) {
        return new Response(JSON.stringify({ error: e.message }), { status: 500, headers });
      }
    }

    // 3. Training Telemetry History
    if (url.pathname === "/api/train/history" && req.method === "GET") {
      return new Response(JSON.stringify(trainer.getHistory()), { headers });
    }

    // 4. 5-Competitor Matrix
    if (url.pathname === "/api/competitors" && req.method === "GET") {
      return new Response(JSON.stringify(benchmark.getComparison()), { headers });
    }

    return new Response("Not Found", { status: 404, headers });
  }
});
