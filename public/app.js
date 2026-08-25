/**
 * 🧠 RL-REASONING GYM CLIENT SCRIPT
 * Handles Canvas 2D Loss/Reward Curves, GRPO Optimization Steps (real Ollama
 * rollouts + real reward/advantage math), and the Continual Post-Training
 * executable test suite.
 */

let trainingHistory = [];

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupGRPOActions();
  setupPostTrainingActions();
  checkOllamaStatus();
});

function setupTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const targetId = `tab-${tab.getAttribute("data-tab")}`;
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");
    });
  });
}

async function checkOllamaStatus() {
  const chip = document.getElementById("chip-ollama-status");
  if (!chip) return;
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    chip.textContent = data.ollamaReachable
      ? "🟢 Ollama reachable (real generations)"
      : "🔴 Ollama NOT reachable — start it first";
    chip.style.color = data.ollamaReachable ? "#34d399" : "#f87171";
  } catch {
    chip.textContent = "🔴 Server status unavailable";
    chip.style.color = "#f87171";
  }
}

// 1. GRPO Actions & Training Step (real Ollama rollouts, real reward math)
function setupGRPOActions() {
  const btnRun = document.getElementById("btn-run-grpo-step");
  const inputPrompt = document.getElementById("input-train-prompt");
  const inputExpected = document.getElementById("input-train-expected");
  const inputModel = document.getElementById("input-train-model");
  const candidatesContainer = document.getElementById("candidates-container");
  const canvas = document.getElementById("grpo-loss-canvas");
  const errorBox = document.getElementById("grpo-error-box");

  async function executeStep() {
    btnRun.disabled = true;
    btnRun.textContent = "🧠 Sampling real rollouts from Ollama (may take a while)...";
    errorBox.style.display = "none";
    try {
      const res = await fetch("/api/train/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: inputPrompt.value,
          expectedAnswer: inputExpected.value,
          groupSize: 4,
          model: inputModel.value
        })
      });
      const data = await res.json();

      if (!res.ok) {
        errorBox.style.display = "block";
        errorBox.textContent = `⚠️ ${data.error || "GRPO step failed."}`;
        return;
      }

      trainingHistory.push(data);

      document.getElementById("chip-policy-loss").textContent = `📉 Policy Loss: ${data.policyLoss.toFixed(4)}`;
      document.getElementById("chip-mean-reward").textContent = `🎯 Mean Reward: ${data.meanReward.toFixed(2)}`;
      document.getElementById("badge-step-counter").textContent = `Step #${data.step}`;
      document.getElementById("badge-std-reward").textContent = `Std Reward: ${data.stdReward.toFixed(3)}`;

      if (data.generationErrors && data.generationErrors.length > 0) {
        errorBox.style.display = "block";
        errorBox.textContent = `⚠️ ${data.generationErrors.length} candidate(s) failed to generate: ${data.generationErrors.join(" | ")}`;
      }

      // Render Group Candidates (real Ollama completions)
      candidatesContainer.innerHTML = "";
      data.evaluations.forEach((cand, i) => {
        const isWinner = i === 0;
        const card = document.createElement("div");
        card.className = `candidate-card ${isWinner ? 'cand-winner' : ''}`;
        card.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong style="color: #fff;">Candidate #${i + 1} (${cand.candidateId}) ${isWinner ? '🏆 [HIGHEST REWARD]' : ''}</strong>
            <span style="font-family: var(--font-mono); color: #34d399; font-weight: 700;">Reward: ${cand.totalReward} | Adv: ${cand.normalizedAdvantage}</span>
          </div>
          <div style="font-size: 11px; color: var(--text-muted);">
            Think Tags: ${cand.hasThinkTags ? '✓ <think>' : '✗ Missing'} • Accuracy: ${(cand.accuracyReward * 100).toFixed(0)}% • Syntax: ${(cand.syntaxReward * 100).toFixed(0)}%
          </div>
          <pre style="background: #080c14; padding: 8px; border-radius: 4px; color: #93c5fd; font-family: var(--font-mono); font-size: 11px; white-space: pre-wrap; max-height: 90px; overflow-y: auto;">${escapeHtml(cand.response)}</pre>
        `;
        candidatesContainer.appendChild(card);
      });

      // Render Loss Curves on Canvas
      drawCurves(canvas, trainingHistory);
    } catch (e) {
      errorBox.style.display = "block";
      errorBox.textContent = `⚠️ Request failed: ${e.message}`;
    } finally {
      btnRun.disabled = false;
      btnRun.textContent = "🧠 Sample Real Rollouts from Ollama & Score Group";
    }
  }

  btnRun?.addEventListener("click", executeStep);
  executeStep(); // Auto-run initial step
}

function drawCurves(canvas, history) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
  ctx.lineWidth = 1;
  for (let y = 30; y < h; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  if (history.length === 0) return;

  // Plot Mean Reward (Green Line)
  ctx.beginPath();
  ctx.strokeStyle = "#34d399";
  ctx.lineWidth = 2.5;
  history.forEach((step, i) => {
    const x = (i / Math.max(1, history.length - 1)) * (w - 60) + 30;
    const y = h - (step.meanReward * (h - 60)) - 30;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Plot Policy Loss (Red Line)
  ctx.beginPath();
  ctx.strokeStyle = "#f87171";
  ctx.lineWidth = 2;
  history.forEach((step, i) => {
    const x = (i / Math.max(1, history.length - 1)) * (w - 60) + 30;
    const y = (step.policyLoss * (h - 60)) + 30;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Legend
  ctx.fillStyle = "#34d399";
  ctx.font = "10px 'Fira Code'";
  ctx.fillText("— Mean Reward (Ascending)", 30, 20);
  ctx.fillStyle = "#f87171";
  ctx.fillText("— Policy Loss (Descending)", 220, 20);
}

function escapeHtml(text) {
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
  return text.replace(/[&<>"']/g, m => map[m]);
}

// 2. Continual Post-Training tab: executes a real 5-case JS test suite
//    (baseline vs optimized snippet) and reports the ACTUAL pass/fail
//    percentages returned by the server — no field is invented client-side.
function setupPostTrainingActions() {
  const btnEpoch = document.getElementById("btn-run-posttrain-epoch");
  const baseInput = document.getElementById("input-base-checkpoint");
  const domainSelect = document.getElementById("select-posttrain-domain");
  const resultsBox = document.getElementById("posttrain-results-box");

  async function runEpoch() {
    btnEpoch.disabled = true;
    btnEpoch.textContent = "⚡ Running baseline vs optimized test suite...";
    try {
      const res = await fetch("/api/train/post-training", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          baseCheckpoint: baseInput.value,
          domain: domainSelect.value
        })
      });
      const data = await res.json();

      if (!res.ok) {
        resultsBox.innerHTML = `<span style="color:#f87171;">⚠️ ${data.error || "Post-training epoch failed."}</span>`;
        return;
      }

      resultsBox.innerHTML = `
        <strong style="color: #fff; font-size: 13px;">Epoch #${data.epoch}: ${data.baseCheckpoint}</strong><br>
        • <strong>Domain (label):</strong> <span style="color: #38bdf8;">${data.activeDomain}</span><br>
        • <strong>Test cases:</strong> ${data.totalTestSuiteCases}<br>
        • <strong>Baseline pass rate:</strong> <span style="color: #f87171;">${data.baselineTestsPassed}/${data.totalTestSuiteCases} (${data.empiricalAccuracyBeforePercent}%)</span>
          ➔ <strong style="color: #34d399; font-size: 14px;">Optimized: ${data.optimizedTestsPassed}/${data.totalTestSuiteCases} (${data.empiricalAccuracyAfterPercent}%)</strong><br>
        • <strong>Measured gain:</strong> <span style="font-family: var(--font-mono); color: #fbbf24;">${data.empiricalGainPercent >= 0 ? '+' : ''}${data.empiricalGainPercent}%</span> (computed from the real test run above, not fixed)<br>
        <span style="color: #34d399; font-weight: 600;">✓ ${data.optimizedTestsPassed}/${data.totalTestSuiteCases} optimized snippets passed real execution.</span>
      `;
    } catch (e) {
      resultsBox.innerHTML = `<span style="color:#f87171;">⚠️ Request failed: ${e.message}</span>`;
    } finally {
      btnEpoch.disabled = false;
      btnEpoch.textContent = "⚡ Execute Baseline vs Optimized Test Suite";
    }
  }

  btnEpoch?.addEventListener("click", runEpoch);
  runEpoch(); // Auto-run initial epoch
}
