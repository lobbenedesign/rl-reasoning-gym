/**
 * 🔁 Multi-round GRPO loop with real few-shot self-refinement feedback.
 *
 * HONESTY NOTE (read this before trusting the numbers this module produces):
 * this is NOT weight-space training. There is still no gradient step, no
 * optimizer, no checkpoint — exactly like grpo_trainer.ts. What this module
 * adds is a *prompt-space* loop: it runs several real GRPO rounds back to
 * back on the same task, and after each round it takes the single
 * highest-reward REAL response from that round and feeds it back into the
 * next round's system prompt as a one-shot example (via
 * GRPOTrainer.runStep's `fewShotPreamble` argument). This is a cheap, real
 * proxy for how iterative self-refinement / rejection-sampling fine-tuning
 * (RFT) loops are described in the GRPO/DeepSeekMath literature and in
 * verl's iterative-rollout design — except here nothing is fine-tuned, the
 * "learning" (if any) comes entirely from the in-context example changing
 * what the frozen model is prompted with each round.
 *
 * This module makes NO claim that reward will improve monotonically. Each
 * round is a real Ollama rollout scored by the real reward model in
 * reward_models.ts; if round 3's mean reward is lower than round 1's, that
 * is reported as-is in `monotonicallyImproved: false` and the raw
 * per-round numbers — never smoothed, clamped, or discarded.
 */

import { GRPOTrainer, type GRPOTrainingStep } from "./grpo_trainer";
import type { CodeTestCase } from "./code_execution_reward";

export interface MultiRoundResult {
  taskPrompt: string;
  expectedAnswer: string;
  model: string;
  groupSize: number;
  rounds: GRPOTrainingStep[];
  meanRewardByRound: number[];
  /** true only if every round's mean reward is >= the previous round's (real check, not assumed). */
  monotonicallyImproved: boolean;
  /** meanReward of the last round minus meanReward of the first round (can be negative). */
  netRewardDelta: number;
  summary: string;
}

export class MultiRoundTrainer {
  private trainer = new GRPOTrainer();

  public async runMultiRound(
    prompt: string,
    expectedAnswer: string,
    rounds: number = 3,
    groupSize: number = 3,
    model: string = "llama3.2:3b",
    codeTestCases?: CodeTestCase[]
  ): Promise<MultiRoundResult> {
    if (rounds < 1) throw new Error("rounds must be >= 1");
    if (rounds > 8) throw new Error("rounds capped at 8 to keep sequential local-Ollama latency reasonable");

    const stepResults: GRPOTrainingStep[] = [];
    let fewShot: string | undefined = undefined;

    for (let r = 0; r < rounds; r++) {
      const step = await this.trainer.runStep(prompt, expectedAnswer, groupSize, model, codeTestCases, fewShot);
      stepResults.push(step);

      // Feed the REAL best (highest totalReward) response from this round
      // forward as the next round's few-shot example — but only if it
      // actually scored a positive accuracy reward; feeding forward a wrong
      // answer as a "successful example" would be dishonest.
      const best = step.evaluations[0]; // evaluations are sorted desc by totalReward in grpo_trainer.ts
      if (best && best.accuracyReward > 0) {
        fewShot = best.response.length > 800 ? best.response.slice(0, 800) + " …" : best.response;
      }
      // If nothing in the round was correct, fewShot is left as whatever it
      // was before (or undefined) — we don't overwrite a good example with a
      // bad one, and we don't fabricate one either.
    }

    const meanRewardByRound = stepResults.map(s => s.meanReward);
    let monotonic = true;
    for (let i = 1; i < meanRewardByRound.length; i++) {
      if (meanRewardByRound[i] < meanRewardByRound[i - 1]) { monotonic = false; break; }
    }
    const netDelta = Number((meanRewardByRound[meanRewardByRound.length - 1] - meanRewardByRound[0]).toFixed(3));

    const summary = monotonic
      ? `Mean reward improved or held steady every round (${meanRewardByRound.join(" -> ")}), net change ${netDelta >= 0 ? "+" : ""}${netDelta}.`
      : `Mean reward did NOT improve monotonically (${meanRewardByRound.join(" -> ")}) — net change ${netDelta >= 0 ? "+" : ""}${netDelta}. Few-shot self-refinement on a frozen ${model} is not guaranteed to help every round; this is reported honestly rather than cherry-picked.`;

    return {
      taskPrompt: prompt,
      expectedAnswer,
      model,
      groupSize,
      rounds: stepResults,
      meanRewardByRound,
      monotonicallyImproved: monotonic,
      netRewardDelta: netDelta,
      summary
    };
  }
}
