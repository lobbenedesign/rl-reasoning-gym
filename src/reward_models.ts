/**
 * 🎯 Verifiable Reward Models for GRPO Training
 * Evaluates candidate responses using verifiable logic, syntax checking,
 * reasoning tag validation (<think>), and mathematical equivalence.
 *
 * When real `codeTestCases` are supplied (RLVR-style), the accuracy reward
 * is computed from ACTUAL code execution (see code_execution_reward.ts)
 * instead of substring matching — closing a gap against HuggingFace TRL /
 * open-r1's executable "code_reward" for verifiable-reward RL. See
 * code_execution_reward.ts for the full rationale and competitor citation.
 */

import { CodeExecutionRewardEngine, type CodeTestCase, type CodeExecutionResult } from "./code_execution_reward";
import { cascadeScore, type VerifyResult } from "./tasks";

export interface CandidateEvaluation {
  candidateId: string;
  response: string;
  hasThinkTags: boolean;
  formatReward: number; // 0.0 - 1.0
  accuracyReward: number; // 0.0 - 1.0
  syntaxReward: number; // 0.0 - 1.0
  totalReward: number; // weighted sum
  normalizedAdvantage: number;
  codeExecution?: CodeExecutionResult; // present only when codeTestCases were supplied
  verification?: VerifyResult; // how the accuracy reward was decided (exact/numeric/substring/none)
}

export class RewardModelEngine {
  private codeEngine = new CodeExecutionRewardEngine();

  public async evaluateGroup(
    candidates: { id: string; response: string }[],
    expectedAnswer: string,
    codeTestCases?: CodeTestCase[]
  ): Promise<CandidateEvaluation[]> {
    const evals: CandidateEvaluation[] = [];

    for (const cand of candidates) {
      const resp = cand.response;

      // 1. Format Reward (<think> tags check)
      const hasThink = resp.includes("<think>") && resp.includes("</think>");
      const formatScore = hasThink ? 1.0 : 0.2;

      // 2. Accuracy Reward
      let accScore = 0.0;
      let codeExecution: CodeExecutionResult | undefined;
      let verification: VerifyResult | undefined;
      if (codeTestCases && codeTestCases.length > 0) {
        // RLVR path: real execution-based reward, not text matching.
        codeExecution = await this.codeEngine.evaluate(resp, codeTestCases);
        accScore = codeExecution.passRate;
      } else {
        // Cascade scorer (reasoning-gym style, see tasks.ts): exact match ->
        // numeric equivalence -> substring fallback. Replaces the old
        // "substring only" check, which scored a correct numeric answer 0
        // whenever it wasn't phrased with the exact expected string (e.g.
        // response "The answer is 42." vs expectedAnswer "42" used to fail
        // the substring check only if expectedAnswer itself wasn't literally
        // present — cascadeScore extracts and compares the actual number).
        verification = cascadeScore(resp, expectedAnswer);
        accScore = verification.score;
      }

      // 3. Syntax & Structure Reward
      let syntaxScore = 1.0;
      if (resp.includes("```")) {
        const codeBlocks = resp.split("```");
        if (codeBlocks.length % 2 === 0) syntaxScore = 0.4; // Unclosed code fence
      }

      const total = Number((formatScore * 0.3 + accScore * 0.5 + syntaxScore * 0.2).toFixed(3));

      evals.push({
        candidateId: cand.id,
        response: resp,
        hasThinkTags: hasThink,
        formatReward: formatScore,
        accuracyReward: accScore,
        syntaxReward: syntaxScore,
        totalReward: total,
        normalizedAdvantage: 0,
        codeExecution,
        verification
      });
    }

    // Calculate Group Relative Advantages: A_i = (R_i - mean(R)) / (std(R) + eps)
    const totalSum = evals.reduce((sum, e) => sum + e.totalReward, 0);
    const mean = totalSum / evals.length;
    const variance = evals.reduce((sum, e) => sum + Math.pow(e.totalReward - mean, 2), 0) / evals.length;
    const std = Math.sqrt(variance) || 0.001;

    evals.forEach(e => {
      e.normalizedAdvantage = Number(((e.totalReward - mean) / std).toFixed(3));
    });

    return evals;
  }
}
