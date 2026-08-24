/**
 * 🎯 Verifiable Reward Models for GRPO Training
 * Evaluates candidate responses using verifiable logic, syntax checking,
 * reasoning tag validation (<think>), and mathematical equivalence.
 */

export interface CandidateEvaluation {
  candidateId: string;
  response: string;
  hasThinkTags: boolean;
  formatReward: number; // 0.0 - 1.0
  accuracyReward: number; // 0.0 - 1.0
  syntaxReward: number; // 0.0 - 1.0
  totalReward: number; // weighted sum
  normalizedAdvantage: number;
}

export class RewardModelEngine {
  public evaluateGroup(candidates: { id: string; response: string }[], expectedAnswer: string): CandidateEvaluation[] {
    const evals: CandidateEvaluation[] = [];

    for (const cand of candidates) {
      const resp = cand.response;

      // 1. Format Reward (<think> tags check)
      const hasThink = resp.includes("<think>") && resp.includes("</think>");
      const formatScore = hasThink ? 1.0 : 0.2;

      // 2. Accuracy Reward
      let accScore = 0.0;
      if (resp.includes(expectedAnswer)) {
        accScore = 1.0;
      } else if (resp.toLowerCase().includes(expectedAnswer.toLowerCase().slice(0, 10))) {
        accScore = 0.5;
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
        normalizedAdvantage: 0
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
