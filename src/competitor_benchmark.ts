/**
 * 📊 5-Competitor Benchmark Matrix for Reinforcement Learning Studios
 * Compares RL-Reasoning Gym against:
 * 1. OpenRLHF
 * 2. Unsloth (Unsloth Studio GRPO)
 * 3. HuggingFace Open-R1
 * 4. rLLM
 * 5. ReST-RL
 */

export interface RLCompetitor {
  name: string;
  grpoSupport: boolean;
  requiresSeparateCritic: boolean;
  localAppleSiliconMac: boolean;
  visualTrainingCurves: boolean;
  verifiableRewardEngine: boolean;
  minVramRequiredGB: number;
}

export class RLCompetitorBenchmark {
  public getComparison(): RLCompetitor[] {
    return [
      {
        name: "🧠 RL-Reasoning Gym (Our Software)",
        grpoSupport: true,
        requiresSeparateCritic: false,
        localAppleSiliconMac: true,
        visualTrainingCurves: true,
        verifiableRewardEngine: true,
        minVramRequiredGB: 8
      },
      {
        name: "Unsloth Studio (GRPO)",
        grpoSupport: true,
        requiresSeparateCritic: false,
        localAppleSiliconMac: false, // CUDA/Triton specific
        visualTrainingCurves: false,
        verifiableRewardEngine: true,
        minVramRequiredGB: 7
      },
      {
        name: "OpenRLHF",
        grpoSupport: true,
        requiresSeparateCritic: false,
        localAppleSiliconMac: false,
        visualTrainingCurves: false,
        verifiableRewardEngine: false,
        minVramRequiredGB: 24
      },
      {
        name: "HuggingFace Open-R1",
        grpoSupport: true,
        requiresSeparateCritic: false,
        localAppleSiliconMac: false,
        visualTrainingCurves: false,
        verifiableRewardEngine: true,
        minVramRequiredGB: 32
      },
      {
        name: "rLLM",
        grpoSupport: true,
        requiresSeparateCritic: false,
        localAppleSiliconMac: false,
        visualTrainingCurves: false,
        verifiableRewardEngine: false,
        minVramRequiredGB: 16
      },
      {
        name: "ReST-RL",
        grpoSupport: false, // DPO/PPO
        requiresSeparateCritic: true,
        localAppleSiliconMac: false,
        visualTrainingCurves: false,
        verifiableRewardEngine: false,
        minVramRequiredGB: 48
      }
    ];
  }
}
