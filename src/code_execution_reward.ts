/**
 * 🧪 Verifiable Code-Execution Reward (RLVR-style)
 *
 * Gap identified vs. real competitors this project already positions itself
 * against (see README's "Group Relative Policy Optimization... popularized
 * by DeepSeek-R1" framing, which places this repo alongside HuggingFace
 * TRL/OpenRLHF/open-r1-style GRPO tooling): those frameworks support
 * "Reinforcement Learning with Verifiable Rewards" (RLVR) for code tasks —
 * TRL's GRPOTrainer accepts pluggable `reward_funcs`, and open-r1's
 * `code_reward` function actually EXECUTES the candidate code against real
 * unit tests and scores the pass rate. Before this file existed,
 * src/reward_models.ts only checked whether the expected answer string
 * appeared as a substring of the response — no code was ever executed, so a
 * candidate could get full accuracy credit for merely mentioning the right
 * text without the code actually working, and a candidate with genuinely
 * correct logic but slightly different wording would score worse than a
 * candidate that just echoed the expected string.
 *
 * This module closes that specific gap: given a completion containing a
 * fenced code block and a set of real {input, expectedOutput} test cases, it
 * extracts the code, executes it in an isolated Bun Worker (same sandboxing
 * pattern used by the reasoning-tree-mcts sibling project), and returns the
 * REAL measured pass rate. No score here is guessed or hardcoded — a
 * candidate that doesn't compile or fails every test gets 0.
 */

export interface CodeTestCase {
  input: any[];
  expectedOutput: any;
}

export interface CodeExecutionResult {
  syntaxValid: boolean;
  testsPassed: number;
  totalTests: number;
  passRate: number; // 0.0 - 1.0, real measured testsPassed/totalTests
  logs: string[];
}

const WORKER_URL = new URL("./code_exec_worker.ts", import.meta.url);

/** Extracts the first fenced code block (```ts/```js/``` ) from a raw LLM response. */
export function extractCodeBlock(response: string): string | null {
  const match = response.match(/```(?:ts|typescript|js|javascript)?\n([\s\S]*?)```/);
  return match ? match[1] : null;
}

export class CodeExecutionRewardEngine {
  /** Executes candidate code from `response` against real test cases in an isolated worker. */
  public async evaluate(response: string, testCases: CodeTestCase[], timeoutMs = 4000): Promise<CodeExecutionResult> {
    const code = extractCodeBlock(response);
    if (!code) {
      return { syntaxValid: false, testsPassed: 0, totalTests: testCases.length, passRate: 0, logs: ["✗ No fenced code block found in the completion — cannot execute anything."] };
    }

    return new Promise<CodeExecutionResult>((resolve) => {
      let settled = false;
      let worker: Worker;
      try {
        worker = new Worker(WORKER_URL.href);
      } catch (e: any) {
        resolve({ syntaxValid: false, testsPassed: 0, totalTests: testCases.length, passRate: 0, logs: [`✗ Could not start isolated worker: ${e.message}`] });
        return;
      }

      const finish = (payload: any, timedOut = false) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        try { worker.terminate(); } catch {}
        const testsPassed = timedOut ? 0 : (payload.testsPassed ?? 0);
        const totalTests = payload.totalTests ?? testCases.length;
        resolve({
          syntaxValid: timedOut ? false : (payload.syntaxValid ?? false),
          testsPassed,
          totalTests,
          passRate: totalTests > 0 ? Number((testsPassed / totalTests).toFixed(3)) : 0,
          logs: timedOut ? ["✗ Real timeout: candidate code exceeded the execution limit and was terminated (possible infinite loop)."] : (payload.logs ?? [])
        });
      };

      const timer = setTimeout(() => finish({}, true), timeoutMs);
      worker.onmessage = (event: MessageEvent) => {
        const data = event.data as any;
        if (data && data.type === "ready") {
          worker.postMessage({ code, testCases });
          return;
        }
        finish(data);
      };
      worker.onerror = (e: any) => finish({ syntaxValid: false, logs: [`✗ Worker error: ${e.message ?? e}`] });
    });
  }
}
