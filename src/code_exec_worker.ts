/**
 * 🧪 Isolated code-execution worker for the verifiable code reward.
 * Runs in a separate Bun Worker thread so a hanging/malicious candidate
 * cannot take down the server. Receives extracted JS/TS source plus a set
 * of {input, expectedOutput} test cases, REALLY calls the candidate's first
 * exported (or top-level) function with each input, and reports genuine
 * pass/fail counts. Nothing here is simulated — a failing candidate fails.
 */

interface TestCase {
  input: any[];
  expectedOutput: any;
}

function safeStringify(v: unknown): string {
  try {
    return (JSON.stringify(v) ?? String(v)).slice(0, 200);
  } catch {
    return String(v).slice(0, 200);
  }
}

self.postMessage({ type: "ready" });

self.onmessage = async (event: MessageEvent) => {
  const { code, testCases } = event.data as { code: string; testCases: TestCase[] };
  const logs: string[] = [];

  let stripped: string;
  try {
    const transpiler = new Bun.Transpiler({ loader: "ts" });
    stripped = transpiler.transformSync(code);
  } catch (e: any) {
    self.postMessage({ syntaxValid: false, testsPassed: 0, totalTests: testCases.length, logs: [`✗ Compile error: ${e.message}`] });
    return;
  }

  const exportedNames = Array.from(
    stripped.matchAll(/export\s+(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)/g)
  ).map((m) => m[1]).concat(
    Array.from(stripped.matchAll(/export\s+const\s+([A-Za-z_$][\w$]*)\s*=/g)).map((m) => m[1])
  );

  // If the model didn't export anything, fall back to the first top-level
  // `function name(` declaration so plain (non-exported) candidate code
  // written by an LLM still gets a fair, real execution attempt.
  let harnessBody = stripped.replace(/export\s+default\s+/g, "").replace(/export\s+/g, "");
  let callableName = exportedNames[0];
  if (!callableName) {
    const topLevel = stripped.match(/function\s+([A-Za-z_$][\w$]*)\s*\(/);
    callableName = topLevel?.[1];
  }

  if (!callableName) {
    self.postMessage({ syntaxValid: true, testsPassed: 0, totalTests: testCases.length, logs: ["✗ No callable function found in candidate code."] });
    return;
  }

  const harness = `${harnessBody}\nreturn { ${callableName}: typeof ${callableName} !== 'undefined' ? ${callableName} : undefined };`;

  let mod: Record<string, unknown>;
  try {
    const factory = new Function(harness);
    mod = factory();
  } catch (e: any) {
    self.postMessage({ syntaxValid: false, testsPassed: 0, totalTests: testCases.length, logs: [`✗ Runtime load error: ${e.message}`] });
    return;
  }

  const fn = mod[callableName] as ((...a: any[]) => any) | undefined;
  if (typeof fn !== "function") {
    self.postMessage({ syntaxValid: true, testsPassed: 0, totalTests: testCases.length, logs: [`✗ "${callableName}" is not a callable function.`] });
    return;
  }

  let passed = 0;
  for (const [idx, tc] of testCases.entries()) {
    try {
      let result = fn(...tc.input);
      if (result && typeof (result as any).then === "function") result = await result;
      const ok = JSON.stringify(result) === JSON.stringify(tc.expectedOutput);
      if (ok) {
        passed++;
        logs.push(`✓ test ${idx + 1}: ${callableName}(${tc.input.map(safeStringify).join(", ")}) → ${safeStringify(result)} PASSED`);
      } else {
        logs.push(`✗ test ${idx + 1}: expected ${safeStringify(tc.expectedOutput)}, got ${safeStringify(result)}`);
      }
    } catch (e: any) {
      logs.push(`✗ test ${idx + 1}: runtime exception — ${e.message}`);
    }
  }

  self.postMessage({ syntaxValid: true, testsPassed: passed, totalTests: testCases.length, logs, fnName: callableName });
};
