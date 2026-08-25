/**
 * 🏋️ Procedurally-generated verifiable reasoning tasks (reasoning-gym style).
 *
 * Gap identified vs. the real competitor this project's name most directly
 * echoes: github.com/open-thought/reasoning-gym. That project ships 100+
 * procedurally-generated task *datasets* across domains (arithmetic, algebra,
 * string/computation, logic, games, ...), each with a `score_answer` method
 * that verifies a model's answer programmatically — no LLM grading, no
 * substring guesswork. It also documents a "cascade scorer": try an exact
 * match first, then a numeric-equivalence comparison, then a lenient
 * fallback, to avoid penalizing a correct answer for cosmetic formatting
 * differences (e.g. "42" vs "The answer is 42." vs "= 42").
 *
 * Before this file existed, this repo only had ONE way to pose a task (a
 * free-text prompt + a plain substring match against `expectedAnswer`), which
 * is fragile: a correct numeric answer written differently ("6" vs "six",
 * "  6" vs "6.0") would score 0 even though it is right. This module adds:
 *
 *   1. Three genuinely distinct, procedurally generated task *types* (not
 *      three static example prompts) with real ground truth computed by the
 *      code itself — arithmetic word problems, string-manipulation tasks,
 *      and list/logic tasks — mirroring reasoning-gym's domain spread on a
 *      much smaller, honest scale.
 *   2. A real programmatic verifier per task (`verify(response)`), never an
 *      LLM-graded judgment.
 *   3. A `cascadeScore()` used generically by reward_models.ts for ANY
 *      accuracy check (task-based or plain expectedAnswer): exact trimmed
 *      match -> numeric equivalence -> substring fallback. This is a direct,
 *      measurable improvement over the old "only substring" logic.
 *
 * A seeded PRNG (mulberry32) makes every generated task reproducible from
 * its `seed` — the same seed always yields the same problem and the same
 * ground truth, so results here are inspectable and re-runnable, not
 * generated-then-thrown-away.
 */

export type TaskType = "arithmetic_word_problem" | "string_manipulation" | "list_logic";

export interface GeneratedTask {
  taskType: TaskType;
  seed: number;
  prompt: string;
  expectedAnswer: string;
  /** Real programmatic verifier — never LLM-graded. */
  verify: (response: string) => VerifyResult;
}

export interface VerifyResult {
  correct: boolean;
  score: number; // 0.0 - 1.0
  method: "exact" | "numeric" | "substring" | "none";
  extracted: string | null;
}

/** Deterministic PRNG so a given seed always reproduces the same task. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randInt(rng: () => number, min: number, max: number): number {
  return Math.floor(rng() * (max - min + 1)) + min;
}

function pick<T>(rng: () => number, arr: T[]): T {
  return arr[Math.floor(rng() * arr.length)];
}

/** Extracts the first numeric literal (int or float, optionally negative) from free text. */
function extractFirstNumber(text: string): number | null {
  const m = text.match(/-?\d+(?:\.\d+)?/);
  if (!m) return null;
  const n = Number(m[0]);
  return Number.isFinite(n) ? n : null;
}

/**
 * Cascade scorer (reasoning-gym style): exact match first, then numeric
 * equivalence, then substring fallback. Used for BOTH task-generated
 * ground truth and the plain expectedAnswer path in reward_models.ts.
 */
export function cascadeScore(response: string, expectedAnswer: string): VerifyResult {
  const respTrim = response.trim();
  const expTrim = expectedAnswer.trim();

  // 1. Exact match (ignoring surrounding whitespace/case).
  if (respTrim.toLowerCase() === expTrim.toLowerCase()) {
    return { correct: true, score: 1.0, method: "exact", extracted: respTrim };
  }

  // 2. Numeric equivalence: if expectedAnswer parses as a number, look for
  //    ANY number in the response that equals it exactly (handles "42",
  //    "The answer is 42.", "= 42", "42.0").
  const expNum = Number(expTrim);
  if (Number.isFinite(expNum) && expTrim !== "") {
    const respNum = extractFirstNumber(response);
    // Also check every number in the response, not just the first, in case
    // the model's reasoning mentions other numbers before the final answer.
    const allNums = [...response.matchAll(/-?\d+(?:\.\d+)?/g)].map(m => Number(m[0]));
    const match = allNums.find(n => Math.abs(n - expNum) < 1e-9);
    if (match !== undefined) {
      return { correct: true, score: 1.0, method: "numeric", extracted: String(match) };
    }
    if (respNum !== null) {
      // Wrong number, but a number was extracted — 0 credit, but honestly
      // report what was found instead of silently failing.
      return { correct: false, score: 0, method: "numeric", extracted: String(respNum) };
    }
  }

  // 3. Substring fallback (legacy behaviour, kept for free-text answers).
  if (response.includes(expectedAnswer)) {
    return { correct: true, score: 1.0, method: "substring", extracted: expectedAnswer };
  }
  if (expTrim.length > 0 && response.toLowerCase().includes(expTrim.toLowerCase().slice(0, 10))) {
    return { correct: false, score: 0.5, method: "substring", extracted: null };
  }

  return { correct: false, score: 0, method: "none", extracted: null };
}

// ---------------------------------------------------------------------------
// Task generators
// ---------------------------------------------------------------------------

function genArithmeticWordProblem(seed: number): GeneratedTask {
  const rng = mulberry32(seed);
  const a = randInt(rng, 3, 47);
  const b = randInt(rng, 2, 25);
  const op = pick(rng, ["+", "-", "*"] as const);
  const names = ["Marco", "Giulia", "Luca", "Elena", "Sara", "Paolo"];
  const items = ["palloni da volley", "punti segnati", "allenamenti", "biglietti", "mele", "monete"];
  const person = pick(rng, names);
  const item = pick(rng, items);

  let expected: number;
  let prompt: string;
  if (op === "+") {
    expected = a + b;
    prompt = `${person} ha ${a} ${item}. Ne riceve altri ${b}. Quanti ${item} ha in totale ${person}? Rispondi solo con il numero finale.`;
  } else if (op === "-") {
    // ensure non-negative and a >= b for a well-posed subtraction problem
    const hi = Math.max(a, b), lo = Math.min(a, b);
    expected = hi - lo;
    prompt = `${person} ha ${hi} ${item}. Ne usa/perde ${lo}. Quanti ${item} gli restano? Rispondi solo con il numero finale.`;
  } else {
    expected = a * b;
    prompt = `Ci sono ${a} gruppi da ${b} ${item} ciascuno. Quanti ${item} ci sono in totale? Rispondi solo con il numero finale.`;
  }

  const expectedAnswer = String(expected);
  return {
    taskType: "arithmetic_word_problem",
    seed,
    prompt,
    expectedAnswer,
    verify: (response: string) => cascadeScore(response, expectedAnswer)
  };
}

function genStringManipulation(seed: number): GeneratedTask {
  const rng = mulberry32(seed);
  const words = ["schiacciata", "ricezione", "pallavolo", "muro", "attacco", "battuta", "rotazione", "libero"];
  const word = pick(rng, words);
  const mode = pick(rng, ["reverse", "count_vowels", "first_last"] as const);

  let expected: string;
  let prompt: string;
  if (mode === "reverse") {
    expected = word.split("").reverse().join("");
    prompt = `Scrivi la parola "${word}" al contrario (lettera per lettera invertita). Rispondi solo con la parola invertita, senza spiegazioni.`;
  } else if (mode === "count_vowels") {
    const vowels = word.match(/[aeiouAEIOU]/g) || [];
    expected = String(vowels.length);
    prompt = `Quante vocali (a, e, i, o, u) contiene la parola "${word}"? Rispondi solo con il numero.`;
  } else {
    expected = `${word[0]}${word[word.length - 1]}`;
    prompt = `Qual è la prima e l'ultima lettera della parola "${word}", concatenate insieme (es. per "casa" la risposta è "aa")? Rispondi solo con le due lettere.`;
  }

  return {
    taskType: "string_manipulation",
    seed,
    prompt,
    expectedAnswer: expected,
    verify: (response: string) => {
      // For non-numeric string tasks the numeric branch of cascadeScore is a
      // no-op (expTrim won't parse as a number), so exact + substring still
      // apply correctly.
      return cascadeScore(response, expected);
    }
  };
}

function genListLogic(seed: number): GeneratedTask {
  const rng = mulberry32(seed);
  const len = randInt(rng, 4, 7);
  const nums: number[] = [];
  for (let i = 0; i < len; i++) nums.push(randInt(rng, 1, 99));
  const mode = pick(rng, ["max", "min", "sum"] as const);

  let expected: number;
  let verb: string;
  if (mode === "max") { expected = Math.max(...nums); verb = "il valore massimo"; }
  else if (mode === "min") { expected = Math.min(...nums); verb = "il valore minimo"; }
  else { expected = nums.reduce((s, n) => s + n, 0); verb = "la somma di tutti i valori"; }

  const prompt = `Dati questi numeri: [${nums.join(", ")}], qual è ${verb}? Rispondi solo con il numero.`;
  const expectedAnswer = String(expected);

  return {
    taskType: "list_logic",
    seed,
    prompt,
    expectedAnswer,
    verify: (response: string) => cascadeScore(response, expectedAnswer)
  };
}

const GENERATORS: Record<TaskType, (seed: number) => GeneratedTask> = {
  arithmetic_word_problem: genArithmeticWordProblem,
  string_manipulation: genStringManipulation,
  list_logic: genListLogic
};

export const TASK_TYPES: TaskType[] = Object.keys(GENERATORS) as TaskType[];

export function generateTask(taskType: TaskType, seed?: number): GeneratedTask {
  if (!GENERATORS[taskType]) {
    throw new Error(`Unknown task type "${taskType}". Valid types: ${TASK_TYPES.join(", ")}`);
  }
  const usedSeed = seed ?? (Date.now() % 1_000_000);
  return GENERATORS[taskType](usedSeed);
}
