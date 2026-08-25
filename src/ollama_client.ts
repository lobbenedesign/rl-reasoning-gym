/**
 * 🔌 Minimal real Ollama client used to sample actual rollouts from a local model.
 * No fallback text is ever fabricated: if Ollama is unreachable or the model
 * is missing, the caller receives a thrown error and must surface it honestly
 * (the server returns HTTP 502 with the real error message instead of pretending
 * a candidate was generated).
 */

const OLLAMA_HOST = process.env.OLLAMA_HOST || "http://localhost:11434";

export interface OllamaGenerateOptions {
  model: string;
  prompt: string;
  temperature?: number;
  seed?: number;
}

export async function ollamaGenerate(opts: OllamaGenerateOptions): Promise<string> {
  const res = await fetch(`${OLLAMA_HOST}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: opts.model,
      prompt: opts.prompt,
      stream: false,
      options: {
        temperature: opts.temperature ?? 0.8,
        seed: opts.seed
      }
    })
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Ollama request failed (${res.status}): ${text || res.statusText}`);
  }

  const data: any = await res.json();
  if (typeof data.response !== "string") {
    throw new Error("Ollama returned an unexpected payload (no 'response' field)");
  }
  return data.response;
}

export async function ollamaIsReachable(): Promise<boolean> {
  try {
    const res = await fetch(`${OLLAMA_HOST}/api/tags`, { method: "GET" });
    return res.ok;
  } catch {
    return false;
  }
}
