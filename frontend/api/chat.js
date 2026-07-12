/**
 * POST /api/chat
 * Backend for the AI DOC medical assistant chat widget.
 * Keeps the LLM API key server-side — never exposed to the browser.
 *
 * Requires an environment variable set in your Vercel project settings
 * (Project → Settings → Environment Variables), NOT in any committed file:
 *   ANTHROPIC_API_KEY = sk-ant-...
 *
 * Get a key at https://console.anthropic.com  (Settings → API Keys).
 * To switch providers (e.g. OpenAI) swap the fetch call in callLLM() below —
 * everything else (guardrails, validation) stays the same.
 */

const SYSTEM_PROMPT = `You are the AI DOC Medical Assistant, embedded in a rare-disease research
platform (an academic project, not a certified medical device). You help users
understand symptoms, conditions, and general health topics.

Hard rules, never break these:
1. You do NOT diagnose. Never tell a user what condition they personally have.
   You can discuss conditions in general educational terms only.
2. You do NOT prescribe medication, dosages, or specific treatment plans.
3. If a message describes possible emergency symptoms (chest pain, difficulty
   breathing, stroke signs, severe bleeding, loss of consciousness, suicidal
   thoughts, self-harm, or similar), your FIRST sentence must tell them to
   seek emergency care immediately (in the US: call 911, or the 988 Suicide &
   Crisis Lifeline for suicidal thoughts) before anything else.
4. Always encourage consulting a licensed clinician for personal medical
   decisions — but don't repeat this as a robotic disclaimer every message;
   say it naturally where it fits.
5. You may mention that this platform's "Predict" tool exists (an AI-assisted
   differential diagnosis aid from symptoms/scans) but make clear its output
   is a research-use ranked list, not a diagnosis either.
6. Stay on medical/health topics. Politely decline unrelated requests
   (coding help, general trivia, etc.) and redirect back to health questions.
7. Keep answers concise and in plain language; explain medical terms you use.`;

async function callLLM(messages) {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "x-api-key": process.env.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify({
            model: "claude-haiku-4-5-20251001",
            max_tokens: 600,
            system: SYSTEM_PROMPT,
            messages,
        }),
    });

    if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Upstream LLM error (${response.status}): ${errText}`);
    }

    const data = await response.json();
    const textBlock = data.content?.find((b) => b.type === "text");
    return textBlock?.text || "Sorry, I wasn't able to generate a response. Please try again.";
}

export default async function handler(req, res) {
    if (req.method !== "POST") {
        return res.status(405).json({ error: "Method not allowed" });
    }

    if (!process.env.ANTHROPIC_API_KEY) {
        console.error("ANTHROPIC_API_KEY is not set in this environment");
        return res.status(500).json({ error: "Chat is not configured on the server yet." });
    }

    const { messages } = req.body || {};
    if (!Array.isArray(messages) || messages.length === 0) {
        return res.status(400).json({ error: "A non-empty 'messages' array is required." });
    }

    // ── Basic abuse/cost guardrails ──
    // Cap conversation length and individual message size. This is NOT a
    // substitute for real rate limiting or auth — see the note in ChatWidget.jsx
    // about gating this behind a signed-in user.
    const trimmed = messages.slice(-20);
    for (const m of trimmed) {
        if (
            !m ||
            (m.role !== "user" && m.role !== "assistant") ||
            typeof m.content !== "string" ||
            m.content.length === 0 ||
            m.content.length > 4000
        ) {
            return res.status(400).json({ error: "Invalid message format." });
        }
    }

    try {
        const reply = await callLLM(trimmed);
        return res.status(200).json({ reply });
    } catch (err) {
        console.error("Chat handler error:", err.message);
        return res.status(502).json({ error: "The assistant is temporarily unavailable. Please try again." });
    }
}