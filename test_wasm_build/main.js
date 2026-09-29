import init, { detect_mgm } from "./pkg-release/quicksilver.js";

const els = {
    input: document.getElementById("file-input"),
    pickerText: document.getElementById("picker-text"),
    status: document.getElementById("status"),
    result: document.getElementById("result"),
    verdict: document.getElementById("verdict"),
    confidenceFill: document.getElementById("confidence-fill"),
    confidenceText: document.getElementById("confidence-text"),
    metaFile: document.getElementById("meta-file"),
    metaSize: document.getElementById("meta-size"),
    metaRate: document.getElementById("meta-rate"),
    metaTime: document.getElementById("meta-time"),
};

function setStatus(text, kind = "") {
    els.status.textContent = text;
    els.status.className = `status ${kind}`.trim();
}

function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 ** 2).toFixed(2)} MB`;
}

// detect_mgm() is synchronous and blocks the main thread, so give the browser
// two animation frames to paint the "Analyzing…" state before we start.
function nextPaint() {
    return new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
    );
}

function showResult(file, result, elapsedMs) {
    const isAI = result.label === "AI";
    els.verdict.textContent = isAI ? "AI-generated" : "Human-made";
    els.verdict.className = `verdict ${isAI ? "ai" : "human"}`;

    const pct = Math.round(result.confidence * 100);
    els.confidenceFill.style.width = `${pct}%`;
    els.confidenceText.textContent = `${pct}%`;

    els.metaFile.textContent = file.name;
    els.metaSize.textContent = `${formatBytes(file.size)} (${file.type || "unknown type"})`;
    els.metaRate.textContent = result.sample_rate
        ? `${result.sample_rate.toLocaleString()} Hz`
        : "—";
    els.metaTime.textContent = `${elapsedMs.toFixed(1)} ms`;

    els.result.hidden = false;
    setStatus(`Done — “${file.name}” analyzed.`);
}

async function handleFile(file) {
    if (!file) return;

    els.result.hidden = true;
    els.pickerText.textContent = file.name;
    setStatus(`Reading “${file.name}” (${formatBytes(file.size)})…`);

    try {
        const bytes = new Uint8Array(await file.arrayBuffer());

        setStatus(`Analyzing “${file.name}” — decoding + inference running…`, "busy");
        await nextPaint();

        const t0 = performance.now();
        const result = detect_mgm(bytes, file.name);
        const elapsedMs = performance.now() - t0;

        showResult(file, result, elapsedMs);
    } catch (err) {
        // Errors from Rust surface as JS exceptions (decode failures,
        // unsupported formats, audio too short, ...).
        const message = err instanceof Error ? err.message : String(err);
        setStatus(`Failed: ${message}`, "error");
    }
}

async function boot() {
    try {
        const t0 = performance.now();
        await init();
        const loadMs = performance.now() - t0;

        setStatus(`Ready — model loaded in ${loadMs.toFixed(0)} ms. Pick a file to analyze.`);
        els.input.disabled = false;
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setStatus(`Failed to load the WASM module: ${message}`, "error");
    }

    els.input.addEventListener("change", () => {
        handleFile(els.input.files[0]);
        // Allow picking the same file again to re-run inference.
        els.input.value = "";
    });
}

boot();
