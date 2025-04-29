// ─────────────── 1) Setup Canvas Drawing ───────────────
const canvas = document.getElementById("draw-canvas");
const ctx    = canvas.getContext("2d");
let drawing  = false;

// Initialize: black background, white “pen”
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "white";
ctx.lineWidth   = 2;
ctx.lineCap     = "round";

canvas.addEventListener("pointerdown", () => { drawing = true; });
canvas.addEventListener("pointerup",   () => { drawing = false; ctx.beginPath(); });
canvas.addEventListener("pointermove", e => {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
});

// ─────────────── 2) Clear Button ───────────────
document.getElementById("clear-btn").onclick = () => {
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result").textContent = "Canvas cleared.";
};

// ─────────────── 3) Load ONNX Model ───────────────
let session;
ort.InferenceSession.create("shape_classifier.onnx")
  .then(s => {
    session = s;
    document.getElementById("result").textContent = "Model loaded! Draw and click Predict.";
  })
  .catch(err => console.error("Failed to load ONNX model:", err));

// ─────────────── 4) Predict Button ───────────────
document.getElementById("predict-btn").onclick = async () => {
  if (!session) {
    alert("Model still loading…");
    return;
  }

  // A) Downscale canvas to 28×28 offscreen (no smoothing)
  const off = document.createElement("canvas");
  off.width = 28; off.height = 28;
  const offCtx = off.getContext("2d");

  // ← Disable image smoothing so we get hard pixels, matching training data
  offCtx.imageSmoothingEnabled = false;
  offCtx.imageSmoothingQuality = "low";

  offCtx.drawImage(canvas, 0, 0, 28, 28);

  // ↓ Show a 5× magnified preview so we can debug what the model sees
  const preview = document.getElementById("preview-canvas");
  preview.getContext("2d").imageSmoothingEnabled = false;
  preview.getContext("2d").drawImage(off, 0, 0, 140, 140);

  // B) Extract and normalize pixels into Float32Array [1,1,28,28]
  const imgData = offCtx.getImageData(0, 0, 28, 28).data;
  const input   = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    const idx = 4 * i;
    const avg = (imgData[idx] + imgData[idx + 1] + imgData[idx + 2]) / 3;
    input[i] = avg / 255;
  }

  // C) Run ONNX inference
  const tensor = new ort.Tensor("float32", input, [1, 1, 28, 28]);
  const outputMap = await session.run({ input: tensor });
  const scores = outputMap.output.data;  // Float32Array of length 4

  console.log("Raw model scores:", scores);

  // D) Map to labels
  const labels = ["circle", "square", "triangle", "star"];
  const maxIdx = scores.indexOf(Math.max(...scores));
  document.getElementById("result").textContent =
    `I’m pretty sure that’s a: ${labels[maxIdx]}`;
};
