// ─────────────── 1) Setup Canvas Drawing ───────────────
const canvas = document.getElementById("draw-canvas");
const ctx    = canvas.getContext("2d");
const RES    = 64; 
let drawing  = false;

// Initialize: black background, white “pen”
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "white";
ctx.lineWidth   = 10;
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

  // A) Downscale canvas to 64 x 64
  const off = document.createElement("canvas");
  off.width = RES; off.height = RES;
  const offCtx = off.getContext("2d");

  // ← Disable image smoothing so we get hard pixels, matching training data
  offCtx.imageSmoothingEnabled = false;
  offCtx.imageSmoothingQuality = "low";

  offCtx.drawImage(canvas, 0, 0, RES, RES);

  // ↓ Show a 5× magnified preview so we can debug what the model sees
  const preview = document.getElementById("preview-canvas");
  preview.getContext("2d").imageSmoothingEnabled = false;
  preview.getContext("2d").drawImage(off, 0, 0, 140, 140);

  // B) Extract and normalize pixels into Float32Array
const imgData = offCtx.getImageData(0, 0, RES, RES).data;
let input     = new Float32Array(RES * RES);
for (let i = 0; i < RES * RES; i++) {
  const idx = 4 * i;
  const avg  = (imgData[idx] + imgData[idx + 1] + imgData[idx + 2]) / 3;
  const norm = avg / 255;
  input[i]   = norm > 0.5 ? 1.0 : 0.0;
}

// ─── Morphological Closing: Dilate then Erode ───
const SIZE = RES;
function closing(bin) {
  // 1) Dilation
  let dilated = new Float32Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    if (bin[i] === 1) {
      dilated[i] = 1;
    } else {
      const x = i % SIZE, y = Math.floor(i / SIZE);
      let found = false;
      for (let dy = -1; dy <= 1 && !found; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx >= 0 && xx < SIZE && yy >= 0 && yy < SIZE) {
            if (bin[yy*SIZE + xx] === 1) {
              dilated[i] = 1;
              found = true;
              break;
            }
          }
        }
      }
    }
  }
  // 2) Erosion
  let closed = new Float32Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    if (dilated[i] === 0) {
      closed[i] = 0;
    } else {
      const x = i % SIZE, y = Math.floor(i / SIZE);
      let allOnes = true;
      for (let dy = -1; dy <= 1 && allOnes; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx >= 0 && xx < SIZE && yy >= 0 && yy < SIZE) {
            if (dilated[yy*SIZE + xx] === 0) {
              allOnes = false;
              break;
            }
          }
        }
      }
      closed[i] = allOnes ? 1 : 0;
    }
  }
  return closed;
}

// Apply closing
input = closing(input);

  // C) Run ONNX inference
  const tensor = new ort.Tensor("float32", input, [1, 1, RES, RES]);
  const outputMap = await session.run({ input: tensor });
  const scores = outputMap.output.data;  // Float32Array of length 4

  console.log("Raw model scores:", scores);

  // D) Map to labels
  const labels = ["circle", "square", "triangle", "star"];
  const maxIdx = scores.indexOf(Math.max(...scores));
  document.getElementById("result").textContent =
    `I’m pretty sure that’s a: ${labels[maxIdx]}`;
};
