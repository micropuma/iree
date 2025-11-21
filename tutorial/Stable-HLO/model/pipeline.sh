#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# 用法:
#   ./run_iree_mobilenet.sh image.png
# 默认图片: image.png
# ---------------------------------------------------------------------------

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# 用户传入图片路径，否则默认 image.png
INPUT_IMAGE="${1:-image.png}"

ONNX_MODEL="mobilenetv2-10.onnx"
MLIR_FILE="mobilenetv2.mlir"
VMFB_FILE="mobilenet_cuda.vmfb"
INPUT_NPY="input.npy"
OUTPUT_NPY="out.npy"

echo "Using input image: $INPUT_IMAGE"


# ---------------------------------------------------------------------------
# 0. 检查图片存在
# ---------------------------------------------------------------------------
if [ ! -f "$INPUT_IMAGE" ]; then
  echo "❌ Error: image file '$INPUT_IMAGE' not found!"
  exit 1
fi


# ---------------------------------------------------------------------------
# 1. 将 ONNX 转 MLIR（如已存在可注释掉）
# ---------------------------------------------------------------------------
if [ ! -f "$MLIR_FILE" ]; then
  echo "==> Importing ONNX to MLIR ..."
  iree-import-onnx "$ONNX_MODEL" --opset-version 17 -o "$MLIR_FILE"
else
  echo "Skipping ONNX import (mlir already exists)"
fi


# ---------------------------------------------------------------------------
# 2. 编译 MLIR → VMFB
# ---------------------------------------------------------------------------
echo "==> Compiling MLIR to VMFB ..."
iree-compile \
    --iree-hal-target-backends=cuda \
    --iree-cuda-target=sm_86 \
    --mlir-print-ir-after-all \
    --mlir-pass-statistics \
    --mlir-timing \
    "$MLIR_FILE" -o "$VMFB_FILE" \
    2>&1 | tee compile_output.dump


# ---------------------------------------------------------------------------
# 3. 从真实图片生成 input.npy（支持 PNG/JPG/BMP/TIFF）
# ---------------------------------------------------------------------------
echo "==> Generating input.npy from image: $INPUT_IMAGE"

python3 <<EOF
import numpy as np
from PIL import Image
import sys

path = "$INPUT_IMAGE"

try:
    img = Image.open(path).convert("RGB")
except Exception as e:
    print("❌ Failed to load image:", e)
    sys.exit(1)

# MobileNet 需要 224x224
img = img.resize((224, 224))

arr = np.array(img).astype("float32") / 255.0
arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
arr = arr[np.newaxis, :]            # CHW -> NCHW

np.save("$INPUT_NPY", arr)
print("Saved $INPUT_NPY with shape:", arr.shape)
EOF


# ---------------------------------------------------------------------------
# 4. 执行 IREE inference
# ---------------------------------------------------------------------------
echo "==> Running IREE inference ..."
iree-run-module \
    --device=cuda \
    --module="$VMFB_FILE" \
    --function=torch-jit-export \
    --input=@$INPUT_NPY \
    --output=@$OUTPUT_NPY \
    --print_statistics=true

echo "Output saved to $OUTPUT_NPY"


# ---------------------------------------------------------------------------
# 5. 输出 Top-5 分类结果（自动下载标签库）
# ---------------------------------------------------------------------------
echo "==> Decoding top-5 predictions ..."

if [ ! -f imagenet-simple-labels.json ]; then
  echo "Downloading imagenet-simple-labels.json..."
  wget -q https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json
fi

python3 <<EOF
import numpy as np, json

out = np.load("$OUTPUT_NPY")[0]
labels = json.load(open("imagenet-simple-labels.json"))

# softmax
prob = np.exp(out) / np.sum(np.exp(out))
top5 = prob.argsort()[-5:][::-1]

print("\n=== Top-5 Predictions ===")
for i in top5:
    print(f"{labels[i]:30s} : {prob[i]:.4f}")
EOF

echo "Done!"
