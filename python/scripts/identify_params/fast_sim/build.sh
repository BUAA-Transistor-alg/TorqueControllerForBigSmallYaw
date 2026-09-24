#!/usr/bin/env bash
# ============================================================================
# build.sh —— 编译 fast_sim 共享库（供 ctypes 加载）
# ============================================================================
# ★ **完全独立于主工程**: 不用 CMake、不 include 任何项目头文件、不产出到 build/；
#   只依赖 C++17 标准库 + libm。产物: ./_build/libfast_sim.so
#
# -ffp-contract=off : 禁止 FMA 收缩，让结果与 model.py 的 numpy 参考实现只差 ~1e-14
#                     （那点残差来自 tanh 实现差异）。去掉会再多一层 ULP 级差异。
# -march=native     : 针对本机指令集（换机器请重跑本脚本）
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-g++}"
OUT_DIR="$HERE/_build"
OUT="$OUT_DIR/libfast_sim.so"
SRC="$HERE/fast_sim.cpp"

mkdir -p "$OUT_DIR"

# 可选: 用 NO_MARCH_NATIVE=1 关掉 -march=native（跨机器分发时用）
MARCH_FLAG=(-march=native)
if [[ "${NO_MARCH_NATIVE:-0}" == "1" ]]; then
    MARCH_FLAG=()
fi

set -x
"$CXX" -O3 "${MARCH_FLAG[@]}" -std=c++17 -shared -fPIC -pthread \
       -ffp-contract=off -fno-math-errno \
       "$SRC" -o "$OUT" -lm
set +x

echo "[fast_sim] 已编译: $OUT"
