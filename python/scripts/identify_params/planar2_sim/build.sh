#!/usr/bin/env bash
# ============================================================================
# build.sh —— 编译 planar2_sim 共享库（供 ctypes 加载）
# ============================================================================
# ★ **完全独立于主工程**: 不用 CMake、不 include 任何项目头文件、不产出到 build/；
#   只依赖 C++17 标准库 + libm。产物: ./_build/libplanar2_sim.so
#
# -ffp-contract=off : 禁止 FMA 收缩，让结果与 planar2.py 的 numpy 参考实现只差
#                     ~1e-13（那点残差来自 sin/cos/tanh 的实现差异）。
# -march=native     : 针对本机指令集（换机器请重跑本脚本）
# ★ 不要加 -fvisibility=hidden: 会把 extern "C" 符号藏掉，ctypes 就找不到
#   planar2_rollout / planar2_accel（旧的 fast_sim 踩过这个坑）。
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-g++}"
OUT_DIR="$HERE/_build"
OUT="$OUT_DIR/libplanar2_sim.so"
SRC="$HERE/planar2_sim.cpp"

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

echo "[planar2_sim] 已编译: $OUT"
