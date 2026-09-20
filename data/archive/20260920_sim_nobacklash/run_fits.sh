#!/bin/bash
# 「没有背隙」仿真（环境 3）的 δ 可辨识性对照 —— 9 个臂，全部同数据、同超参。
#
#   只改 16 参初值里的 (δ,k,c)（其余 11 个从仿真真值出发），并消融两个来源:
#     · β 来源:  --beta-mode=auto（用数据里的在线 β 列） vs fit（拟合一个全局常数）
#     · 状态来源: --state-mode=est（控制器真正看到的） vs true（仿真真值，上限对照）
#
# 每个臂 ≈ 5 min（31 段 × 1500 epoch）；G10k 是 10000 epoch ≈ 30 min。全部并行跑。
set -u
cd /home/huhu233/rm2027/TorqueControllerForBigSmallYaw
A=data/archive/20260920_sim_nobacklash
D="$A/train/*.npz"; V="$A/val/*.npz"
TP="0.050,0.020,0.00866,0.005,0.220,0.055,0.0973,0.028"   # 平面 8 参 = 被控对象真值
TM="0.006,0.030,0.010"                                    # 电机侧 = 真值
CUR=$(python3 -c "import sys; sys.path.insert(0,'python/scripts')
import identify_params_torch as ip
v=ip.default_param_vector(); print(','.join(f'{x:.6g}' for x in v))")

fit() {   # $1=臂名 $2=init-vector $3=beta-mode $4=state-mode $5=epochs
  local n=$1 init=$2 bm=$3 sm=$4 ep=$5
  mkdir -p "$A/$n"
  timeout 7200 python3 python/scripts/identify_params_torch.py \
    --data="$D" --val-data="$V" --epochs="$ep" --batch-segments --substeps=2 --threads=3 \
    --eval-every=150 --print-every=150 --beta-mode="$bm" --state-mode="$sm" \
    --init-vector="$init" --plot-out="$A/$n/ident" --out="$A/$n/params.txt" \
    > "$A/$n/run.log" 2>&1
  echo "$n exit=$?"
}

# ── (δ,k,c) 初值消融，β = 在线列，状态 = 估计 ──
fit A_bogus_delta       "$TP,0.0965,157.8279,2.5911,0.002,$TM,0.0" auto est 1500 &
fit B_tiny_delta        "$TP,0.002,200.0,2.0,0.002,$TM,0.0"       auto est 1500 &
fit C_mid_delta         "$TP,0.03,200.0,2.0,0.002,$TM,0.0"        auto est 1500 &
# ── 完全用当前默认值（连平面 8 参也是实机辨识值）＝"照今天脚本直接跑" ──
fit D_current_defaults  "$CUR"                                    auto est 1500 &
# ── 状态用真值（隔离电机通道估计误差） ──
fit E_true_state        "$TP,0.002,200.0,2.0,0.002,$TM,0.0"       auto true 1500 &
fit F_bogus_true_state  "$TP,0.0965,157.8279,2.5911,0.002,$TM,0.0" auto true 1500 &
# ── β 改成"拟合一个全局常数"（把在线 β 的噪声拿掉） ──
fit G_bogus_betafit     "$TP,0.0965,157.8279,2.5911,0.002,$TM,0.0" fit true 1500 &
fit H_tiny_betafit      "$TP,0.002,200.0,2.0,0.002,$TM,0.0"        fit true 1500 &
# ── G 的加练: 区分"不可辨识"与"log 参数化走得太慢" ──
fit G10k_bogus_betafit  "$TP,0.0965,157.8279,2.5911,0.002,$TM,0.0" fit true 10000 &
wait
echo "ALL DONE"
