#!/bin/bash
# 两组仿真数据（刚性大 yaw = 无背隙 + 近似刚性耦合）: 地面平放 vs 倾斜 10°。
# 小 yaw 上装 P 与大 yaw 转子 Pb 都带**随机重心偏置**，且由同一个 --sim-com-seed 决定
# ⇒ 水平/倾斜、train/val 四次运行共享同一组重心真值（物理对象相同，只是姿态/激励不同）。
set -u
cd /home/huhu233/rm2027/TorqueControllerForBigSmallYaw
A=data/archive/20260920_sim_com_tilt
COMMON="--dry-run --sim-no-backlash --sim-no-backlash-k=2500 --sim-com-random --sim-com-seed=7 --duration-sec=3"
gen() {  # $1=目录 $2=segments $3=seed $4=额外参数
  local out=$1 n=$2 seed=$3 extra=$4
  mkdir -p "$A/$out"
  timeout 3600 python3 python/scripts/collect_sysid.py $COMMON $extra \
    --segments="$n" --seed="$seed" --out="$A/$out" > "$A/gen_$out.log" 2>&1
  echo "$out exit=$?"
}
gen level_train 16 42 ""              &
gen level_val    4 555 ""             &
gen tilt_train  16 42 "--sim-tilt-deg=10" &
gen tilt_val     4 555 "--sim-tilt-deg=10" &
wait
echo "GEN DONE"
