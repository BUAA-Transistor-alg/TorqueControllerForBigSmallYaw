#!/bin/bash
# 4 个辨识臂: {水平, 倾斜} × {β 用在线列(auto), β 拟合全局常数(fit)}
# 初始值: 机械量(平面 8 + 电机 3 + δ/k/c)用**被控对象真值**，而**重心未知 ⇒ Px=Py=Pbx=Pby=0**
#         这样比的才是"重心参数能不能从这份数据里认出来"。
set -u
cd /home/huhu233/rm2027/TorqueControllerForBigSmallYaw
A=data/archive/20260920_sim_com_tilt
# 机械初值（= 被控对象真值；δ 用 0.002 表示"几乎没有空行程"）
INIT="0.050,0.020,0,0,0.220,0.055,0.0973,0.028,0.002,2500.0,2.0,0.002,0.006,0.030,0.010,0.0,0,0"
fit() {  # $1=臂名 $2=数据目录
  local n=$1 d=$2
  mkdir -p "$A/$n"
  timeout 7200 python3 python/scripts/identify_params_torch.py \
    --data="$A/$d/*.npz" --val-data="$A/${d%_train}_val/*.npz" \
    --epochs=3000 --batch-segments --substeps=4 --threads=4 \
    --eval-every=200 --print-every=200 --state-mode=est \
    --init-vector="$INIT" --plot-out="$A/$n/ident" --out="$A/$n/params.txt" \
    > "$A/$n/run.log" 2>&1
  echo "$n exit=$?"
}
fit level_auto level_train &
fit tilt_auto  tilt_train  &
wait
INIT_FIT="$INIT"
fit2() {
  local n=$1 d=$2
  mkdir -p "$A/$n"
  timeout 7200 python3 python/scripts/identify_params_torch.py \
    --data="$A/$d/*.npz" --val-data="$A/${d%_train}_val/*.npz" \
    --epochs=3000 --batch-segments --substeps=4 --threads=4 \
    --eval-every=200 --print-every=200 --state-mode=est --beta-mode=fit \
    --init-vector="$INIT_FIT" --plot-out="$A/$n/ident" --out="$A/$n/params.txt" \
    > "$A/$n/run.log" 2>&1
  echo "$n exit=$?"
}
fit2 level_betafit level_train &
fit2 tilt_betafit  tilt_train  &
wait
echo "FITS DONE"
