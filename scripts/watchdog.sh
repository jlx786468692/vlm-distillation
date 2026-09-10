#!/usr/bin/env bash
set -u
REPO_DIR="/data/workspace2/jlx/workspace/vlm-distillation"
CONFIG="configs/distill_train2014.yaml"
PYTHON_BIN="$(which python)"
TARGET_IMAGES=5000
STALE_MIN=20
CHECK_INTERVAL=60
GPU_ID=0
MAX_RESTARTS=40
MERGED_DIR="outputs/merged_train2014"
POISON_FILE="outputs/poison_ids.txt"
DISTILL_CMD="$PYTHON_BIN scripts/run_full_pipeline.py --config $CONFIG --steps distillation"
STEP2_CMD="$PYTHON_BIN scripts/run_full_pipeline.py --config $CONFIG --steps cleaning prepare_training_data"
PROC_PATTERN="run_full_pipeline.py.*distill_train2014"
cd "$REPO_DIR" || { echo "[ERR] cd 失败"; exit 1; }
mkdir -p logs
touch "$POISON_FILE"
log(){ echo "[$(date '+%F %T')] [watchdog] $*"; }
rebuild_ckpt(){
  $PYTHON_BIN - <<'PYEOF'
import json,glob,os,re
skip=set()
try:
    for line in open('outputs/poison_ids.txt'):
        line=line.strip()
        if line:
            try: skip.add(int(line))
            except: pass
except FileNotFoundError: pass
ids=[]; skipped=0
for p in glob.glob('outputs/merged_train2014/*.json'):
    try:
        d=json.load(open(p))
        if not isinstance(d,dict) or not d: skipped+=1; continue
        m=re.search(r'train2014_(\d+)',os.path.basename(p))
        if m: ids.append(int(m.group(1)))
        else: skipped+=1
    except: skipped+=1
ids=sorted(set(ids)|skip)
json.dump({'processed_ids':ids,'timestamp':'2026-09-10T00:00:00','total_processed':len(ids)},
          open('outputs/checkpoint_latest.json','w'),indent=2)
print(f'rebuild checkpoint_latest.json: {len(ids)} ids (skipped {skipped} invalid, skip_poison={sorted(skip)})')
PYEOF
}
detect_poison(){
  local pid_img
  pid_img=$(grep -oE "Processing image [0-9]+" logs/full_pipeline.log 2>/dev/null | tail -1 | grep -oE "[0-9]+$" || true)
  if [ -n "$pid_img" ]; then
    grep -qx "$pid_img" "$POISON_FILE" 2>/dev/null || echo "$pid_img" >> "$POISON_FILE"
    log "  毒图检测: $pid_img (写入 $POISON_FILE,后续重启跳过)"
  fi
}
log "==== 看门狗启动 PID=$$ ==== python=$PYTHON_BIN poison=$(tr '\n' ' ' <"$POISON_FILE" 2>/dev/null)"
if ! pgrep -f "$PROC_PATTERN" >/dev/null 2>&1; then
  log "未检测到蒸馏进程,重建checkpoint并拉起..."; rebuild_ckpt
  nohup bash -c "CUDA_VISIBLE_DEVICES=$GPU_ID $DISTILL_CMD" >> logs/full_pipeline.log 2>&1 &
  sleep 50
fi
restarts=0
while true; do
  end_time=$($PYTHON_BIN -c "import json;d=json.load(open('outputs/checkpoint_latest.json'));print(d.get('end_time') or '')" 2>/dev/null)
  n_merged=$(ls "$MERGED_DIR" 2>/dev/null | wc -l)
  if [ "$n_merged" -ge "$TARGET_IMAGES" ] || [ -n "$end_time" ]; then
    log "✓ 蒸馏完成! merged=$n_merged"; log "→ 接 ② cleaning+prepare_training_data..."
    bash -c "$STEP2_CMD" >> logs/full_pipeline.log 2>&1 && log "✓ 步骤②完成" || log "✗ 步骤②失败,手动: $STEP2_CMD"
    log "==== 看门狗退出 ===="; exit 0
  fi
  pid=$(pgrep -f "$PROC_PATTERN" | head -1)
  if [ -z "$pid" ]; then
    restarts=$((restarts+1)); [ "$restarts" -gt "$MAX_RESTARTS" ] && { log "✗ 达最大重启,退出"; exit 1; }
    log "进程已退出,第 $restarts 次重启 merged=$n_merged"; detect_poison; rebuild_ckpt
    nohup bash -c "CUDA_VISIBLE_DEVICES=$GPU_ID $DISTILL_CMD" >> logs/full_pipeline.log 2>&1 &
    sleep 50; continue
  fi
  ckpt_mtime=$(stat -c %Y outputs/checkpoint_latest.json 2>/dev/null || echo 0)
  now=$(date +%s); age_min=$(( (now-ckpt_mtime)/60 ))
  if [ "$age_min" -ge "$STALE_MIN" ]; then
    restarts=$((restarts+1)); [ "$restarts" -gt "$MAX_RESTARTS" ] && { log "✗ 达最大重启,退出"; exit 1; }
    log "⚠ checkpoint停滞 ${age_min}m 判定挂死 PID=$pid merged=$n_merged"
    detect_poison
    log "  kill -9 $pid"; kill -9 "$pid" 2>/dev/null; pkill -9 -f "$PROC_PATTERN" 2>/dev/null
    for i in $(seq 1 20); do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID 2>/dev/null | tr -d ' ')
      [ "${used:-0}" -lt 1000 ] && { log "  GPU${GPU_ID} 释放: ${used}MiB"; break; }
      sleep 5
    done
    rebuild_ckpt; log "  续训重启..."
    nohup bash -c "CUDA_VISIBLE_DEVICES=$GPU_ID $DISTILL_CMD" >> logs/full_pipeline.log 2>&1 &
    sleep 50; continue
  fi
  n_ckpt=$($PYTHON_BIN -c "import json;print(len(json.load(open('outputs/checkpoint_latest.json')).get('processed_ids',[])))" 2>/dev/null || echo "?")
  log "运行中 PID=$pid merged=$n_merged/$TARGET_IMAGES ckpt=$n_ckpt age=${age_min}m restarts=$restarts poison=$(tr '\n' ' ' <"$POISON_FILE" 2>/dev/null)"
  sleep "$CHECK_INTERVAL"
done
