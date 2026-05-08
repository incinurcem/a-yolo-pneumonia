#!/usr/bin/env bash
# A-YOLO Ablation: 3 varyantı sırayla eğitir + değerlendirir.
# Süre tahmini: ~6-9 saat (50 epoch × 3 varyant, A100 üzerinde).

set -e

ROOT="/content/drive/MyDrive/Spring Semester/deep learning project"
DATA="$ROOT/diffusion_guided_detr_data"
SRC="$ROOT/src/a_yolo"
OUT="$ROOT/outputs/a_yolo/ablation"

EPOCHS=20
BATCH=32
LR=1.5e-4
IMG=224

mkdir -p "$OUT"

for STRAT in none random gaussian; do
    NAME=$STRAT

    # Alpha: SSL kapalıysa 0, açıksa 0.2
    if [ "$STRAT" = "none" ]; then
        ALPHA=0.0
    else
        ALPHA=0.2
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  ABLATION: mask=$STRAT  alpha=$ALPHA  ($EPOCHS epoch)"
    echo "═══════════════════════════════════════════════════════"

    python "$SRC/train2.py" \
        --img_dir    "$DATA/images" \
        --train_csv  "$DATA/metadata/train_master.csv" \
        --val_csv    "$DATA/metadata/val_master.csv" \
        --output_dir "$OUT/$NAME" \
        --epochs $EPOCHS \
        --batch_size $BATCH \
        --lr $LR \
        --img_size $IMG \
        --mask_strategy $STRAT \
        --alpha $ALPHA \
        --patience 8 \
        --num_workers 8

    echo ""
    echo "  ✅ Train bitti — şimdi evaluation..."

    python "$SRC/evaluate.py" \
        --model_path "$OUT/$NAME/checkpoints/best_ayolo.pth" \
        --img_dir    "$DATA/images" \
        --csv_path   "$ROOT/../dataset/processed_metadata/rsna_master_metadata.csv" \
        --output_dir "$OUT/$NAME/eval" \
        --img_size   $IMG \
        --batch_size 512
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "✅ Tüm ablation varyantları tamamlandı."
echo "   Tablo üretmek için:"
echo "   python scripts/aggregate_ablation.py"
echo "═══════════════════════════════════════════════════════"