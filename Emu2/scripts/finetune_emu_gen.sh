source activate emnu
python -m accelerate.commands.launch --mixed_precision bf16 \
    emu_gen_finetune.py \
    --use_quant_model \
    --output_dir results/Emu2-Gen \
    --val 