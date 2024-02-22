source activate emnu
python -m accelerate.commands.launch --mixed_precision bf16 \
    emu_gen_finetune.py \
    --model_name Emu2 \
    --use_quant_model \
    --val \
    --output_dir results/Emu2