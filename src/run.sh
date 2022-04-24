python src/test.py  \
--context_json ./data/context.json \
--test_json ./data/test.json \
--pred_file ./pred.csv \
--QA_model ./ckpt/QA/hfl_chinese-roberta-wwm-ext/ \
--MC_model ./ckpt/MC/best/ \
--tokenizer ./ckpt/tokenizer/hfl_chinese-roberta-wwm-ext/

# --QA_model ./ckpt/QA/hfl_chinese-roberta-wwm-ext/ \
# --MC_model ./ckpt/MC/hfl_chinese-roberta-wwm-ext/ \
