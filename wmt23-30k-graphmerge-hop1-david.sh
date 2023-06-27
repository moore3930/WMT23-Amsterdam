#! /bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --nodelist=ilps-cn117
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-10
##SBATCH --begin=now+1minute
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=d.wu@uva.nl


#SBATCH -o xxx
#SBATCH -e xxx


export PATH=/home/diwu/anaconda3/bin:$PATH
source activate py38cuda11
export CUDA_HOME="/usr/local/cuda-11.0"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="/home/diwu/cudalibs:/usr/lib64/nvidia:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

DATA_DIR=/ivi/ilps/projects/ltl-mt/wmt23-wiki/data_bin_30k
CHECKPOINT_DIR=xxx

fairseq-train ${DATA_DIR} \
    --save-dir ${CHECKPOINT_DIR} \
    --langs en,he \
    --lang-pairs en-he,he-en \
    --arch transformer_iwslt_de_en_id \
    --share-decoder-input-output-embed \
    --dropout 0.1 \
    --task translation_multi_simple_epoch \
    --sampling-method temperature \
    --sampling-temperature 1.0 \
    --encoder-langtok src \
    --decoder-langtok \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --weight-decay 0.0 \
    --max-tokens 16384 --update-freq 4 --patience 20 --max-update 200000 \
    --save-interval-updates 1000 --keep-interval-updates 5 \
    --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 20 \
    --skip-invalid-size-inputs-valid-test \
    --user-dir YOUR_DIR_TO_FAIRSEQ/fairseq/graphsage_v3_sparse \
    --graph-path ${DATA_DIR}/alignment_matrix.npz \
    --tie-graph-proj \
    --hop-num 1 \
    --fp16

# eval
PAIRS=('en-he' 'he-en')
for i in "${!PAIRS[@]}"; do
    PAIR=${PAIRS[i]}
    SRC=${PAIR%-*}
    TGT=${PAIR#*-}
    fairseq-generate ${DATA_DIR} \
        --task translation_multi_simple_epoch \
        --langs en,he \
        --lang-pairs $PAIR \
        --source-lang $SRC \
        --target-lang $TGT \
        --sacrebleu \
        --remove-bpe 'sentencepiece' \
        --arch transformer_iwslt_de_en_id \
        --path ${CHECKPOINT_DIR}/checkpoint_best.pt \
        --sampling-method temperature \
        --skip-invalid-size-inputs-valid-test \
        --encoder-langtok src \
        --decoder-langtok \
        --gen-subset test \
        --share-decoder-input-output-embed \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --max-tokens 10000 \
        --beam 5 \
        --seed 222 \
        --results-path ${CHECKPOINT_DIR}/${SRC}-${TGT} \
        --user-dir ${YOUR_DIR_TO_FAIRSEQ}/fairseq/graphsage_v3_sparse \
        --graph-path ${DATA_DIR}/alignment_matrix.npz \
        --graph-dropout 0.0 \
        --tie-graph-proj \
        --hop-num 1 \
        --fp16
done

