
cd MERGETUNE/mergetune
bash scripts/coop/base2new_train.sh caltech101 1
bash scripts/coop/base2new_test.sh caltech101 1
bash scripts/coop/base2base_test.sh caltech101 1

bash scripts/coop_LMC/base2new_train_coop.sh caltech101 10.0 cosine True 1.0 1 vit_b16_ep100_ctxv1
bash scripts/coop_LMC/base2new_test_coop.sh caltech101 10.0 cosine True 1.0 1 vit_b16_ep100_ctxv1
bash scripts/coop_LMC/base2base_test_coop.sh caltech101 10.0 cosine True 1.0 1 vit_b16_ep100_ctxv1


bash scripts/kgcoop/base2new_train.sh caltech101 8.0 1
bash scripts/kgcoop_LMC/base2new_train_kgcoop.sh caltech101 10.0 1.0 1 vit_b16_ep100_ctxv1


bash scripts/mma/base2new_train.sh caltech101 1 vit_b16_ep5
bash scripts/mma_LMC/base2new_train_mma.sh caltech101 10.0 vit_b16_ep5 1

# get pretrained student model from promptkd
bash scripts/promptkd_LMC/base2new_train_promptkd.sh caltech101 1 10.0

bash scripts/coop/base2new_train_siglip2.sh caltech101
bash scripts/coop_LMC/base2new_train_coop_siglip2.sh caltech101 10.0 cosine True 1.0 siglip2_b16_ep100_ctxv1

