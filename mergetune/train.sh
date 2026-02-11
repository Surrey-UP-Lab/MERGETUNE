
cd MERGETUNE/mergetune
bash scripts/coop/base2new_train.sh caltech101 1
bash scripts/coop/base2new_test.sh caltech101 1
bash scripts/coop/base2base_test.sh caltech101 1

bash scripts/coop_LMC/base2new_train_coop.sh caltech101 10.0 cosine True 1.0 1 vit_b16_ep100_ctxv1
bash scripts/coop_LMC/base2new_test_coop.sh caltech101 10.0 cosine True 1.0 1 vit_b16_ep100_ctxv1
bash scripts/coop_LMC/base2base_test_coop.sh caltech101 10.0 cosine True 1.0 1 vit_b16_ep100_ctxv1
