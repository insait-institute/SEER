echo $HOSTNAME
git checkout tmp_prop_merge3
git pull
mkdir -p images
python3 main.py \
-btr $((${1}-1))_1_2 \
-bte $((${1}-1))_1 \
--prop_mode max \
--task end2end \
--acc_grad 10 \
--learning_rate 1e-4 \
--data_path ../data \
--res_path ../models \
--attack_cfg modern \
--print_interval 30 \
--test_interval 50 \
--act ReLU \
--public_labels True \
--par_sel_size 8400 \
--par_sel_frac 0.001 \
--mid_rep_frac 1.0 \
--num_test_img 1000 ${@:2}
