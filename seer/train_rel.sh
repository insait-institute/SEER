echo $HOSTNAME
git checkout tmp_prop_merge3
git pull
python3 main.py \
-btr $((${1}-1))_1_2 \
-bte $((${1}-1))_1 \
--prop_mode max \
--acc_grad 10 \
--learning_rate 1e-4 \
--data_path ../data \
--res_path ../models \
--attack_cfg modern \
--print_interval 30 \
--num_epochs 1000 \
--epoch_steps 1000 \
--test_interval 10 \
--big_test_interval 100 \
--act ReLU \
--public_labels True \
--par_sel_size 8400 \
--par_sel_frac 0.001 \
--mid_rep_frac 1.0 \
--num_test_img 20 ${@:2}
