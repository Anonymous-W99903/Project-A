The repo is for Can Wireless Sensing Models Achieve Both Universality and Lightweight Design?

The dataset is 20181130.zip in Widar 3.0

### Training

---

For Teacher training, you can use command like this, Change `train_domain`，`test_domain` to get teacher on different source domains, and change `domain name` to use different domain. 

```
python3 widar3_train_teacher.py --save_root "../widar3/val_location/T2_r1"  --data_dir "../CSI_20181130/dfs"    --domain_name "loc"   --train_domain 2   --test_domain 2   --num_class 6    --print_freq 5  --batch_size 64 --num_workers 0  --epochs 200  --lr 0.01   --cuda 1 --gpu_id 0 --weight_decay 0.001 --adjust_lr 'cosine' --warmup 20
```

---

For Baseline, use command like this:

```
python3 widar3_train_baseline.py --save_root "../widar3/val_location/baseline1_r1"  --data_dir "../CSI_20181130/dfs"    --domain_name "loc"   --train_domain 2 3 4 5   --test_domain 1   --num_class 6    --print_freq 10  --batch_size 64 --num_workers 0  --epochs 200  --lr 0.01   --cuda 1 --gpu_id 0 --weight_decay 0.001 --adjust_lr 'cosine' --warmup 20
```

---

For Student, use the following command, make sure that you have prepared all the Teacher models.

`t_dir` refers the path for Teacher models.

If you want train Fake Student model, modify the `train domain` according to your settings.

```
python3 widar3_train_student.py --opt Adam --save_root "../widar3/loc/v2_2_T1_S_r1" --data_dir "../CSI_20181130/dfs" --domain_name "loc" \
--train_domain 2 3 4 5   --test_domain 1   --num_class 6    --print_freq 5  --batch_size 64 --epochs 200  --lr 0.0002 --gpu_id 0 --adjust_lr linear \
--warmup 0 --lambda_kd 0 1.0 1.0 1.0 1.0 --lambda_cls 1  --T 4 --baseline_path  "../widar3/val_location/baseline1/checkpoint_current.pth.tar" \
--t_dir "../widar3/val_location/" --lambda_kd_baseline 1
```

----

To get the pruning structure:

Change `pruning_amount` for different amount of pruned structure. 

```
python3 widar3_train_student_pruning.py --opt Adam --save_root "../widar3/val_location/v5_2_T1_S_p0.5_r1" --data_dir "../CSI_20181130/dfs" --domain_name "loc" \
--train_domain 2 3 4 5   --test_domain 1   --num_class 6    --print_freq 5  --batch_size 64 --epochs 200  --lr 0.001 --gpu_id 0 --adjust_lr linear \
--warmup 0 --lambda_kd 0 1 1 1 1 --lambda_cls 1  --T 4 --baseline_path  "../widar3/val_location/baseline1/checkpoint_current.pth.tar" \
--t_dir "../widar3/val_location/" --prune_amount 0.5
```

If you have already get the pruned model, use this command instead to extract the model structure for training. It will use the network structure but discard the parameters in it.

```
python3 widar3_train_student_pruned.py --opt Adam --save_root "../widar3/loc/v5pruned1_T2_fs_0.8_r1" --data_dir "../CSI_20181130/dfs" --domain_name "loc" \
--train_domain 1 3 4 5   --test_domain 2   --num_class 6    --print_freq 5  --batch_size 64 --epochs 200  --lr 0.001 --gpu_id 0 --adjust_lr linear \
--warmup 0 --lambda_kd 1 0 1 1 1 --lambda_cls 1  --T 4 --baseline_path  "../widar3/val_location/baseline2/checkpoint_current.pth.tar" \
--t_dir "../widar3/val_location/" --pruned_model_path "../widar3/val_location/v5_2_Fake_T1_S_p0.8_r1/pruned_model.pth.tar" --lambda_kd_baseline 1
```

### Similarity

If you want to quantify Teacher Model contributions, you can get their similarity to the target domain, and then calculate the `lambda_kd` in the above section.

``` 
python3 widar3_extract_features.py --save_root "../widar3/val_ori/EF_T3_r1/"  --model_path "../widar3/val_location/T2_r1/model_best.pth.tar" --data_dir "../CSI_20181130/dfs"  \
--domain_name "loc"  --domain_cmp 1,2  --num_class 6    \
--print_freq 5  --batch_size 64 --num_workers 0  --epochs 200  --lr 0.01   --cuda 1 --gpu_id 0 --weight_decay 0.001 --adjust_lr 'cosine' --warmup 20
```

`model_path` is the model you want used to extract features on the `domian_cmp`.
