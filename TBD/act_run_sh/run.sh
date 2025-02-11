# srun --partition=AI4Good_S --gres=gpu:1 --job-name=Test --kill-on-bad-exit=1 --async --output=act_save_log/log.out  bash ~/Mgt_detect/run_sh/save_lmhead.sh
# srun --partition=AI4Good_S --gres=gpu:1 --job-name=Test1 --kill-on-bad-exit=1 --async --output=act_save_log/log1.out  bash ~/Mgt_detect/run_sh/save_lmhead1.sh
# srun --partition=AI4Good_S --gres=gpu:1 --job-name=Test2 --kill-on-bad-exit=1 --async --output=act_save_log/log2.out  bash ~/Mgt_detect/run_sh/save_lmhead2.sh

# nohup  bash ~/Mgt_detect/TBD/act_run_sh/save_lmhead.sh > ~/Mgt_detect/act_save_log/log.log 2>&1 &
# nohup  bash ~/Mgt_detect/TBD/act_run_sh/save_lmhead1.sh > ~/Mgt_detect/act_save_log/log1.log 2>&1 &
# nohup  bash ~/Mgt_detect/TBD/act_run_sh/save_lmhead4.sh > ~/Mgt_detect/act_save_log/log4.log 2>&1 &
# nohup  bash ~/Mgt_detect/TBD/act_run_sh/save_lmhead5.sh > ~/Mgt_detect/act_save_log/log5.log 2>&1 &

#----------------------MAGE----------------------#
nohup  bash ~/Mgt_detect/TBD/act_run_sh/save_lmhead6.sh > ~/Mgt_detect/act_save_log/log6.log 2>&1 &
nohup  bash ~/Mgt_detect/TBD/act_run_sh/save_lmhead11.sh > ~/Mgt_detect/act_save_log/log11.log 2>&1 &
nohup  bash ~/Mgt_detect/TBD/act_run_sh/save_lmhead12.sh > ~/Mgt_detect/act_save_log/log12.log 2>&1 &

#-------------------DEC---------------------#
nohup  bash ~/Mgt_detect/TBD/act_run_sh/save_lmhead6_dec.sh > ~/Mgt_detect/act_save_log/log6_dec.log 2>&1 &
nohup  bash ~/Mgt_detect/TBD/act_run_sh/save_lmhead11_dec.sh > ~/Mgt_detect/act_save_log/log11_dec.log 2>&1 &
nohup  bash ~/Mgt_detect/TBD/act_run_sh/save_lmhead12_dec.sh > ~/Mgt_detect/act_save_log/log12_dec.log 2>&1 &



