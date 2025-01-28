#run with bash shell
#!/bin/bash

bash ./inference_hf.sh arc_c
bash ~/send_noti.sh "arc_c"
bash ./inference_hf.sh arc_e
bash ~/send_noti.sh "arc_e"
bash ./inference_hf.sh copa
bash ~/send_noti.sh "copa"
bash ./inference_hf.sh mrpc
bash ~/send_noti.sh "mrpc"
bash ./inference_hf.sh mnli
bash ~/send_noti.sh "mnli_m" #change to mnli_mm or mnli_m
bash ./inference_hf.sh openbookqa
bash ~/send_noti.sh "openbookqa"
bash ./inference_hf.sh squad_v1
bash ~/send_noti.sh "squad_v1"
bash ~/send_noti.sh "done"
