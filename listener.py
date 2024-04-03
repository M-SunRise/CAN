# author: muzhan
import os
import time

GPU_list=[6,7]
NUM = 2

def gpu_info(gpu_index=2, power_limit=0, memory_limit=0):
    info = os.popen('nvidia-smi|grep %').read().split('\n')[gpu_index].split('|')
    power = int(info[1].split()[-3][:-1])
    memory = int(info[2].split('/')[0].strip()[:-3])
    return (power < power_limit) and ( memory < memory_limit), power, memory

def narrow_setup_available(interval=2):
    cmd = 'CUDA_VISIBLE_DEVICES="GPU" python -m torch.distributed.launch --nproc_per_node GNUM --master_port 2515  trainCAN.py >>./logs/data.log 2>&1 &'

    while(True):
        num = 0
        avail_list=[]
        for index in range(len(GPU_list)):
            flag = gpu_info(GPU_list[index], 35, 700)[0]
            if flag:
                num += 1
                avail_list.append(GPU_list[index])
        if num >= NUM:
            gpu_ = ''
            for idx in range(NUM):
                if gpu_ == '':
                    gpu_=str(avail_list[idx])
                else:
                    gpu_ += ','+str(avail_list[idx])
            cmd = cmd.replace('GPU', gpu_)
            cmd = cmd.replace('GNUM', str(NUM))
            print('\n' + cmd)
            os.system(cmd)
            break
        else:
            time.sleep(0)

if __name__ == '__main__':
    narrow_setup_available()