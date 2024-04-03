1、修改listener.py第15行

    cmd = 'CUDA_VISIBLE_DEVICES="GPU" python -m torch.distributed.launch --nproc_per_node GNUM --master_port 2515  trainCAN.py >>./logs/data.log 2>&1 &'

2、视情况修改GPU数量、指定GPU编号

    NUM = 2

    GPU_list=[6,7]

3、激活环境，运行命令

    python listener.py

4、更多相关内容

    见论文或Figure文件夹
    
    论文: Where to Focus: Central Attention-Based Face Forgery Detection
[Link](https://link.springer.com/chapter/10.1007/978-981-99-8469-5_4)

5、公开仓库 [Link](https://github.com/M-SunRise/Code_for_CAN)