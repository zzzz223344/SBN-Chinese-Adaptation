**这里是你想要使用他的话该知道的事，你可能会遇到各种各样的路径和文件名问题或奇妙小问题（显存不够，甚至因为你用的gpu和python太新版本不匹配什么的），自己运行时看报错找一下吧，实话说我也不好说在哪，想复现的话大概会花你一到两个晚上**

**我在autodl上租的服务器https://www.autodl.com/**

**GPU**

**RTX 3080 Ti(12GB) \* 1**

**CPU**

**12 vCPU Intel(R) Xeon(R) Silver 4214R CPU @ 2.40GHz**

**内存**

**90GB**

**硬盘**

**系统盘:30 GB**

**数据盘:50GB**



**PyTorch  1.10.0**

**Python  3.8(ubuntu20.04)**

**CUDA  11.3**



**一、 绝对必须修改的代码位置**

1\. 指定显卡（GPU）

文件：run.py



原代码：os.environ\['CUDA\_VISIBLE\_DEVICES'] = '5'



修改建议：改成你服务器上空闲的 GPU 编号（例如 '0'），如果只有一张卡，直接删除这行代码或改为 '0'。



2\. 预训练 BERT 模型路径

文件：run.py



原代码：default="pretrained\_models/bert-base-uncased"



修改建议：请下载您想要的模型放到对应文件夹，并修改路径（我用的bert-base-chinese）

3\. 数据集路径

文件：run.py







原代码：default="./datasets/ASTE-Data-V2-EMNLP2020/"



修改建议：必须指向你下载数据集的实际存放路径（官方仓库在 https://github.com/xuuuluuu/SemEval-Triplet-data）我用的405.Chinese\_Zhang（https://github.com/yangheng95/ABSADatasets）。



4\. 训练集截断

文件：data\_BIO\_loader.py



原代码：lines = lines\[:2000]



修改建议：请务必删除这一行代码（或注释掉）。这是用于快速调试的代码，如果不删除，程序只会读取前 2000 条数据作为训练集



5\. 验证集与测试集的混用

文件：run.py





原代码：



\# aspect\_result,... = eval(..., devset, args)  # 被注释掉

aspect\_result,... = eval(..., testset, args)   # 直接使用测试集早停

修改建议：强烈建议将下面的 testset 注释掉，取消注释 devset。否则你是在拿测试集做验证选模型，这会导致论文中的指标虚高，无法反应真实泛化能力。



6\. 模型加载路径（如果直接运行测试）

文件：run.py



原代码：model\_path = args.model\_dir + args.dataset + '\_' + str(0.6111111111111112) + '.pt'



修改建议：



如果使用 --mode train：这一行不会被执行，可以忽略。



如果使用 --mode test：必须将这里的 0.611... 换成你自己训练保存的 .pt 模型文件名，或者将整个 model\_path 改成你指定的路径。



**二、可供你自由调整的可选参数**

不需要改代码，直接在命令行运行 python run.py 时加上这些参数即可调整模型行为。



1\. 运行模式

\--mode {train, test}：训练新模型或加载已有模型进行测试。



2\. 数据集选择

\--dataset {lap14, res14, res15, res16, 405.Chinese\_Zhang}：选择不同的数据集。



3\. 模型结构核心参数

\--span\_generation {Max, Average, Start\_end, CNN, ATT}：跨度表示的生成方式。推荐使用 Max（论文默认）。



\--block\_num：跨注意力模块的堆叠层数。默认为 1。



\--max\_span\_length：最大跨度长度（词数），默认 8。



4\. 训练超参数

\--train\_batch\_size：训练批次大小，默认 16（若显存不足可调小，如 8）。



\--epochs：训练轮数，默认 130。



\--learning\_rate：BERT 层的学习率（默认 1e-5）。



\--task\_learning\_rate：下游任务层（Step 1 \& 2）的学习率（默认 1e-4）。



5\. 特殊功能开关

\--kl\_loss：是否启用跨度表示分离损失（用于区分相似跨度）。默认为 True（开启）。



\--kl\_loss\_mode {KLLoss, JSLoss, EMLoss, CSLoss}：分离损失的具体算法。



\--Filter\_Strategy：是否在预测后使用启发式策略去除冲突的三元组（如重叠实体），默认为 True（开启，建议保留）。



**下面我给你包括所有可选参数的运行命令（你该认真看一下）**



python run.py \\

&#x20; --mode train \\                           # train / test

&#x20; --model\_dir savemodel/ \\                 # 模型保存目录

&#x20; --device cuda \\                          # cuda / cpu

&#x20; --init\_model pretrained\_models/bert-base-uncased \\   # BERT 路径

&#x20; --init\_vocab pretrained\_models/bert-base-uncased \\   # 词汇表路径

&#x20; --bert\_feature\_dim 768 \\                 # BERT 输出维度（固定）

&#x20; --do\_lower\_case \\                        # 是否小写（uncased 模型默认）

&#x20; --max\_seq\_length 100 \\                   # 最大输入长度

&#x20; --drop\_out 0.1 \\                         # Dropout 比例

&#x20; --max\_span\_length 8 \\                    # 最大跨度长度（词数）

&#x20; --embedding\_dim4width 200 \\              # 跨度宽度嵌入维度

&#x20; --task\_learning\_rate 1e-4 \\              # 任务层学习率

&#x20; --learning\_rate 1e-5 \\                   # BERT 层学习率

&#x20; --accumulation\_steps 1 \\                 # 梯度累积步数

&#x20; --muti\_gpu \\                             # 是否多卡（不加则单卡）

&#x20; --epochs 130 \\                           # 训练轮数

&#x20; --train\_batch\_size 16 \\                  # 批次大小

&#x20; --RANDOM\_SEED 2022 \\                     # 随机种子

&#x20; --dataset\_path ./datasets/ASTE-Data-V2-EMNLP2020/ \\   # 数据集根目录

&#x20; --dataset lap14 \\                        # 数据集名称：lap14/res14/res15/res16/405.Chinese\_Zhang

&#x20; --Only\_token\_head \\                      # 仅使用子词首 token（不加则用全部）

&#x20; --span\_generation Max \\                  # 跨度生成方式：Max/Average/Start\_end/CNN/ATT

&#x20; --ATT\_SPAN\_block\_num 1 \\                 # ATT 模式下的块数

&#x20; --kl\_loss \\                              # 是否启用 KL 分离损失（不加则关闭）

&#x20; --kl\_loss\_weight 0.5 \\                   # KL 损失权重

&#x20; --kl\_loss\_mode KLLoss \\                  # 分离损失类型：KLLoss/JSLoss/EMLoss/CSLoss

&#x20; --Filter\_Strategy \\                      # 是否启用后处理筛选冲突三元组（建议开启）

&#x20; --related\_span\_underline \\               # 是否使用相关跨度注意力（默认关闭）

&#x20; --related\_span\_block\_num 1 \\             # 相关跨度注意力块数

&#x20; --block\_num 1 \\                          # 跨注意力块数（Step 2）

&#x20; --output\_path triples.json \\             # 输出文件（用于保存详细结果）

&#x20; --order\_input \\                          # 是否按长度顺序生成跨度（建议开启）

&#x20; --random\_shuffle 0 \\                     # 随机打乱跨度顺序（0 表示不打乱）

&#x20; --model\_para\_test \\                      # 是否测试模型参数量（默认关闭）

&#x20; --whether\_warm\_up \\                      # 是否使用 Warmup（默认关闭）让学习率由小变大

&#x20; --warm\_up 0.1                            # Warmup 比例（若开启）



显存不足：减小 --train\_batch\_size 或 --max\_seq\_length

