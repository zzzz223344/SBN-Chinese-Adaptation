这里是关于为了中文适配我主要改了什么以及可能存在的问题，关于使用之前你该做什么准备工作，**请读README3**

**数据加载的兼容性和鲁棒性**

**模型训练和评估中的维度匹配问题**

*训练数据截断（lines = lines\[:2000]）：*

*这会大幅减少训练数据量，严重影响模型性能。如果是为了快速调试可以保留，但正式训练必须移除（把那一行代码直接注释掉）。*

*直接使用测试集进行早停：*

*现版本在训练时直接评估测试集，并基于测试集 F1 保存模型。这会导致对测试集的过拟合，无法真实反映模型泛化能力。应改为使用验证集（devset）。*

*情感不一致的处理：*

*现版本静默处理情感不一致，虽然提高了鲁棒性，但可能掩盖数据质量问题。*

***data\_BIO\_loader.py —— 有重要修改***

(1) convert\_examples\_to\_features 函数中的情感处理逻辑（135行)

原版本：



if aspect\_span not in aspect:

&#x20;   aspect\[aspect\_span] = sentiment

&#x20;   opinion\[aspect\_span] = \[(opinion\_span, sentiment)]

else:

&#x20;   assert aspect\[aspect\_span] == sentiment    # 强制要求情感一致

&#x20;   opinion\[aspect\_span].append((opinion\_span, sentiment))

如果同一个 Aspect 对应多个 Opinion 且情感不一致，会直接触发 AssertionError。



现版本：



if aspect\_span not in aspect:

&#x20;   aspect\[aspect\_span] = sentiment

&#x20;   opinion\[aspect\_span] = \[(opinion\_span, sentiment)]

else:

&#x20;   if aspect\_span not in aspect:

&#x20;       aspect\[aspect\_span] = sentiment

&#x20;   else:

&#x20;       pass

&#x20;   # 如果情感不同，保留其中一个（可以根据需要修改）

&#x20;   # 这里保留第一个出现的情感

&#x20;   opinion\[aspect\_span].append((opinion\_span, sentiment))

现版本移除了 assert，改为静默处理，当情感不一致时，保留第一次出现的情感，后续不同的情感被忽略（但 Opinion 仍被记录）。这增强了代码对噪声数据的鲁棒性，但可能掩盖数据质量问题。



**对应的反向映射部分 (opinion\_reverse 和 aspect\_reverse) 也做了类似修改，将 assert 替换为条件判断**。



(2) 后续的 assert 也被移除或弱化（226行）

在构建 spans\_opinion\_label 和 reverse\_aspect\_label 时：



原版本：assert len(set(sentiment\_opinion)) == 1



现版本：如果情感不一致，取第一个情感，并添加注释说明。



(3) load\_data 函数增加训练集截断（320行）



if if\_train:

&#x20;   lines = lines\[:2000]   # 新增：只取前2000条作为训练集

&#x20;   random.seed(args.RANDOM\_SEED)

&#x20;   random.shuffle(lines)

训练数据被限制为前 2000 条，这不是论文中的设置，而是为了快速调试或小规模实验。如果用于正式实验，需要移除或修改这一行。



(4) load\_data\_instances\_txt 增加更多情感标签映射（330行）

原版本：sentiment2sentiment = {'NEG': 'negative', 'POS': 'positive', 'NEU': 'neutral'}



现版本：



sentiment2sentiment = {'NEG': 'negative', 'POS': 'positive', 'NEU': 'neutral',

&#x20;                      'Neutral': 'neutral', 'Negative': 'negative', 'Positive': 'positive'}

支持更多格式（如首字母大写），提高兼容性。





***Metric.py —— 有重要修改***

(1) find\_aspect\_sentiment 和 find\_opinion\_sentiment 方法（336行）

原版本：



def find\_aspect\_sentiment(self, sentence\_index, bert\_spans, span, ...):

&#x20;   bert\_span\_index = \[i for i,x in enumerate(bert\_spans) if span\[1] == x\[0] and span\[2] == x\[1]]

&#x20;   assert len(bert\_span\_index) == 1  # 如果找不到会直接报错

现版本：



def find\_aspect\_sentiment(self, sentence\_index, bert\_spans, span, ...):

&#x20;   start, end = int(span\[1]), int(span\[2])   # 显式类型转换

&#x20;   bert\_span\_index = \[i for i, x in enumerate(bert\_spans) if x\[0] == start and x\[1] == end]

&#x20;   if len(bert\_span\_index) == 0:

&#x20;       bert\_span\_index = \[i for i, x in enumerate(bert\_spans) if int(x\[0]) == start and int(x\[1]) == end]  # 再次尝试

&#x20;   if len(bert\_span\_index) == 0:

&#x20;       print(f"Warning: cannot find aspect span ({start}, {end}) in bert\_spans for sentence {sentence\_index}")

&#x20;       return 'none', 0.0   # 返回默认值，而不是崩溃





增加了 int() 类型转换，避免因数据类型不匹配（如 Tensor vs int）导致的匹配失败。



增加了二次匹配尝试。



增加了异常处理，当找不到对应 Span 时打印警告并返回默认值，而不是让程序崩溃。



**同样的修改也应用到了 find\_opinion\_sentiment 方法。（383行）**



(2) find\_pred\_triples 中增加有效性过滤（461行）

if aspect\_sentiment == 'none':

&#x20;   continue   # 新增：如果 Aspect 被判为 'none'，跳过该三元组

过滤掉 Step 1 中预测为无效的 Aspect，避免进入 Step 2 造成错误预测。



***model.py —— 有重要修改***

(1) stage\_2\_features\_generation 函数 —— **大幅修改（12行）**

**这是两个版本差异最大的地方**。第二个版本重写了该函数，主要解决：



维度不匹配问题：确保 spans\_embedding 始终为 3 维 (batch, num\_spans, hidden\_size)。



span\_index 处理：当 torch.nonzero 返回标量或空张量时的处理。



拼接时的维度统一：在 torch.cat 之前强制所有张量维度一致。



关键修改点：



处理 span\_index 可能是标量的情况

span\_index = torch.nonzero((span\_index\_end > -1), as\_tuple=False).squeeze(0)

if span\_index.dim() == 0:

&#x20;   span\_index = span\_index.unsqueeze(0)



新增：只取第一个匹配的 span（避免多个匹配）

if span\_index.numel() > 1:

&#x20;   span\_index = span\_index\[0].unsqueeze(0)

在拼接前，增加了大量的维度检查和 view 操作，确保所有张量维度一致后再拼接。





修复了第一个版本中因维度不匹配导致的 RuntimeError 或 AssertionError。



提高了代码的鲁棒性，但逻辑复杂度增加。



(2) Dim\_Four\_Block 中的 spans\_embedding 维度（267行）

原版本：spans\_embedding = span\_sum\_embdding.squeeze()



现版本：spans\_embedding = span\_sum\_embdding.squeeze(2)



显式指定压缩维度，避免误压缩其他维度。





***run.py***

(1) 数据集选择增加新选项（490行）



parser.add\_argument("--dataset", default="lap14", type=str,

&#x20;                   choices=\["lap14", "res14", "res15", "res16", "405.Chinese\_Zhang"],

&#x20;                   help="specify the dataset")

增加了对中文数据集 405.Chinese\_Zhang 的支持。



(2) 评估时使用测试集而非验证集（374行）

在训练循环中：



原版本：评估 devset



现版本：评估 testset





aspect\_result, opinion\_result, ... = eval(..., devset, args)

aspect\_result, opinion\_result, ... = eval(..., testset, args)   # 直接使用测试集

注意：这相当于在训练过程中直接使用测试集进行早停和模型选择，这在学术上是不规范的（可能导致过拟合测试集）。正式实验应修改。



模型加载时硬编码 F1 值（441行）

第一个版本：model\_path = args.model\_dir + args.dataset + '\_' + str(0.6265060240963854) + '.pt'



第二个版本：model\_path = args.model\_dir + args.dataset + '\_' + str(0.6111111111111112) + '.pt'

