# CEUA：中文情感理解与应用基准

> **CEUA** 是一个开放的中文情感智能基准，用于评估大型语言模型的两项互补能力：**情感理解（Emotional Understanding，EU）******和******情感应用（Emotional Application，EA）**。

**CEUA** = **C**hinese **E**motional **U**nderstanding and **A**pplication
即：中文情感理解与应用。

**412 道中文问题 · 206 个底层场景 · EU + EA · 官方评测器 · 无第三方依赖**

---

## 一、概述

大型语言模型在语言理解和推理方面已经取得了显著进展，但如何评估模型在**情感丰富的场景中理解情绪并作出恰当回应**的能力，仍然具有挑战性。

CEUA 设计用于评估两项互补能力：

* **情感理解（Emotional Understanding，EU）**：识别一个场景中明确表达或隐含的主要情绪。
* **情感应用（Emotional Application，EA）**：在给定场景中选择恰当的共情回应或行动。

该基准包含 **412 道问题，基于 206 个底层场景构建**。每个场景都会分别从“理解”和“应用”两个角度进行评估。

| 模块     | 描述   |     问题数 |
| ------ | ---- | ------: |
| **EU** | 情感理解 |     206 |
| **EA** | 情感应用 |     206 |
| **总计** |      | **412** |

### 问题类型

* **379 × ****`single_choice`****（单选题）**
* **33 × ****`two_choice`****（双选题）**
* 语言：`zh`

评测流程刻意保持简单：

```text
数据 → 模型推理 → 预测 → 评估 → 得分
```

---

# 二、基准测试结果

我们在 CEUA 上对多种中文大语言模型进行了评估。总体来看，当前主流模型已经具备较强的中文情感理解能力，但在将情感理解进一步应用于自然的人际互动时，仍存在明显挑战。

模型在 **情感理解（EU）** 任务上的表现整体较好，能够较为稳定地从中文场景中识别和判断主要情绪。相比之下，**情感应用（EA）** 的难度更高，模型在选择合适、自然且符合具体互动情境的回应时，表现会有所下降。

这一结果表明，当前大语言模型已经具备较好的基础情感推理能力，但**理解情绪与自然地回应情绪之间仍存在一定差距**。

## 为什么 EA 更难？

一些 CEUA 场景刻意使用了**模糊的情感信号**。模型不能仅仅依靠明确的情绪词，而需要结合上下文、人际关系、语气以及隐含线索来推断潜在情绪。

EA 模块则进一步要求模型将这种情感理解应用到实际互动中。

其中包含一定数量的**双选题**，用于区分两个都具有合理性的回答，但它们在自然程度、表达风格或与具体场景的匹配程度上有所不同。

这一设计源于早期实验中的一个观察：

> 模型可能通过逻辑推理找到理论上正确的回应，但与人类自然交流的方式相比，它最终选择的回答仍可能显得过于僵硬或过于遵循规则。

因此，CEUA 不仅希望评估模型：

> **是否知道什么是正确的回应**

也希望进一步评估：

> **是否知道人类在真实互动中会如何自然地回应。**

双选题进一步增加了 EA 的难度，也使其成为当前模型更具挑战性的部分。

总体而言，CEUA 的结果表明，当前大语言模型在基础情感识别和推理方面已经取得了较好的表现，但在以下方面仍有进一步提升空间：

* 细微的情绪差异识别
* 对隐含情感信号的理解
* 回应的自然程度
* 情感理解向实际互动行为的迁移
* 在多个合理回应之间进行选择


---

# 三、任务

## 3.1 情感理解（EU）

EU 用于评估模型是否能够正确识别给定场景中的主要情绪。

典型问题格式如下：

```text
问题
├── 场景
└── 选项
    ├── A
    ├── B
    ├── C
    └── D
```

模型输出一个答案选项。

---

## 3.2 情感应用（EA）

EA 用于评估模型能否根据一个场景选择恰当的情感回应或行动。

EA 不再局限于简单的情绪分类，而是进一步测试模型能否将情感理解**应用到实际互动中**。

部分 EA 问题采用双选评估：

```text
A + B
```

其中两个被选择的回答都被认为是正确的。

所选择答案的顺序不影响结果：

```text
CD == DC
```

---

# 四、数据集

数据集组织结构如下：

```text
data/
├── eu.jsonl    # 206 道 EU 问题
└── ea.jsonl    # 206 道 EA 问题
```

每条记录包含：

```text
qid
language
module
question_type
question
subject
choices
answer
```

EA 记录还额外包含：

```text
scenario
```

## 问题类型

### 单选题

```text
A
B
C
D
```

### 双选题

以下表示方式都会被规范化为相同的答案：

```text
CD
DC
C,D
C D
```

双选题使用**集合语义（set semantics）**：

```text
CD == DC
```

也就是说，答案的排列顺序不重要。

## 数据集构建（Dataset Construction）

CEUA 的场景为**原创构造的合成（synthetic / fictional）情境**，取材于日常人际关系与情绪主题（家庭、朋友、亲密关系、学习、职场、社会交往、成就与失败、道德与责任、失去与变化、压力与不确定性）。所有情境都围绕虚构人物（如"阿伟""阿丽""小方"）构建，**不以真实人物作为数据主体**，也不包含真实个人的姓名、联系方式、账号等个人信息。

构建过程使用了大型语言模型（LLM）作为**辅助工具**，用于生成候选情境、候选回应与措辞打磨。**LLM 生成的内容不会被自动当作金标准**：所有题目与答案在进入数据集之前，都经过人工整理、筛选、修改和验证。因此发布出来的标签应被理解为"经过人工审核、同时受到 LLM 辅助构建过程影响的基准标注"，而不是未经核验的模型自动输出。

关于 LLM 具体参与了哪些环节、使用了哪些模型，当前版本的记录仍不够系统，后续版本会补充说明。

## 标注与校验（Annotation / Validation）

* **人工审核重点**：情境是否合理、中文表达是否自然、在人际上是否得体、情境与选项是否对应、预期答案是否站得住脚。
* 部分 EA 选项为更贴近日常中文表达而被改写。
* **自动校验**：数据完整性由 `src/validate_dataset.py` 检查（JSONL 格式、Schema、字段类型与取值、qid 唯一性与连续性、空字段、额外字段等）。该脚本**只做检查，不修改数据**。
* **已知短板**：当前版本**未提供完整的标注者间一致性（inter-annotator agreement）统计**，人工审核规模有限。这是一个明确的局限，而不是可以一带而过的细节。

---

# 五、重要的数据集结构

EU 和 EA 建立在相同的 **206 个底层场景**之上。

概念上：

```text
EU_i.question == EA_i.scenario
```

因此：

> **412 道问题 ≠ 412 个相互独立的场景**

这一点在构建训练集/测试集划分或进行统计分析时非常重要。

如果将数据集划分为训练集和评测集，应当以**场景对（scenario-pair）为单位**进行划分，而不是分别独立地划分 EU 和 EA 问题。

否则，同一个底层场景可能出现在不同的数据划分中。

---

# 六、评测

如果你已经有预测文件，那么 CEUA 可以完全**离线评测**：

```bash
python src/evaluate.py \
  --data data/eu.jsonl data/ea.jsonl \
  --predictions my_predictions.jsonl
```

评测器会报告：

* EU Accuracy（EU 准确率）
* EA Accuracy（EA 准确率）
* Overall Accuracy（总体准确率）
* Single-choice Accuracy（单选准确率）
* Two-choice Accuracy（双选准确率）
* EA Single-choice Accuracy（EA 单选准确率）
* EA Two-choice Accuracy（EA 双选准确率）
* Two-choice Partial Score（双选部分得分）

## 预测格式

预测结果以 JSONL 格式保存：

```json
{"qid": "EU_001", "prediction": "B"}
{"qid": "EA_014", "prediction": "CD"}
```

在线推理流程与离线评测器使用相同的预测规范化规则，从而保证模型在线评测与独立评分的一致性。

---

# 七、基线

由于 CEUA 是一个开放基准，并且公开提供标准答案，因此建议在报告模型结果时，同时报告简单的统计基线。

当前数据集存在一些答案分布偏差。

例如：

* `C` 占 EU 单选题标准答案的 **43.2%**
* `A` 占 EA 单选题标准答案的 **50.9%**
* 在 EA 单选题中，**唯一最长的选项**有 **69.9%** 的概率就是标准答案

参考基线：

| 基线                 |     得分 |
| ------------------ | -----: |
| 恒定选择 `C` — Overall | 30.83% |
| 恒定选择 `C` — EU      | 43.20% |
| 恒定选择 `A` — EA 单选   | 50.87% |
| 恒定选择 `AB` — 双选     | 45.45% |
| 选择最长选项 — EA 单选     | 69.94% |
| 选择最长选项 — EU        |  1.46% |

这些基线对于理解 EA 的结果尤其重要。

例如，EA 单选题中“选择最长选项”这一简单策略就可以达到 **69.94%**，这说明在分析模型结果时，不能只看原始准确率，还需要考虑数据集中的**答案位置偏差和选项长度统计特征**。

---

# 八、快速开始

CEUA 只需要：

* **Python 3.8+**
* 不需要第三方运行时依赖

## 1. 克隆仓库

```bash
git clone <your-repo-url>
cd ceua-benchmark
```

## 2. 设置 API Key

对于 OpenAI：

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

其他支持的服务商使用对应的环境变量。

## 3. 运行基准测试

```bash
python -m src.main \
  --provider openai \
  --model <model-name>
```

结果会保存到：

```text
results/<model>/
├── predictions.jsonl
└── raw_responses.jsonl
```

---

# 九、本地测试

项目提供了一个 mock provider，可以在**不使用 API Key** 的情况下测试完整流程：

```bash
python -m src.main \
  --provider mock \
  --model mock-chat \
  --limit 10
```

这对于验证以下功能非常有用：

* 数据集加载
* 推理
* 预测结果规范化
* 结果生成

---

# 十、支持的服务商

CEUA 当前支持：

| 服务商         | 环境变量                  |
| ----------- | --------------------- |
| OpenAI      | `OPENAI_API_KEY`      |
| DeepSeek    | `DEEPSEEK_API_KEY`    |
| OpenRouter  | `OPENROUTER_API_KEY`  |
| SiliconFlow | `SILICONFLOW_API_KEY` |
| Mock        | —                     |

运行方式：

```bash
python -m src.main \
  --provider <provider> \
  --model <model-name>
```

查看所有可用选项：

```bash
python -m src.main -h
```

---

# 十一、可复现性

默认推理配置使用：

```text
temperature = 0
```

推理流程支持：

* timeout 和 retry（超时与重试）
* exponential backoff（指数退避）
* resumable evaluation（可恢复评测）
* `--no-resume`
* 可配置 worker 数量
* 可配置最大 token 数

如果修改了 prompt 或推理参数，建议关闭 resume：

```bash
python -m src.main \
  --provider <provider> \
  --model <model-name> \
  --no-resume
```

这样可以确保所有问题都在新的配置下重新进行评测。

---

# 十二、仓库结构

```text
ceua-benchmark/
├── data/
│   ├── eu.jsonl
│   └── ea.jsonl
├── src/
│   ├── main.py
│   ├── model.py
│   ├── data.py
│   ├── evaluate.py
│   ├── validate_dataset.py
│   └── utils.py
├── tests/
├── results/
├── DATASET_CARD.md
├── DATASET_LICENSE
├── README.md
├── LICENSE
└── CITATION.cff
```

---

# 十三、数据集卡片

如需了解数据集的详细信息，请参阅 `DATASET_CARD.md`。

数据集卡片介绍：

* 数据来源（Data Provenance）
* 数据集构建方式（Dataset Construction）
* 情绪分类体系（Emotion Taxonomy）
* 人工审核（Human Review）
* 标准答案解释（Gold Label Interpretation）
* 已知局限性（Known Limitations）

---

# 十四、与 EmoBench 的关系

**EU / EA 任务划分**以及部分基准设计思想受到 **EmoBench（ACL 2024）**的启发。EmoBench 是概念层面的参考。

关于数据来源，我们采用谨慎表述：

> Based on the repository materials, no direct reuse was identified; final provenance confirmation remains the author's responsibility.

也就是说：基于本仓库现有材料，未发现直接复制或翻译 EmoBench 题目的证据；但"完全独立"的最终来源确认仍由数据作者负责。CEUA 的中文情境、选项与标注均为独立构造。

请不要把 CEUA 描述为 EmoBench 的官方中文版、翻译版或复现版。

---

# 十五、重要说明

## 开放基准

标准答案公开存储在：

```text
data/*.jsonl
```

因此，CEUA：

* **不是隐藏测试集基准**
* **不具备抗数据污染能力**
* **本身不适合作为防作弊的公开排行榜**

对于正式的排行榜评测，建议使用**私有测试集 + 服务端评测**。

## 数据集偏差

CEUA 存在可观察的答案分布偏差。

因此，在比较不同模型时，研究人员应当报告官方提供的简单基线。

## 场景依赖性

EU 和 EA 围绕相同的底层场景成对构建。

因此，在进行统计分析时，应考虑这种依赖关系，而不能把全部 **412 道问题视为 412 个相互独立的场景**。

## 适用范围（Intended Use）

CEUA 适合用于：

* 中文情绪理解（EU）与情绪应用（EA）的研究与模型评测；
* 不同语言模型之间的能力对比；
* 提示词 / 指令效果评估；
* 模型行为的诊断性分析；
* 语言模型中文社会情绪推理能力的基准探索。

使用建议：由于 EU / EA 基于相同底层场景配对构建，划分训练 / 开发 / 测试集时应以**场景对（scenario-pair）为单位**，避免信息泄漏（见第五节）。

## 超出范围（Out-of-Scope Use）

CEUA **不应**用于以下用途：

* 临床诊断、心理健康评估、心理治疗或危机干预；
* 对真实个人进行情绪画像（emotion profiling）、心理画像或敏感属性推断；
* 未经同意的情绪监控（surveillance / monitoring）；
* 招聘、保险、信贷、医疗等会对个人产生重大影响的高风险自动化决策；
* 把模型在本基准上的输出直接当作对任何真实人物心理状态的判断。

## 安全与伦理（Safety Considerations）

* 本数据集的情绪标签（EU 的 `answer` 字段）是 **benchmark 中定义在情境层面的预期 / 主要情绪类别**，**不是**对任何真实人物的临床或心理学诊断。
* 数据中的情境是**虚构的基准示例**，不应被当作临床案例，也不代表对任何群体的价值判断。
* 模型在本基准上的表现衡量的是**可观察的任务表现**，不能作为"模型拥有类人情绪体验"的证据。
* 数据中的虚构情境涉及人际冲突、关系失落、职场压力、倦怠与霸凌取笑等情绪压力主题（例如 `EU_076`、`EU_133`、`EU_194`）。这些属于正常的基准内容；如在教学或公开分享中使用，建议根据受众做适当说明。
* 数据**不包含**真实个人的姓名、联系方式、账号、证件、人脸、声音等个人信息，情境主体均为虚构人物。

---

# 十六、测试

运行测试套件：

```bash
python -m unittest discover -s tests -t .
```

---

# 十七、引用

如果 CEUA 对你的研究有帮助，请使用 `CITATION.cff` 中提供的元数据引用该基准。

---

# 十八、许可证

CEUA 采用**双许可证**结构，请务必区分**数据集**与**代码**：

| 内容  | 许可证            | 适用范围                                          |
| --- | -------------- | --------------------------------------------- |
| 数据集 | **CC BY 4.0**  | `data/*.jsonl` 及数据集文档（`DATASET_CARD.md`、`DATASET_LICENSE`） |
| 代码  | **MIT License** | 仓库源代码（`src/`、`tests/` 等），见 `LICENSE`           |

## 数据集许可证：CC BY 4.0

> **CEUA Benchmark dataset is released under CC BY 4.0. Users are free to use, copy, modify, adapt, and redistribute the dataset, including for commercial purposes, provided that appropriate attribution is given.**

也就是说：

* ✅ 可以复制、下载和再发布
* ✅ 可以修改、改编
* ✅ 可以用于研究
* ✅ 可以用于商业用途
* ➡️ 条件是**适当署名**：给出适当的署名、提供许可证链接，并说明是否做了修改

CC BY 4.0 **不**附加"禁止商业使用""禁止修改""仅限研究使用""必须使用相同许可证"等限制。

* 许可证全文：<https://creativecommons.org/licenses/by/4.0/legalcode>
* 数据集许可证说明文件：`DATASET_LICENSE`

## 代码许可证：MIT

仓库中的源代码采用 **MIT License** 发布，详见 `LICENSE`。

## 重要区分

* 本项目**不是**"整个项目都是 MIT"。
* 如果使用对象是**数据**（`data/*.jsonl`），适用 **CC BY 4.0**。
* 如果使用对象是**代码**（`src/`、`tests/` 等），适用 **MIT License**。

Copyright (c) 2026 Wei Zhao.

---

# 十九、项目信息与联系

* 作者：Wei Zhao
* 数据集卡片：`DATASET_CARD.md`
* 数据集许可证：`DATASET_LICENSE`
* 代码许可证：`LICENSE`
* 引用信息：`CITATION.cff`（其中的仓库地址等元数据仍需在发布前补充）

如需反馈问题或参与贡献，请通过本仓库的 Issue / Pull Request 渠道联系维护者。
