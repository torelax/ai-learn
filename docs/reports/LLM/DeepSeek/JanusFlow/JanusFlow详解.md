# JanusFlow：在一个语言模型中结合自回归理解与连续流图像生成

> 💡 JanusFlow 保留理解与生成的编码分离，让共享 LLM 同时做文字自回归和图像潜空间的速度预测，再用语义表示对齐改善连续图像生成。
>
> [论文](https://arxiv.org/abs/2411.07975) · [官方仓库](https://github.com/deepseek-ai/Janus) · [补充解读](JanusFlow解读.md)

Janus 用离散视觉 token 统一两类任务，生成质量仍受 tokenizer 的量化与重建限制。JanusFlow 因而保留编码分离，将生成侧改成连续潜空间中的 rectified flow。共享 LLM 在文字任务上预测下一 token，在图像任务上多次预测当前噪声状态的速度，不再逐个生成 VQ 索引。报告还利用理解侧 SigLIP 的语义特征约束生成中的中间表示，在同一框架内连接“看懂图像”和“逐步生成图像”。实验同时检查理解、生成质量和语义遵循，并用单任务模型与组件消融衡量统一的代价。

## 从离散 AR 到连续 rectified flow

文字仍使用 $\log P_\theta(x)=\sum_i\log P_\theta(x_{i+1}\mid x_{\le i})$。图像先编码到 SDXL-VAE 潜空间，记真实潜变量 $x$、高斯噪声 $z_0$，构造直线路径：

$$
z_t=tx+(1-t)z_0,\qquad \mathcal L_{RF}=\mathbb E\|v_\theta(z_t,t\mid c)-(x-z_0)\|^2.
$$

$c$ 是文本条件，$t$ 采用 logit-normal 采样。训练学习的是从噪声到数据的速度场；推理数值求解 $dz_t/dt=v_\theta(z_t,t\mid c)$。Euler 一步为 $z_{t+\Delta t}=z_t+v_\theta\Delta t$，重复到 $t=1$ 再经 VAE 解码。

这里一步会更新整张潜图，不是一步生成一个视觉 token。LLM 依然使用 causal attention，初步实验没有发现改 mask 的收益；因果 mask 也不妨碍多次整图速度更新。

## 架构与编码分离

![理解分支、连续生成分支与共享 LLM](figs/S3-F2-01.png)

理解侧使用 SigLIP-Large-Patch16，经线性映射进入 LLM。生成侧使用从头初始化的 ConvNeXt encoder/decoder，前者将带噪 VAE latent patchify 为输入，后者把 LLM 输出恢复成同形速度；两者间有长 skip connection，合计约 70M 参数。

主干为增强版 DeepSeek-LLM 1.3B，24 层、长度 4096，图像分辨率 384。理解 SigLIP 约 300M，生成 encoder 包含 $2\times2$ patchify、两个 ConvNeXt block 和线性层，decoder 使用相应 block 与 pixel shuffle。VAE 是表示与重建工具，速度预测由共享 LLM 参与完成。

## 语义表示对齐 REPA

取 LLM 第六层的生成特征 $q_\theta(z_t)$，经三层 MLP $h_\varphi$，与干净图像的 SigLIP 特征对齐：

$$
\mathcal L_{REPA}=-\mathbb E\,\operatorname{cos}\left(\operatorname{stopgrad}(f_{enc}(x)),h_\varphi(q_\theta(z_t))\right).
$$

生成特征的空间网格与 SigLIP 网格对齐，逐位置平均余弦相似度。stop-gradient 阻止目标分支为迁就生成而移动，鼓励含噪输入的中间表示提前建立语义结构。文字任务用 $\mathcal L_{AR}$，生成用 $\mathcal L_{RF}+\mathcal L_{REPA}$。

## 三阶段训练与推理参数

第一阶段只训练新初始化模块；第二阶段解冻除理解编码器外的主体，联合文本、理解和图像生成；第三阶段再解冻 SigLIP，用对话及高质量生成样本做 SFT。

|  | Stage 1 | Stage 2 | Stage 3 |
| --- | --- | --- | --- |
| Learning Rate | $1.0\times 10^{-4}$ | $1\times 10^{-4}$ | $2.0\times 10^{-5}$ |
| LR Scheduler | Constant | Constant | Constant |
| Weight Decay | $0.0$ | $0.0$ | $0.0$ |
| Gradient Clip | $1.0$ | $1.0$ | $1.0$ |
| Optimizer | AdamW ( $\beta_{1}=0.9,\beta_{2}=0.95$ ) | AdamW ( $\beta_{1}=0.9,\beta_{2}=0.95$ ) | AdamW ( $\beta_{1}=0.9,\beta_{2}=0.95$ ) |
| Warm-up Steps | $2,000$ | $2,000$ | $1,000$ |
| Training Steps | $10,000$ | $390,000$ | $26,000$ |
| Batch Size | $512$ | $512$ | $256$ |
| Data Ratio | $50:50:0$ | $14:80:6$ | $21:70:9$ |

Data Ratio 为理解:生成:文本，不能沿用 Janus 表中的排列顺序。理解图像按长边缩放再补齐，生成图像按短边缩放再裁剪；packing 到 4096，训练约 1,600 A100 GPU-days，EMA 0.99。

训练随机丢弃 10% 文本条件以支持 CFG。推理速度为 $v=w v_c+(1-w)v_u$；FID 主设置使用 $w=2$、30 个采样步，步数与 guidance 的变化另在附录扫描。因此不能只看一个 FID 而省略采样成本。

## 主要结果与评测含义

| Method | Params | FID $\downarrow$ |
| --- | --- | --- |
| LWM [ 59 ] | 7B | 17.77 |
| VILA-U 256 [ 99 ] | 7B | 12.81 |
| VILA-U 384 [ 99 ] | 7B | 7.69 |
| Show-o [ 100 ] | 1.3B | 15.18 |
| Janus [ 97 ] | 1.3B | 10.10 |
| JanusFlow (Ours) | 1.3B | 9.51 |

MJHQ-30K FID 从 Janus 的 10.10 改为 9.51，低于同级 Show-o 15.18，但没有超过表中 7B VILA-U-384 的 7.69。该比较同时涉及模型架构和规模，不能描述成纯生成目标的严格因果对照。

| Type | Method | Params | Single Obj. | Two Obj. | Count. | Colors | Pos. | Color Attri. | Overall $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gen. Only | LlamaGen [ 86 ] | $0.8$ B | $0.71$ | $0.34$ | $0.21$ | $0.58$ | $0.07$ | $0.04$ | $0.32$ |
| Gen. Only | LDM [ 77 ] | $1.4$ B | $0.92$ | $0.29$ | $0.23$ | $0.70$ | $0.02$ | $0.05$ | $0.37$ |
| Gen. Only | SDv $1.5$ [ 77 ] | $0.9$ B | $0.97$ | $0.38$ | $0.35$ | $0.76$ | $0.04$ | $0.06$ | $0.43$ |
| Gen. Only | PixArt- $\alpha$ [ 9 ] | $0.6$ B | $0.98$ | $0.50$ | $0.44$ | $0.80$ | $0.08$ | $0.07$ | $0.48$ |
| Gen. Only | SDv $2.1$ [ 77 ] | $0.9$ B | $0.98$ | $0.51$ | $0.44$ | $0.85$ | $0.07$ | $0.17$ | $0.50$ |
| Gen. Only | DALL-E $2$ [ 76 ] | $6.5$ B | $0.94$ | $0.66$ | $0.49$ | $0.77$ | $0.10$ | $0.19$ | $0.52$ |
| Gen. Only | Emu $3$ -Gen [ 95 ] | $8$ B | $0.98$ | $0.71$ | $0.34$ | $0.81$ | $0.17$ | $0.21$ | $0.54$ |
| Gen. Only | SDXL [ 73 ] | $2.6$ B | $0.98$ | $0.74$ | $0.39$ | $0.85$ | $0.15$ | $0.23$ | $0.55$ |
| Gen. Only | IF-XL [ 17 ] | $4.3$ B | $0.97$ | $0.74$ | $0.66$ | $0.81$ | $0.13$ | $0.35$ | $0.61$ |
| Gen. Only | DALL-E $3$ [ 6 ] | - | $0.96$ | $0.87$ | $0.47$ | $0.83$ | $0.43$ | $0.45$ | $0.67$ |
| Unified | Chameleon [ 88 ] | $34$ B | - | - | - | - | - | - | $0.39$ |
| Unified | LWM [ 59 ] | $7$ B | $0.93$ | $0.41$ | $0.46$ | $0.79$ | $0.09$ | $0.15$ | $0.47$ |
| Unified | SEED-X † [ 27 ] | $17$ B | $0.97$ | $0.58$ | $0.26$ | $0.80$ | $0.19$ | $0.14$ | $0.49$ |
| Unified | Show-o [ 100 ] | $1.3$ B | $0.95$ | $0.52$ | $0.49$ | $0.82$ | $0.11$ | $0.28$ | $0.53$ |
| Unified | Janus [ 97 ] | $1.3$ B | $0.97$ | $0.68$ | $0.30$ | $0.84$ | $0.46$ | $0.42$ | $0.61$ |
| Unified | Transfusion [ 108 ] | $7.3$ B | - | - | - | - | - | - | 0.63 |
| Unified | JanusFlow (Ours) | 1.3B | 0.97 | 0.59 | 0.45 | 0.83 | 0.53 | 0.42 | 0.63 |

GenEval overall 0.63，考察物体、数量与关系遵循；它与 FID 分工不同。DPG-Bench 用密集长提示评估更复杂的语义约束，报告也给出各类别分数。

| Type | Model | LLM Param | POPE | MME-P | MMB dev | SEED | VQAv2 test | GQA | MMMU | MM-Vet | ChartQA | TextVQA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Und. Only | MobileVLM [ 12 ] | $2.7$ B | $84.9$ | $1288.9$ | $59.6$ | - | - | $59.0$ | - | - | - | $47.5$ |
| Und. Only | MobileVLM-V2 [ 13 ] | $2.7$ B | $84.7$ | $1440.5$ | $63.2$ | - | - | $61.1$ | - | - | - | $57.5$ |
| Und. Only | LLaVA-Phi [ 109 ] | $2.7$ B | $85.0$ | $1335.1$ | $59.8$ | - | $71.4$ | - | - | $28.9$ | - | $48.6$ |
| Und. Only | LLaVA [ 58 ] | $7$ B | $76.3$ | $809.6$ | $38.7$ | $33.5$ | - | - | - | $25.5$ | - | - |
| Und. Only | LLaVA-v1.5 [ 56 ] | $7$ B | $85.9$ | $1510.7$ | $64.3$ | $58.6$ | $78.5$ | $62.0$ | $35.4$ | $31.1$ | - | $58.2$ |
| Und. Only | InstructBLIP [ 15 ] | $7$ B | - | - | $36.0$ | $53.4$ | - | $49.2$ | - | $26.2$ | - | $50.1$ |
| Und. Only | Qwen-VL-Chat [ 4 ] | $7$ B | - | $1487.5$ | $60.6$ | $58.2$ | $78.2$ | $57.5$ | - | - | $66.3$ | $61.5$ |
| Und. Only | LLaVA-NeXT [ 57 ] | $7$ B | - | $1519.3$ | - | - | - | - | $35.1$ | - | $54.8$ | - |
| Und. Only | Qwen2-VL [ 94 ] | $7$ B | - | - | - | - | - | - | $54.1$ | $62.0$ | $83.0$ | $84.3$ |
| Und. Only | IDEFICS- $9$ B [ 44 ] | $8$ B | - | - | $48.2$ | - | $50.9$ | $38.4$ | - | - | - | $25.9$ |
| Und. Only | Emu $3$ -Chat [ 95 ] | $8$ B | $85.2$ | - | $58.5$ | $68.2$ | $75.1$ | $60.3$ | $31.6$ | - | $68.6$ | $64.7$ |
| Und. Only | InstructBLIP [ 15 ] | $13$ B | $78.9$ | $1212.8$ | - | - | - | $49.5$ | - | $25.6$ | - | $50.7$ |
| Und. Only | LLaVA-v1.5-Phi-1.5 [ 100 ] | $1.3$ B | $84.1$ | $1128.0$ | - | - | $75.3$ | $56.5$ | $30.7$ | - | - | - |
| Und. Only | MobileVLM [ 12 ] | $1.4$ B | $84.5$ | $1196.2$ | $53.2$ | - | - | $56.1$ | - | - | - | $41.5$ |
|  | MobileVLM-V2 [ 13 ] | $1.4$ B | $84.3$ | $1302.8$ | $57.7$ | - | - | $59.3$ | - | - | - | $52.1$ |
| Unified | Gemini-Nano-1 [ 89 ] | $1.8$ B | - | - | - | - | $62.7$ | - | $26.3$ | - | $53.6$ | $62.5$ |
| Unified | LWM [ 59 ] | $7$ B | $75.2$ | - | - | - | $55.8$ | $44.8$ | - | $9.6$ | - | - |
| Unified | VILA-U [ 99 ] | $7$ B | $85.8$ | $1401.8$ | - | $59.0$ | $79.4$ | $60.8$ | - | $33.5$ | - | $60.8$ |
| Unified | Chameleon [ 88 ] | $7$ B | - | - | - | - | - | - | $22.4$ | $8.3$ | - | - |
| Unified | DreamLLM † [ 19 ] | $7$ B | - | - | - | - | $72.9$ | - | - | $36.6$ | - | $41.8$ |
| Unified | LaVIT † [ 37 ] | $7$ B | - | - | - | - | $66.0$ | $46.8$ | - | - | - | - |
| Unified | Emu † [ 87 ] | $13$ B | - | - | - | - | $52.0$ | - | - | - | - | - |
| Unified | NExT-GPT † [ 98 ] | $13$ B | - | - | - | - | $66.7$ | - | - | - | - | - |
| Unified | Show-o [ 100 ] | $1.3$ B | $73.8$ | $948.4$ | - | - | $59.3$ | $48.7$ | $25.1$ | - | - | - |
| Unified | Janus [ 97 ] | $1.3$ B | $87.0$ | $1338.0$ | $69.4$ | $63.7$ | $77.3$ | $59.1$ | $30.5$ | $34.3$ | - | - |
|  | JanusFlow (Ours) | $1.3$ B | $88.0$ | $1333.1$ | $74.9$ | $70.5$ | $79.8$ | $60.3$ | $29.3$ | $30.9$ | $64.6$ | $55.5$ |

理解表覆盖 POPE、MMBench、VQAv2、MMMU、ChartQA 和 TextVQA，证明引入连续生成后仍能保留有竞争力的理解能力，而不是仅报告生成更好。

## 关键消融

| Exp. ID | Model Setting | Model Setting | Model Setting | Model Setting | Train. Iter. | Evaluation Benchmarks | Evaluation Benchmarks | Evaluation Benchmarks | Evaluation Benchmarks | Evaluation Benchmarks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp. ID | REPA | Und. Modules | Gen. Modules | Type | Train. Iter. | POPE $\uparrow$ | VQAv2 ${}_{val}\uparrow$ | GQA $\uparrow$ | FID $\downarrow$ | CLIP $\uparrow$ |
| A | $\times$ | SigLIP | VAE † +ConvNeXt | Unified | 50,000 | 82.40 | 69.62 | 54.43 | 19.84 | 24.94 |
| B | $\checkmark$ | Shared VAE † +ConvNeXt | Shared VAE † +ConvNeXt | Unified | 50,000 | 78.13 | 53.94 | 44.04 | 18.05 | 26.38 |
| C | $\checkmark$ | VAE+ConvNeXt | VAE † +ConvNeXt | Unified | 50,000 | 75.30 | 55.41 | 44.44 | 17.53 | 26.32 |
| D | $\checkmark$ | SigLIP | - | Und. Only | 13,000 | 85.03 | 69.10 | 54.23 | - | - |
| E | $\checkmark$ | - | VAE † +ConvNeXt | Gen. Only | 37,000 | - | - | - | 16.69 | 26.89 |
| F | $\checkmark$ | SigLIP | VAE † +ConvNeXt | Unified | 50,000 | 84.73 | 69.20 | 54.83 | 17.61 | 26.40 |

消融统一在 **256 分辨率**、主要为 50k 次训练完成，不能与 384 分辨率最终模型直接混读。A→F 加入 REPA，FID 从 19.84 到 17.61，CLIP 从 24.94 到 26.40。B/C 使用 VAE+ConvNeXt 的共享或独立分支，理解明显弱于 F 的预训练 SigLIP 路线，说明除了分离，还需要适合语义任务的编码器。

D/E 为仅理解/仅生成，按对应数据比例匹配任务训练量。F 与它们接近，却不是所有指标完全相同：例如生成专用 E 的 FID 16.69 仍优于 F 的 17.61。论文的支持点是统一代价较小，不能写成没有任何任务竞争。

## 局限与结论

连续流减少离散码本的约束，但仍受 VAE 重建、384 分辨率及多步采样成本限制。该报告以 1.3B 主干验证，没有提供足够证据直接断言大规模多任务训练也有相同取舍。

JanusFlow 表明共享 LLM 不必让所有输出都采用离散自回归；文字和图像可使用不同训练目标，借助编码分离及语义对齐协调。它与 Janus-Pro 的数据和规模扩展是两条不同演进路线。
