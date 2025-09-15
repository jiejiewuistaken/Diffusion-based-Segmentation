# 3D肺肿瘤分割DDPM实验设计

## 第3章：预训练的3D条件扩散分割模型（肺部通用分割）

### 3.1 模型设计

#### 3.1.1 总体结构：3D UNet-DDPM
本研究采用基于扩散概率模型（DDPM）的3D UNet架构进行肺肿瘤分割。模型的核心组件包括：

**噪声预测网络**：采用3D UNet作为去噪网络，预测噪声ε或速度ν
- 输入：3D CT图像 + 噪声分割掩码
- 输出：预测的噪声或速度场
- 时间嵌入：通过正弦位置编码将时间步t嵌入到网络特征中

**时间嵌入模块**：
```python
# 时间嵌入维度为model_channels * 4
time_embed_dim = model_channels * 4
self.time_embed = nn.Sequential(
    linear(model_channels, time_embed_dim),
    nn.SiLU(),
    linear(time_embed_dim, time_embed_dim),
)
```

#### 3.1.2 模块说明

**编码器-解码器架构**：
- 编码器：4层下采样，通道数递增[64, 128, 256, 512]
- 瓶颈层：最高分辨率特征提取
- 解码器：4层上采样，通过跳跃连接恢复空间分辨率
- 残差块：每层包含2个ResBlock，支持时间条件

**3D/轴向注意力机制**：
- 在1/16分辨率层（L4）、瓶颈层（L5）、解码1/16层（U4）启用注意力
- 采用多头自注意力，头数为4
- 支持3D空间注意力和轴向注意力两种模式

**噪声调度与损失函数**：
- 噪声调度：线性调度，β_start=0.0001, β_end=0.02
- 损失函数：MSE损失预测噪声ε
- 训练步数：1000步扩散过程

#### 3.1.3 训练-采样流程

**DDIM采样**：
- 确定性采样，步数可调节（50-100步）
- 采样公式：x_{t-1} = √(ᾱ_{t-1}) * x̂_0 + √(1-ᾱ_{t-1}-σ²) * ε̂ + σ * z

**DPM-Solver加速**：
- 高阶求解器，减少采样步数
- 支持2-3阶精度，显著提升推理速度

### 3.2 数据集与预处理

#### 3.2.1 数据集图像元数据
**数据来源**：
- LIDC-IDRI数据集：1018例肺部CT扫描
- 院内肺部分割数据：200例高分辨率CT（补充数据）
- 数据格式：NIfTI格式，体素间距标准化为1.0×1.0×1.0 mm³

**图像规格**：
- 空间分辨率：512×512×Z（Z为切片数，范围200-600）
- 强度范围：-1000到400 HU
- 切片厚度：1-3mm
- 重建核：B70f（标准肺窗）

#### 3.2.2 数据增强策略

**强度增强**：
- 窗宽窗位调整：窗宽1500HU，窗位-600HU
- 高斯噪声：σ=0.01-0.05
- 高斯模糊：σ=0.5-1.5像素
- 强度缩放：±20%随机变化

**几何增强**：
- 3D随机翻转：沿x、y、z轴，概率0.5
- 3D随机旋转：±15°范围内
- 3D随机缩放：0.9-1.1倍
- 弹性变形：控制点网格8×8×8，最大位移5像素

**数据预处理流程**：
```python
# 体素间距标准化
def normalize_spacing(image, target_spacing=(1.0, 1.0, 1.0)):
    # 重采样到目标体素间距
    
# 强度归一化
def normalize_intensity(image, window_center=-600, window_width=1500):
    # 窗宽窗位标准化
    
# 3D Patch提取
def extract_3d_patches(image, patch_size=(128, 128, 128), overlap=0.5):
    # 前景优先采样策略
```

### 3.3 评价指标

#### 3.3.1 分割精度指标
- **Dice相似系数**：衡量分割重叠度
- **Hausdorff距离95%**：评估边界精度
- **平均表面距离（ASSD）**：边界一致性
- **体积差异**：预测与真实体积的差异

#### 3.3.2 效率指标
- **推理延时**：单次分割耗时（秒）
- **显存占用**：峰值GPU内存使用（GB）
- **采样步数**：DDIM/DPM-Solver步数

### 3.4 实验环境和参数设置

#### 3.4.1 硬件配置
- **GPU**：NVIDIA A100 80GB × 4
- **CPU**：Intel Xeon Gold 6248R × 2
- **内存**：256GB DDR4
- **存储**：NVMe SSD 2TB

#### 3.4.2 软件环境
- **深度学习框架**：PyTorch 1.12.0
- **CUDA版本**：11.6
- **Python版本**：3.8.10
- **医学图像处理**：SimpleITK 2.2.0

#### 3.4.3 训练参数
```python
# 模型参数
MODEL_FLAGS = {
    "image_size": 128,  # 3D patch尺寸
    "num_channels": 64,
    "num_res_blocks": 2,
    "num_heads": 4,
    "attention_resolutions": [16],  # 1/16分辨率
    "dims": 3,  # 3D模型
    "use_scale_shift_norm": True,
    "use_fp16": True
}

# 扩散参数
DIFFUSION_FLAGS = {
    "diffusion_steps": 1000,
    "noise_schedule": "linear",
    "model_mean_type": "EPSILON",
    "model_var_type": "LEARNED_RANGE"
}

# 训练参数
TRAIN_FLAGS = {
    "lr": 1e-4,
    "batch_size": 4,  # 3D patch batch size
    "microbatch": 2,
    "ema_rate": 0.9999,
    "weight_decay": 1e-6,
    "gradient_checkpointing": True
}
```

### 3.5 实验设计与结果分析

#### 3.5.1 对比模型
1. **3D UNet**：标准3D UNet基线模型
2. **nnUNet**：自配置的nnUNet（可选）
3. **3D DDPM**：本研究提出的扩散模型

#### 3.5.2 实验结果分析

**性能对比**（需要实际数据填充）：
| 模型 | Dice (%) | HD95 (mm) | ASSD (mm) | 体积差 (%) | 推理时间 (s) |
|------|----------|-----------|-----------|------------|--------------|
| 3D UNet | [数据] | [数据] | [数据] | [数据] | [数据] |
| nnUNet | [数据] | [数据] | [数据] | [数据] | [数据] |
| 3D DDPM | [数据] | [数据] | [数据] | [数据] | [数据] |

**结果讨论**：
- 扩散模型在分割精度上的优势
- 计算代价与性能的平衡
- 为第4章微调提供特征迁移价值

---

## 第4章：消融区域预测

### 4.1 模型设计

#### 4.1.1 FiLM条件注入
**功率/时长条件编码**：
```python
class FiLMConditioning(nn.Module):
    def __init__(self, condition_dim=2, feature_dim=512):
        super().__init__()
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim * 2)  # scale + shift
        )
    
    def forward(self, features, power, duration):
        # power: [B, 1], duration: [B, 1]
        conditions = torch.cat([power, duration], dim=1)
        scale_shift = self.condition_mlp(conditions)
        scale, shift = scale_shift.chunk(2, dim=1)
        return features * (1 + scale) + shift
```

**条件注入位置**：
- 与时间嵌入并行注入
- 在每个ResBlock的scale-shift层应用
- 条件维度：功率(W) + 时长(s) = 2维

#### 4.1.2 探针距离引导的注意力偏置

**距离场构建**：
```python
def compute_probe_distance_field(probe_position, image_shape, sigma=10.0):
    """
    计算探针位置到图像各点的距离场
    probe_position: [B, 3] 探针3D坐标
    image_shape: (D, H, W) 图像尺寸
    sigma: 高斯核标准差
    """
    # 构建3D坐标网格
    coords = torch.meshgrid(torch.arange(image_shape[0]),
                           torch.arange(image_shape[1]),
                           torch.arange(image_shape[2]))
    coords = torch.stack(coords, dim=0).float()
    
    # 计算欧几里得距离
    distances = torch.norm(coords - probe_position.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), dim=0)
    
    # 高斯归一化
    distance_field = torch.exp(-distances**2 / (2 * sigma**2))
    return distance_field
```

**注意力偏置形式**：
```python
def apply_attention_bias(attention_weights, distance_field, alpha=0.5):
    """
    在注意力权重上应用距离偏置
    attention_weights: [B, H, N, N] 注意力权重
    distance_field: [B, 1, D, H, W] 距离场
    alpha: 偏置强度
    """
    # 将距离场下采样到注意力分辨率
    bias = F.interpolate(distance_field, size=attention_weights.shape[-1], mode='trilinear')
    
    # 计算偏置项：logits += α·(s_i + s_j)
    bias_i = bias.unsqueeze(-1).expand(-1, -1, -1, -1, bias.shape[-1])
    bias_j = bias.unsqueeze(-2).expand(-1, -1, -1, bias.shape[-1], -1)
    attention_bias = alpha * (bias_i + bias_j)
    
    return attention_weights + attention_bias
```

**层位选择策略**：
- **L4层**（编码1/16）：捕获全局空间关系
- **L5层**（瓶颈层）：最高语义特征
- **U4层**（解码1/16）：细节恢复阶段

#### 4.1.3 训练细节

**3D Patch采样**：
- Patch尺寸：128×128×128体素
- 前景优先采样：80%包含肿瘤区域
- 重叠率：50%滑窗采样

**条件丢弃策略**：
- 训练时随机丢弃条件（概率0.1）
- 实现无条件生成能力
- 提升模型鲁棒性

**损失函数**：
```python
def ablation_loss(pred_mask, gt_mask, power, duration, probe_pos):
    # 分割损失
    dice_loss = 1 - dice_coefficient(pred_mask, gt_mask)
    
    # 条件一致性损失
    condition_loss = mse_loss(pred_conditions, torch.cat([power, duration], dim=1))
    
    # 距离感知损失
    distance_loss = distance_aware_loss(pred_mask, gt_mask, probe_pos)
    
    total_loss = dice_loss + 0.1 * condition_loss + 0.05 * distance_loss
    return total_loss
```

### 4.2 数据集

#### 4.2.1 消融手术数据
**术前/术中CT**：
- 术前CT：消融前基线扫描
- 术中CT：探针插入后实时扫描
- 术后CT：消融完成后验证扫描

**消融区分割标注**：
- 专家手工标注的消融区域
- 多时间点标注（术中、术后24h、术后1周）
- 标注一致性验证（ICC > 0.8）

**探针轨迹/位姿**：
- 探针3D坐标序列
- 插入角度和深度
- 实时位置追踪数据

**功率/时长日志**：
- 消融功率：10-200W
- 消融时长：30-600秒
- 温度曲线：实时温度监测

### 4.3 评价指标

#### 4.3.1 分割精度指标
- **Dice相似系数**：消融区域重叠度
- **Hausdorff距离95%**：边界精度
- **体积差异**：预测与真实消融体积差异

#### 4.3.2 临床评价指标
**安全边界覆盖**：
- 10mm安全边界覆盖率
- 欠消融体积（< 5mm边界）
- 过消融体积（> 15mm边界）

**可控性评价**：
- 功率-体积响应曲线
- 时长-体积响应曲线
- 参数单调性验证

**不确定性量化**（可选）：
- 预期校准误差（ECE）
- 风险-覆盖曲线
- 置信度区间分析

### 4.4 实验环境和参数设置

#### 4.4.1 微调参数
```python
# 微调参数
FINE_TUNE_FLAGS = {
    "lr": 5e-5,  # 较低学习率
    "batch_size": 2,  # 3D patch batch size
    "num_epochs": 100,
    "warmup_epochs": 10,
    "condition_dropout": 0.1,
    "probe_bias_alpha": 0.5,  # 注意力偏置强度
    "distance_sigma": 10.0,   # 距离场高斯核标准差
}
```

#### 4.4.2 推理加速
- **DDIM采样**：50-100步
- **DPM-Solver**：20-50步
- **滑窗融合**：重叠区域加权平均

### 4.5 实验设计与结果分析

#### 4.5.1 消融实验设计

**基线对比**：
1. **UNet-3D**（非扩散）：标准3D UNet
2. **UNet-3D + Diffusion**（无FiLM、无几何偏置）
3. **FiLM**（仅条件注入）
4. **几何注意力偏置**（仅探针偏置）
5. **FiLM + 几何偏置**（完整模型）

**层位与偏置形式消融**：
- 仅L5 vs L4+L5 vs L4+L5+U4
- 线性偏置 vs 高斯偏置
- α参数扫描：0.1, 0.3, 0.5, 0.7, 1.0

#### 4.5.2 Vendor椭球体对比

**对比方法**：
- 厂商提供的椭球体预测模型
- 基于功率-时长经验公式的预测

**评价维度**：
- 形状偏差：椭球度、长轴比
- 边界覆盖：真实消融边界覆盖率
- 欠/过消融体积：临床安全性评估

#### 4.5.3 预期结果分析

**性能提升**（需要实际数据填充）：
| 方法 | Dice (%) | 安全边界覆盖 (%) | 欠消融体积 (cm³) | 过消融体积 (cm³) |
|------|----------|------------------|------------------|------------------|
| UNet-3D | [数据] | [数据] | [数据] | [数据] |
| +Diffusion | [数据] | [数据] | [数据] | [数据] |
| +FiLM | [数据] | [数据] | [数据] | [数据] |
| +Probe Bias | [数据] | [数据] | [数据] | [数据] |
| 完整模型 | [数据] | [数据] | [数据] | [数据] |
| Vendor椭球 | [数据] | [数据] | [数据] | [数据] |

**结果解读**：
1. **三维拓扑一致性**：沿探针轴的分割连续性
2. **参数可控性**：功率/时长与消融形状的单调关系
3. **计算效率**：推理时间与精度的平衡
4. **临床适用性**：与现有工作流程的兼容性

---

## 实验实施计划

### 阶段1：3D DDPM预训练（4周）
- 数据预处理和增强
- 3D UNet-DDPM模型训练
- 基线模型对比实验

### 阶段2：消融预测模型开发（6周）
- FiLM条件注入模块实现
- 探针距离引导注意力偏置
- 端到端微调训练

### 阶段3：实验验证与优化（4周）
- 消融实验和参数调优
- Vendor模型对比
- 临床数据验证

### 阶段4：结果分析与论文撰写（2周）
- 实验结果统计
- 图表制作
- 论文初稿完成

---

## 数据需求说明

### 需要收集的数据：
1. **LIDC-IDRI数据集**：1018例肺部CT
2. **院内肺部分割数据**：200例高分辨率CT
3. **消融手术数据**：100例术前/术中/术后CT
4. **探针轨迹数据**：实时位置追踪
5. **功率时长日志**：消融参数记录

### 需要制作的图表：
1. **模型架构图**：3D UNet-DDPM + FiLM + 注意力偏置
2. **训练曲线**：损失函数收敛曲线
3. **分割结果可视化**：3D渲染对比图
4. **消融实验柱状图**：各方法性能对比
5. **参数响应曲线**：功率/时长与体积关系
6. **注意力热力图**：探针位置引导的可视化
7. **ROC曲线**：安全边界覆盖性能
8. **计算效率对比**：推理时间vs精度权衡

---

*注：本文档中的[数据]标记需要在实际实验中填充具体的数值结果。*