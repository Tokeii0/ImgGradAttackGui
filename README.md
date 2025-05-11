

### 模型输入恢复工具 (model_recovery_tool.py)

这个工具使用梯度优化方法，通过最大化模型输出来恢复可能的输入图像。适用于图像分类、检测等模型。

主要功能：
- 加载任意PyTorch模型和权重
- 自定义输入图像尺寸
- 支持GPU加速
- 实时显示恢复过程
- 保存恢复结果


## 使用方法

1. 确保已安装所需依赖：
   ```
   pip install torch torchvision numpy Pillow PySide6 matplotlib
   ```

2. 运行相应的工具：
   ```
   python model_recovery_tool.py                        # 图像恢复工具

3. 在界面中配置相应参数后点击"开始恢复"或"开始模拟"

