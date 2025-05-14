import torch
import torch.nn as nn
import numpy as np
import time
import os
from torchdiffeq import odeint

# 定义字节码指令集
PUSH_CONST, MATMUL, ADD, SIGMOID, ODE_SOLVE, RETURN = (0x01, 0x02, 0x03, 0x04, 0x05, 0xFF)

# 获取项目根目录的路径
def get_project_root():
    # 当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录（假设当前文件在src目录下）
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    return project_root

# 确保输出目录存在
def ensure_output_dir():
    project_root = get_project_root()
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(project_root, "image")
    os.makedirs(image_dir, exist_ok=True)
    return output_dir, image_dir

# 初始化输出目录
OUTPUT_DIR, IMAGE_DIR = ensure_output_dir()

class JITODESolver(nn.Module):
    """JIT编译友好的ODE求解器"""
    def __init__(self, const_pool):
        super().__init__()
        self.w_gate = nn.Parameter(torch.tensor(const_pool[0], dtype=torch.float32))
        self.b_gate = nn.Parameter(torch.tensor(const_pool[1], dtype=torch.float32))
        self.w = nn.Parameter(torch.tensor(const_pool[2], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(const_pool[3], dtype=torch.float32))
        self.tau = nn.Parameter(torch.tensor(const_pool[4], dtype=torch.float32))
        self.w_out = nn.Parameter(torch.tensor(const_pool[5], dtype=torch.float32))
        self.b_out = nn.Parameter(torch.tensor(const_pool[6], dtype=torch.float32))
    
    def ode_func(self, t, h, x):
        """计算ODE的dh/dt"""
        # 确保输入形状正确
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 组合输入和隐藏状态
        combined = torch.cat([x, h], dim=1)
        
        # 计算门控机制
        gate_input = torch.matmul(combined, self.w_gate) + self.b_gate
        gate = torch.sigmoid(gate_input)
        
        # 计算ODE
        f = torch.matmul(combined, self.w) + self.b
        dhdt = (-h + f * gate) / self.tau
        return dhdt
    
    # 使用TorchScript支持的方式实现ode_func_wrapper
    def forward(self, x, t_eval):
        """前向传递"""
        if x.dim() == 1:
            x = x.unsqueeze(1)
        batch_size = x.size(0)
        hidden_size = self.w_out.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(batch_size, hidden_size, dtype=torch.float32)
        
        # 使用odeint求解ODE
        # 注意：我们使用闭包方式访问x，这在TorchScript中是支持的
        h_final = self._solve_ode(h0, x, t_eval)
        
        # 输出层
        return torch.matmul(h_final, self.w_out) + self.b_out
    
    def _solve_ode(self, h0, x, t_eval):
        """求解ODE，避免使用嵌套函数"""
        # 存储中间状态
        h = h0
        
        # 简化版的欧拉方法求解ODE
        dt = t_eval[1] - t_eval[0] if len(t_eval) > 1 else torch.tensor(0.1)
        
        for t_idx in range(len(t_eval) - 1):
            dhdt = self.ode_func(t_eval[t_idx], h, x)
            h = h + dhdt * dt
            
        return h

class JITStackVM:
    def __init__(self, const_pool, bytecode, t_eval):
        """初始化JIT Stack VM
        
        Args:
            const_pool: 常量池列表
            bytecode: 字节码
            t_eval: 评估时间点
        """
        self.const_pool = const_pool
        self.bytecode = bytecode
        self.t_eval = torch.tensor(t_eval, dtype=torch.float32)
        
        # 创建JIT编译版本的ODE求解器
        self.jit_ode_solver = self._create_jit_ode_solver()
        
    def _create_jit_ode_solver(self):
        """创建JIT编译版本的ODE求解器"""
        model = JITODESolver(self.const_pool)
        return torch.jit.script(model)
    
    def run(self, x_input, x_mean, x_std, y_mean, y_std):
        """运行JIT Stack VM
        
        Args:
            x_input: 输入值，可以是标量或张量
            x_mean: 输入均值
            x_std: 输入标准差
            y_mean: 输出均值
            y_std: 输出标准差
            
        Returns:
            标量或张量结果
        """
        # 转换输入为张量
        if isinstance(x_input, (int, float)):
            x = torch.tensor([[x_input]], dtype=torch.float32)
        else:
            x = torch.tensor(x_input, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(1)
        
        # 标准化输入
        x_norm = (x - x_mean) / x_std
        
        # 通过JIT编译的模型运行
        with torch.no_grad():
            y_norm = self.jit_ode_solver(x_norm, self.t_eval)
        
        # 反标准化输出
        y = y_norm * y_std + y_mean
        
        # 如果输入是标量，则返回标量
        if isinstance(x_input, (int, float)):
            return y.item()
        return y

# 辅助函数，从模型创建JIT Stack VM
def create_jit_vm_from_model(model, x_mean, x_std, y_mean, y_std):
    """从模型创建JIT Stack VM
    
    Args:
        model: LTC模型
        x_mean: 输入均值
        x_std: 输入标准差
        y_mean: 输出均值
        y_std: 输出标准差
        
    Returns:
        jit_vm: JIT Stack VM实例
        predict_fn: 预测函数
    """
    try:
        # 翻译模型到字节码
        const_pool, bytecode, t_eval = translate_model(model)
        bytecode = memoryview(bytecode)
        
        # 创建JIT Stack VM
        jit_vm = JITStackVM(const_pool, bytecode, t_eval)
        
        # 创建预测函数
        def predict(x):
            return jit_vm.run(x, x_mean, x_std, y_mean, y_std)
        
        return jit_vm, predict
    except Exception as e:
        print(f"创建JIT VM错误: {e}")
        raise

def translate_model(model):
    """翻译LTC模型到字节码和常量池"""
    const_pool, code = [], bytearray()
    
    def add_const(t):
        const_pool.append(t.detach().cpu().numpy().astype(np.float32))
        return len(const_pool) - 1
    
    cell = model.cell
    w_gate_idx = add_const(cell.W_gate.weight.t())  # Shape: [65, 64]
    b_gate_idx = add_const(cell.W_gate.bias)       # Shape: [64]
    w_idx = add_const(cell.W.weight.t())           # Shape: [65, 64]
    b_idx = add_const(cell.W.bias)                 # Shape: [64]
    tau_idx = add_const(torch.exp(cell.log_tau).clamp(min=1e-3))  # Shape: [64]
    w_out_idx = add_const(model.fc_out.weight.t()) # Shape: [64, 1]
    b_out_idx = add_const(model.fc_out.bias)       # Shape: [1]
    
    print(f"权重形状: W_gate={const_pool[w_gate_idx].shape}, W={const_pool[w_idx].shape}, W_out={const_pool[w_out_idx].shape}")
    
    code += bytes([ODE_SOLVE, w_gate_idx, b_gate_idx, w_idx, b_idx, tau_idx, w_out_idx, b_out_idx, RETURN])
    
    # 保存常量池和字节码（可选）
    const_pool_path = os.path.join(OUTPUT_DIR, "const_pool.npz")
    bytecode_path = os.path.join(OUTPUT_DIR, "bytecode.bin")
    
    np.savez_compressed(const_pool_path, *const_pool)
    with open(bytecode_path, "wb") as f:
        f.write(code)
    
    print(f"✓ 编译完成: {len(code)} 字节, {len(const_pool)} 常量")
    return const_pool, code, model.t_eval.cpu().numpy()

# 示例使用
if __name__ == "__main__":
    # 需要导入原始模型相关定义
    from LNNStackVM import prepare_data, LTC, train_model
    
    # 准备数据和训练模型
    x, y, x_train, y_train, x_test, y_test = prepare_data(N=1024)
    model = LTC(hidden_size=64)
    predict_orig, x_mean, x_std, y_mean, y_std = train_model(model, x_train, y_train, x_test, y_test, epochs=2000, lr=0.001, batch_size=256)
    
    # 创建JIT Stack VM
    jit_vm, predict_jit = create_jit_vm_from_model(model, x_mean, x_std, y_mean, y_std)
    
    # 验证输出
    sample_x = 1.234
    with torch.no_grad():
        pt_out = predict_orig(torch.tensor([[sample_x]], dtype=torch.float32)).item()
    
    jit_out = predict_jit(sample_x)
    print(f"PyTorch输出: {pt_out:.6f}")
    print(f"JIT VM输出: {jit_out:.6f}")
    print(f"绝对差异 (PyTorch vs JIT VM): {abs(pt_out - jit_out):.6f}")
    
    # 性能测试
    def bench(fn, n_iter=1000):
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        return (time.perf_counter() - t0) / n_iter * 1e6
    
    try:
        pt_us = bench(lambda: predict_orig(torch.tensor([[sample_x]], dtype=torch.float32)).item())
        jit_us = bench(lambda: predict_jit(sample_x))
        print(f"\n性能测试:")
        print(f"PyTorch: {pt_us:.2f} µs/调用")
        print(f"JIT VM: {jit_us:.2f} µs/调用")
        print(f"加速比: {pt_us/jit_us:.2f}x")
    except Exception as e:
        print(f"性能测试错误: {e}")