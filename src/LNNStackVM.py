import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from torchdiffeq import odeint
from torch.utils.data import TensorDataset, DataLoader
import ctypes
import os

# ===== 1. Data Preparation =====
def prepare_data(N=1024, train_ratio=0.8):
    """Generate synthetic data for function approximation."""
    x = torch.linspace(0, 2 * math.pi, N, dtype=torch.float32).unsqueeze(1)  # Shape: [N, 1]
    y = torch.sin(x) + 0.5 * torch.sin(3 * x)                              # Shape: [N, 1]
    
    perm = torch.randperm(N)
    train_idx = perm[:int(train_ratio * N)]
    test_idx = perm[int(train_ratio * N):]
    
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    
    print(f"Training set size: {x_train.shape[0]}, Test set size: {x_test.shape[0]}")
    return x, y, x_train, y_train, x_test, y_test

# ===== 2. LTC Model Definition =====
class LTCCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gating mechanism weights
        self.W_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # ODE weights
        self.W = nn.Linear(input_size + hidden_size, hidden_size)
        # Time constant (positive)
        self.log_tau = nn.Parameter(torch.randn(hidden_size))
        
    def forward(self, t, h, x):
        """Compute dh/dt for the ODE."""
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        combined = torch.cat([x, h], dim=1)
        tau = torch.exp(self.log_tau).clamp(min=1e-3)
        gate = torch.sigmoid(self.W_gate(combined))
        dhdt = (-h + self.W(combined) * gate) / tau
        return dhdt

class LTC(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.t_eval = torch.linspace(0, 5, 10, dtype=torch.float32)
        self.cell = LTCCell(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """Forward pass through the LTC model."""
        if x.dim() == 1:
            x = x.unsqueeze(1)
        batch_size = x.size(0)
        
        h0 = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32)
        
        def ode_func(t, h):
            return self.cell(t, h, x)
        
        h = odeint(ode_func, h0, self.t_eval, method='euler')[-1]
        return self.fc_out(h)

# ===== 3. Training Function =====
def train_model(model, x_train, y_train, x_test, y_test, epochs=2000, lr=0.001, batch_size=256):
    """Train the LTC model with mini-batching and normalization."""
    x_mean, x_std = x_train.mean(), x_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()
    print(f"Normalization: x_mean={x_mean:.4f}, x_std={x_std:.4f}, y_mean={y_mean:.4f}, y_std={y_std:.4f}")
    
    x_train_norm = (x_train - x_mean) / x_std
    x_test_norm = (x_test - x_mean) / x_std
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    train_dataset = TensorDataset(x_train_norm, y_train_norm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(train_dataset)
        
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(x_test_norm)
                test_loss = criterion(test_pred, y_test_norm)
            print(f"Epoch {epoch:5d} | Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss.item():.4f} | Time: {time.time() - start_time:.2f}s")
    
    def predict(x):
        x_norm = (x - x_mean) / x_std
        with torch.no_grad():
            return model(x_norm) * y_std + y_mean
    
    return predict, x_mean.item(), x_std.item(), y_mean.item(), y_std.item()

# ===== 4. Bytecode Translator =====
PUSH_CONST, MATMUL, ADD, SIGMOID, ODE_SOLVE, RETURN = (0x01, 0x02, 0x03, 0x04, 0x05, 0xFF)

def translate_model(model):
    """Translate LTC model to bytecode and constant pool."""
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
    
    print(f"Weight shapes: W_gate={const_pool[w_gate_idx].shape}, W={const_pool[w_idx].shape}, W_out={const_pool[w_out_idx].shape}")
    
    code += bytes([ODE_SOLVE, w_gate_idx, b_gate_idx, w_idx, b_idx, tau_idx, w_out_idx, b_out_idx, RETURN])
    
    np.savez_compressed("./output/const_pool.npz", *const_pool)
    with open("./output/bytecode.bin", "wb") as f:
        f.write(code)
    
    print(f"✓ Compilation complete: {len(code)} bytes, {len(const_pool)} constants")
    return const_pool, code, model.t_eval.cpu().numpy()

# ===== 5. Python Stack Virtual Machine (Original) =====
class VMState:
    def __init__(self):
        self.hidden_state = None

# ===== 6. C Stack Virtual Machine Interface =====
def vm_run_c(x_input, x_mean, x_std, y_mean, y_std, consts, t_eval):
    """Run the LTC model on the C-based stack VM."""
    # Load the shared library
    lib = ctypes.cdll.LoadLibrary("./output/lib_vm.so")
    
    # Define the function signature
    lib.vm_run.restype = ctypes.c_float
    lib.vm_run.argtypes = [
        ctypes.c_float,  # x_input
        ctypes.c_float,  # x_mean
        ctypes.c_float,  # x_std
        ctypes.c_float,  # y_mean
        ctypes.c_float,  # y_std
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # const_data
        ctypes.POINTER(ctypes.c_size_t),  # const_rows
        ctypes.POINTER(ctypes.c_size_t),  # const_cols
        ctypes.c_size_t,  # const_count
        ctypes.c_char_p,  # bytecode_file
        ctypes.POINTER(ctypes.c_float),  # t_eval
        ctypes.c_size_t  # t_eval_len
    ]
    
    # Prepare constants
    const_count = len(consts)
    const_data = []
    const_rows = (ctypes.c_size_t * const_count)()
    const_cols = (ctypes.c_size_t * const_count)()
    
    for i, c in enumerate(consts):
        # Ensure contiguous array
        c_contig = np.ascontiguousarray(c, dtype=np.float32)
        const_data.append(c_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        const_rows[i] = c.shape[0]
        const_cols[i] = c.shape[1] if c.ndim > 1 else 1
    
    const_data_array = (ctypes.POINTER(ctypes.c_float) * const_count)(*const_data)
    
    # Prepare t_eval
    t_eval_contig = np.ascontiguousarray(t_eval, dtype=np.float32)
    t_eval_ptr = t_eval_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Call the C function
    bytecode_file = b"./output/bytecode.bin"
    result = lib.vm_run(
        ctypes.c_float(x_input),
        ctypes.c_float(x_mean),
        ctypes.c_float(x_std),
        ctypes.c_float(y_mean),
        ctypes.c_float(y_std),
        const_data_array,
        const_rows,
        const_cols,
        ctypes.c_size_t(const_count),
        bytecode_file,
        t_eval_ptr,
        ctypes.c_size_t(len(t_eval))
    )
    
    if np.isnan(result):
        raise ValueError("C VM returned NaN")
    
    return result

# ===== 7. Main Program =====
if __name__ == "__main__":
    torch.set_num_threads(8)
    
    # Prepare data and train model
    x, y, x_train, y_train, x_test, y_test = prepare_data(N=1024)
    model = LTC(hidden_size=64)
    predict, x_mean, x_std, y_mean, y_std = train_model(model, x_train, y_train, x_test, y_test, epochs=2000, lr=0.001, batch_size=256)
    
    # Translate model to bytecode
    try:
        consts, bytecode, t_eval = translate_model(model)
        bytecode = memoryview(bytecode)
    except Exception as e:
        print(f"Bytecode generation error: {e}")
        exit(1)
    
    # Validate outputs
    sample_x = 1.234
    with torch.no_grad():
        pt_out = predict(torch.tensor([[sample_x]], dtype=torch.float32)).item()
    
    state = VMState()
    
    # Run C VM
    try:
        c_out = vm_run_c(sample_x, x_mean, x_std, y_mean, y_std, consts, t_eval)
        print(f"C VM Output: {c_out:.6f}")
        print(f"Absolute Difference (PyTorch vs C VM): {abs(pt_out - c_out):.6f}")
    except Exception as e:
        print(f"C VM error: {e}")
    
    # Benchmark
    def bench(fn, n_iter=1000):
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        return (time.perf_counter() - t0) / n_iter * 1e6
    
    try:
        pt_us = bench(lambda: predict(torch.tensor([[sample_x]], dtype=torch.float32)).item())
        c_us = bench(lambda: vm_run_c(sample_x, x_mean, x_std, y_mean, y_std, consts, t_eval))
        print(f"\nPerformance Test:")
        print(f"PyTorch: {pt_us:.2f} µs/call")
        print(f"C VM: {c_us:.2f} µs/call")
    except Exception as e:
        print(f"Performance test error: {e}")
    
    # Plot results
    with torch.no_grad():
        y_pred_torch = predict(x).numpy().squeeze()
    
    state = VMState()
    y_pred_c = []
    for xi in x.numpy().flatten():
        y_pred_c.append(vm_run_c(xi, x_mean, x_std, y_mean, y_std, consts, t_eval))
    y_pred_c = np.array(y_pred_c)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x.numpy(), y.numpy(), 'k-', lw=2, label='True Function')
    plt.plot(x.numpy(), y_pred_torch, 'r--', lw=1.5, label='PyTorch Prediction')
    plt.plot(x.numpy(), y_pred_c, 'g-.', lw=1.5, label='C VM Prediction')
    plt.xlabel('Input x')
    plt.ylabel('Output y')
    plt.title('Function Approximation with LTC')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('./image/ltc_fit_result_c.png')
    plt.close()