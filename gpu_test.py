"""
GPU Neural Network Test
Tests: GPU detection, training speed, memory usage
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time

print("=" * 60)
print("  GPU Neural Network Test")
print("=" * 60)

# 1. GPU Info
print("\n[1] GPU Information")
print(f"  PyTorch version : {torch.__version__}")
print(f"  CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version    : {torch.version.cuda}")
    print(f"  GPU count       : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}           : {torch.cuda.get_device_name(i)}")
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  Total memory    : {mem:.1f} GB")
else:
    print("  [FAIL] No GPU detected! Falling back to CPU.")
    device = torch.device("cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device    : {device}")

# 2. Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

# 3. Training test
print("\n[2] Training Test (MNIST-like simulation)")
BATCH_SIZE = 256
NUM_EPOCHS = 20
NUM_BATCHES = 100

model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {total_params:,}")
print(f"  Batch size      : {BATCH_SIZE}")
print(f"  Epochs          : {NUM_EPOCHS}")
print(f"  Batches/epoch   : {NUM_BATCHES}")
print()

# Warm-up
dummy_data = torch.randn(BATCH_SIZE, 784, device=device)
dummy_labels = torch.randint(0, 10, (BATCH_SIZE,), device=device)
_ = model(dummy_data)
if device.type == "cuda":
    torch.cuda.synchronize()

# Training loop
start_time = time.time()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for _ in range(NUM_BATCHES):
        x = torch.randn(BATCH_SIZE, 784, device=device)
        y = torch.randint(0, 10, (BATCH_SIZE,), device=device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / NUM_BATCHES
    if (epoch + 1) % 5 == 0 or epoch == 0:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

if device.type == "cuda":
    torch.cuda.synchronize()
total_time = time.time() - start_time

# 4. Results
print("\n[3] Results")
print(f"  Total training time : {total_time:.2f}s")
print(f"  Throughput          : {NUM_EPOCHS * NUM_BATCHES * BATCH_SIZE / total_time:,.0f} samples/sec")

if device.type == "cuda":
    mem_used = torch.cuda.memory_allocated() / 1024**2
    mem_reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"  GPU memory used     : {mem_used:.1f} MB")
    print(f"  GPU memory reserved : {mem_reserved:.1f} MB")
    try:
        util = torch.cuda.utilization()
        print(f"  GPU utilization     : {util}%")
    except Exception:
        pass

print("\n" + "=" * 60)
if device.type == "cuda":
    print("  RESULT: GPU is working correctly!")
else:
    print("  RESULT: Running on CPU only (no GPU)")
print("=" * 60)
