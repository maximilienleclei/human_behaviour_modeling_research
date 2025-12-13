# Device Configuration

Global PyTorch device management used across all neural network modules.

## Purpose
Provides a single global DEVICE variable that can be configured at runtime to use CPU or specific GPU.

## Contents
- `DEVICE` - Global torch.device (defaults to cuda:0 if available, else cpu)
- `set_device()` - Function to set DEVICE to specific GPU or CPU
