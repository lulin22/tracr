# Encryption Modes Documentation

## Overview

The system supports two encryption modes for secure tensor processing:

1. **Transmission Mode** - Encrypts data for secure network transmission, decrypts on server before processing
2. **Full Mode** - Full homomorphic encryption where the server processes encrypted tensors without decryption

## Encryption Modes

### Transmission Mode (Default)

This is the standard encryption mode that provides security during network transmission:

**Workflow:**
1. **Client**: Serialize → Encrypt → Compress → Send
2. **Server**: Receive → Decompress → Decrypt → Process → Encrypt → Compress → Send  
3. **Client**: Receive → Decompress → Decrypt

**Characteristics:**
- Data is encrypted during transmission
- Server decrypts data before processing
- Supports all PyTorch models
- No computational overhead during inference
- Currently fully functional

### Full Mode (Homomorphic Encryption)

This mode enables processing on encrypted data without decryption:

**Workflow:**
1. **Client**: Serialize → Encrypt → Compress → Send
2. **Server**: Receive → Decompress → Process Encrypted → Compress → Send
3. **Client**: Receive → Decompress → Decrypt

**Characteristics:**
- Server never sees decrypted data
- Requires special homomorphic-compatible models
- Higher computational overhead
- Infrastructure implemented but requires model adaptations

## Configuration

### YAML Configuration

```yaml
# Transmission mode (default)
encryption:
  enabled: true
  mode: "transmission"  # Server decrypts before processing
  password: "your_password"
  degree: 8192
  scale: 26

# Full mode (homomorphic)
encryption:
  enabled: true
  mode: "full"  # Server processes encrypted data
  password: "your_password"
  degree: 8192
  scale: 26
```

### Mode Selection

The encryption mode is specified in the configuration file:
- `mode: "transmission"` - Use transmission mode (default if not specified)
- `mode: "full"` - Use full homomorphic encryption mode

## Implementation Details

### Code Changes

The encryption mode is handled by the `DataCompression` class:

```python
# Initialize with encryption mode
compressor = DataCompression(
    config, 
    encryption=encryption_module,
    encryption_mode="transmission"  # or "full"
)
```

### Server Behavior

- **Transmission Mode**: Server calls `decompress_data()` normally, which decrypts after decompression
- **Full Mode**: Server would call `decompress_data(skip_decryption=True)` to get encrypted tensors

## Current Status

### Working
- ✅ Transmission mode fully functional
- ✅ Infrastructure for full mode implemented
- ✅ Mode selection via configuration

### Limitations of Full Mode
- ❌ Standard PyTorch models cannot process TenSEAL encrypted tensors
- ❌ Requires custom homomorphic model implementations
- ❌ Limited to basic operations (add, multiply)

## Security Comparison

| Aspect | Transmission Mode | Full Mode |
|--------|------------------|-----------|
| Network Security | ✅ Encrypted | ✅ Encrypted |
| Server Data Access | Decrypted | Never Decrypted |
| Model Compatibility | All Models | Special Models Only |
| Performance | Fast | Slow |
| Implementation | Complete | Partial |

## Recommendations

1. **For Production Use**: Use transmission mode for secure inference with existing models
2. **For Development**: Full mode infrastructure is ready for experimentation with homomorphic models
3. **For Maximum Security**: Full mode provides strongest privacy guarantees once fully implemented

## Future Work

To complete full mode implementation:
1. Develop homomorphic-compatible neural network layers
2. Create polynomial approximations for activation functions
3. Optimize homomorphic operations for practical inference speeds
4. Provide conversion tools for existing models 