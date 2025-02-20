import torch
import torch.nn as nn

class Quantizer:
    def __init__(self, model, bits=8):
        self.model = model
        self.bits = bits
        self.scale = 2**(bits - 1) - 1

    def quantize_linear(self, tensor):
        # Simple symmetric linear quantization
        max_val = tensor.abs().max()
        if max_val == 0:
            return tensor
        
        # Scale and round to integer values
        quantized_tensor = torch.round(tensor / max_val * self.scale)
        
        # Dequantize back to float
        dequantized_tensor = quantized_tensor / self.scale * max_val
        return dequantized_tensor

    def quantize_model(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"Quantizing layer: {name}")
                # Quantize weights
                module.weight.data = self.quantize_linear(module.weight.data)
                if module.bias is not None:
                    module.bias.data = self.quantize_linear(module.bias.data)
        print("Model quantization complete.")
        return self.model

class LLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, hidden_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    # Example usage
    vocab_size = 10000
    embed_dim = 256
    num_layers = 2
    num_heads = 8
    hidden_dim = 512

    # Create a dummy LLM
    model = LLM(vocab_size, embed_dim, num_layers, num_heads, hidden_dim)
    print("Original model weights (first layer):
", model.transformer_blocks[0].linear1.weight.data[:2, :2])

    # Quantize the model
    quantizer = Quantizer(model, bits=8)
    quantized_model = quantizer.quantize_model()
    print("Quantized model weights (first layer):
", quantized_model.transformer_blocks[0].linear1.weight.data[:2, :2])

    # Simulate some input
    dummy_input = torch.randint(0, vocab_size, (1, 10)) # Batch size 1, sequence length 10
    output = quantized_model(dummy_input)
    print("
Output shape after quantization:", output.shape)
