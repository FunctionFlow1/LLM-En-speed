# LLM-En-speed

LLM-En-speed is a collection of tools for fine-tuning and quantizing Large Language Models (LLMs) for edge devices. It provides a streamlined workflow for optimizing LLMs for performance and efficiency, ensuring they can be deployed on resource-constrained devices.

## Key Features

- **Fine-tuning Support:** Efficient fine-tuning of LLMs for specific tasks and domains.
- **Quantization Techniques:** Advanced quantization methods for reducing model size and improving inference speed.
- **Edge Device Optimization:** Tailored optimizations for popular edge devices, such as mobile phones and embedded systems.
- **Easy Integration:** Seamless integration with popular LLM frameworks and libraries.
- **Comprehensive Documentation:** Detailed guides and examples for using the toolkit.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.15+
- (Optional) NVIDIA GPU with CUDA support for enhanced performance

### Installation

```bash
git clone https://github.com/FunctionFlow1/LLM-En-speed.git
cd LLM-En-speed
pip install -r requirements.txt
```

### Usage Example (Python)

```python
import llm_en_speed as les

# Initialize the LLM-En-speed toolkit
toolkit = les.LLMEnSpeed(config_path='config.yaml')

# Fine-tune an LLM for a specific task
toolkit.fine_tune(model_name='gpt-2', dataset_path='data.txt')

# Quantize the fine-tuned model for edge devices
toolkit.quantize(model_path='fine_tuned_model.pt', output_path='quantized_model.pt')

# Deploy the quantized model on an edge device
toolkit.deploy(model_path='quantized_model.pt', device='mobile')
```

## Contributing

We welcome contributions from the community! Please read our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## License

LLM-En-speed is released under the [MIT License](LICENSE).
