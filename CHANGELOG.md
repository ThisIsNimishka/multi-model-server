# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release
- Multi-model server manager with GPU allocation
- Unified API router with OpenAI-compatible endpoints
- Support for Qwen2.5-VL-7B (vision-language)
- Support for Mistral-7B-Instruct
- Support for Gemma-7B
- Support for Qwen-72B (all GPUs)
- Health monitoring and metrics
- Windows batch scripts for easy startup
- Test client with automated tests and interactive chat
- Streaming response support
- Smart routing based on model capabilities

### Configuration
- Tensor parallel support for multi-GPU inference
- Configurable GPU memory utilization
- Adjustable context length per model
- Per-model port assignment

## [1.0.0] - 2024-12-23

### Added
- Initial public release
- Core server management functionality
- API routing with load balancing
- Documentation and examples

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2024-12-23 | Initial release |
