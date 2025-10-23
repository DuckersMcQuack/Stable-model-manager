# Advanced Stable Diffusion Model Cross-Referencing System

## 🎯 Overview

This is a sophisticated model and media file organization system with advanced cross-referencing capabilities. It can intelligently sort stable diffusion models and related media files based on metadata analysis, proximity detection, and external database cross-referencing.

## ✨ Key Features

### 🔍 Advanced Scanning & Detection
- **Multi-hash system**: SHA256 (exact matching) + AutoV3 (SafeTensors weights) + BlurHash (visual similarity)
- **Comprehensive file type detection**: Models, LoRAs, VAEs, Embeddings, ControlNets, Images, Videos
- **Model format support**: .safetensors, .ckpt, .pt, .pth, .bin files
- **Media format support**: .png, .jpg, .webp, .mp4, .gif files

### 🎨 Metadata Extraction & Analysis
- **PNG Parameters**: Steps, Sampler, CFG Scale, Seed, Model Hash, LoRA usage
- **EXIF Data**: Camera settings, creation time, software used
- **Generation Tool Detection**: Automatic1111, ComfyUI, NovelAI, etc.
- **Component Detection**: Automatically identifies LoRAs, LyCORIS, VAEs, ControlNets
- **Weight Analysis**: Extracts component weights and usage context

### 🔗 Cross-Referencing System
- **Proximity Detection**: Media files next to models are kept together
- **Metadata Text Files**: Reads .txt files with generation parameters
- **Civitai Integration**: Cross-references with civitai.sqlite database
- **Hash Matching**: SHA256 and BlurHash matching for orphaned files
- **Component Mapping**: Links media to specific LoRAs/models used

### 📁 Intelligent Sorting
- **Organized Structure**: `models/checkpoints/SD1.5/model_name/`
- **Component Folders**: `models/loras/`, `models/vae/`, etc.
- **Duplicate Handling**: `duplicates/media/lora/` categorization
- **Proximity Preservation**: Media stays with associated models
- **Configurable Layouts**: Customize folder structures

## 🚀 Getting Started

### Prerequisites
```bash
pip install pillow configparser
```

### Configuration
Edit `config.ini`:
```ini
[DEFAULT]
source_directory = /path/to/your/models
destination_directory = ./sorted_models
database_path = Database/file_scanner.sqlite
civitai_database_path = Database/civitai.sqlite
verbose = true
max_file_size = 10737418240
use_model_type_subfolders = true

model_type_folders = checkpoint,loras,embedding,textualinversion,hypernetwork,controlnet,other
```

## 📖 Usage Examples

### Basic Scanning
```bash
# Scan a directory for new files
python file_scanner.py --scan-directory /path/to/models --verbose

# Force re-scan all files (ignores cache)
python file_scanner.py --scan-directory /path/to/models --force-rescan
```

### Advanced Cross-Referencing
```bash
# Process orphaned media files with intelligent cross-referencing
python file_scanner.py --process-orphaned --verbose

# Dry run to see what would happen (recommended first)
python file_scanner.py --process-orphaned --dry-run --verbose

# Extract metadata from media files
python file_scanner.py --extract-metadata --verbose
```

### Multiple Operations
```bash
# Comprehensive processing: scan + extract + process
python file_scanner.py --scan-directory /new/models --extract-metadata --process-orphaned --verbose
```

## 🧠 Cross-Referencing Logic

The system follows this intelligent workflow:

### 1. **Proximity Detection**
- Media files found next to model files are kept together
- No cross-referencing performed - preserves existing organization

### 2. **Embedded Metadata Analysis**
- Extracts model hash from PNG parameters
- Identifies generation tool (A1111, ComfyUI, etc.)
- Detects LoRA usage: `<lora:model_name:0.8>`
- Maps to local model database

### 3. **Text File Parsing**
- Looks for `.txt` files with same name as media
- Parses generation parameters and model references
- Extracts LoRA, checkpoint, VAE mentions
- Cross-references with local model database

### 4. **Civitai Cross-Referencing**
- For orphaned files with no metadata
- Uses SHA256 + BlurHash matching
- Queries civitai.sqlite database
- Attempts to find related models

### 5. **Intelligent Sorting**
```
✅ Match found → models/loras/model_name/image.png
🔄 Duplicate exists → duplicates/media/lora/image.png
❌ No match → marked as ignored (not moved)
```

## 📊 Database Schema

### Core Tables
- **scanned_files**: All discovered files with hashes
- **media_metadata**: Comprehensive metadata (16+ fields)
- **component_usage**: LoRA/VAE usage tracking
- **model_components**: Cross-reference registry

### Example Metadata Fields
- `width`, `height`: Image dimensions
- `model_name`, `model_hash`: Generation model
- `sampler`, `steps`, `cfg_scale`: Generation settings
- `seed`: Random seed used
- `generation_tool`: A1111, ComfyUI, etc.
- `blur_hash`: Visual similarity hash
- `raw_parameters`: Full parameter string

## 🛡️ Safety Features

### Dry Run Mode
```bash
python file_scanner.py --process-orphaned --dry-run
```
- Shows exactly what would be moved
- No files are actually relocated
- Detailed logging of decisions

### Duplicate Protection
- Files are never overwritten
- Duplicates go to categorized folders
- Original locations preserved in database

### Reversible Operations
- Database tracks all file movements
- Original paths maintained
- Operations can be analyzed and undone

## 🔧 Advanced Configuration

### Model Type Folders
```ini
model_type_folders = checkpoint,loras,embedding,textualinversion,hypernetwork,controlnet,vae,upscaler,other
```

### Custom Folder Structure
```ini
# Enable/disable subfolder organization
use_model_type_subfolders = true

# Result: models/checkpoints/SD1.5/model_name/
# Disable: models/SD1.5/model_name/
```

### Performance Tuning
```ini
# Maximum file size to process (10GB default)
max_file_size = 10737418240

# Enable detailed logging
verbose = true
```

## 📈 Component Detection Examples

### LoRA Detection
```
Input: "masterpiece, <lora:choco-pynoise-000012:1.0>, detailed"
Output: LoRA 'choco-pynoise-000012' @ weight 1.0
```

### Generation Parameters
```
Steps: 30, Sampler: Euler a, CFG scale: 9.0
Model hash: abc123def, Tool: Automatic1111
```

### Text File Parsing
```
File: image.txt
Content: "Used model: realistic_vision_v2.safetensors
LoRA: add_detail:0.5, more_art:0.8"
```

## 🚨 Troubleshooting

### Common Issues

1. **No civitai.sqlite found**
   - Cross-referencing limited to local data
   - Download civitai database separately

2. **Permission errors**
   - Ensure write access to destination directory
   - Check file ownership and permissions

3. **Large file processing**
   - Adjust `max_file_size` in config
   - Use `--verbose` to track progress

### Debug Mode
```bash
python file_scanner.py --process-orphaned --verbose --dry-run
```

## 📚 API Reference

### Key Functions

#### `process_orphaned_media(media_files, dry_run=False)`
Main cross-referencing function

#### `cross_reference_with_civitai(sha256_hash, blur_hash, civitai_db_path)`
External database matching

#### `extract_comprehensive_metadata(file_path)`
Complete metadata extraction

#### `find_metadata_text_file(image_path)`
Locate associated text files

## 🤝 Contributing

This system is designed to be extensible:
- Add new model formats in file detection
- Extend metadata extraction for new tools  
- Enhance cross-referencing algorithms
- Improve sorting logic

## 📄 License

Open source - feel free to modify and extend for your needs.

---

**🎯 Perfect for**: Model collectors, AI artists, dataset organizers, and anyone managing large collections of stable diffusion models and generated content.