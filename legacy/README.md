# Legacy ANPR Script

This directory contains the original `anpr-system.py` script that was used during development.

## Files

- `anpr-system.py` - Original monolithic script (397 lines)

## Migration

The functionality has been refactored into the `anpr_system/` package with:
- Modular design (separate files for CLI, web, core, utils)
- Professional packaging (setup.py, pyproject.toml)
- Multiple interfaces (CLI, web, Docker)
- Better error handling and configuration

## Usage

If you need to use the original script:
```bash
python legacy/anpr-system.py --help
```

For new projects, use the package:
```bash
pip install -e .
anpr --help
```
