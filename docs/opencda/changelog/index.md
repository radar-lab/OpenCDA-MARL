# OpenCDA Changelog

This changelog documents the evolution of OpenCDA within the OpenCDA-MARL project. While we develop
the MARL extension, we also continuously improve and modify the core OpenCDA functionality.

!!! info "Tracking OpenCDA Evolution"
    This changelog tracks all modifications made to the original OpenCDA codebase. Changes are organized by category and version for easy reference.

## Latest Changes

### Version 0.1.3.1 (Current)

-   [Architecture Changes](architecture_changes.md) - Core structural modifications
-   [API Changes](api_changes.md) - Interface and function updates
-   [Platform Changes](platform_changes.md) - Environment and compatibility updates
-   [Config Changes](config_changes.md) - YAML configuration updates
-   [Breaking Changes](breaking_changes.md) - Critical changes requiring action
-   [Migration & Docs](migration_and_docs.md) - Migration guide and documentation changes

## Change Categories

### 🏗️ [Architecture Changes](architecture_changes.md)

Structural modifications to the OpenCDA core including:

-   Version management system
-   Module organization
-   Package structure

### 🔧 [API Changes](api_changes.md)

Updates to interfaces and functions:

-   Command line arguments
-   Function signatures
-   Class modifications

### ⚙️ [Configuration Changes](config_changes.md)

Configuration system updates:

-   YAML structure changes
-   Default parameters
-   Scenario configurations

### 📦 [Platform & Dependencies](platform_changes.md)

Environment and compatibility:

-   Package management (Conda → Pixi)
-   Python version requirements
-   CARLA version updates
-   Platform-specific fixes

### 📚 [Migration & Documentation](migration_and_docs.md)

Migration guide and documentation updates:

-   Step-by-step migration from original OpenCDA
-   Documentation system migration (Sphinx → MkDocs)
-   Common issues and solutions

## Quick Reference

| Change Type        | Impact | Migration Required    |
| ------------------ | ------ | --------------------- |
| Version Management | Low    | Update imports        |
| Command Line       | Medium | Update scripts        |
| Package Manager    | High   | Reinstall environment |
| Documentation      | Low    | New location          |

## Contributing Changes

When making changes to OpenCDA core:

1. Document the change in the appropriate category file
2. Update this index if adding new categories
3. Include migration notes for breaking changes
4. Reference related GitHub issues/PRs

## Migration Guide

For users migrating from original OpenCDA, see:

-   [Migration & Documentation Guide](migration_and_docs.md)
-   [Breaking Changes](breaking_changes.md)
-   [FAQ](../../faq.md)

---

!!! tip "Staying Updated" Subscribe to our
[GitHub releases](https://github.com/lgcyaxi/opencda-marl/releases) to get notified of new changes.
