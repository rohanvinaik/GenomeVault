# AGENTS.md

This file provides guidance to LLM agents (e.g., ChatGPT, Claude, etc.) when working with this repository.

## Interaction Guidelines

- Review `CLAUDE.md` for detailed project overview and common commands
- Make small, incremental changes rather than sweeping rewrites
- Maintain existing code style and module boundaries
- Reference files and line numbers in explanations and commit messages
- Never commit secrets, credentials, or real genomic data

## Development Workflow

1. Ensure the working tree is clean (`git status --short`)
2. Add tests under `tests/` when implementing new features or fixing bugs
3. Run programmatic checks before committing:
   ```bash
   ruff check .
   pytest
   ```
   If checks fail, describe the failure and partial progress in the PR
4. Commit with clear, descriptive messages

## Coding Style

- Follow settings in `pyproject.toml` and `.ruff.toml` (Python 3.11+, line length 100)
- Use type hints and docstrings for new functions
- Prefer composition and small focused functions

## Documentation

- Update relevant README or module documentation for behavioral changes
- Use Markdown headings and bullet lists for clarity

## Security

- Use placeholder values for examples
- Do not commit private keys, tokens, or sensitive genomic data

## Pull Request Expectations

- Summarize the change and its motivation
- Include the commands and results of tests and linters
- Leave the repository in a clean state (`git status --short` should show no changes)
