---
name: Release checklist
about: Checklist for maintainers when preparing a release
title: "Release v"
labels: release
assignees: ''
---

## Release Checklist for v[X.Y.Z]

### Pre-release

- [ ] All tests passing on main branch
- [ ] Documentation is up to date
- [ ] CHANGELOG.md is updated with all changes
- [ ] Version number updated in `pyproject.toml`
- [ ] All deprecation warnings are addressed
- [ ] Breaking changes are documented

### Code Quality

- [ ] All pre-commit hooks pass
- [ ] Code coverage meets threshold (>75%)
- [ ] No new security vulnerabilities
- [ ] Type hints are complete for public API

### Documentation

- [ ] API documentation is generated
- [ ] Tutorials run without errors
- [ ] Release notes drafted
- [ ] Migration guide (if breaking changes)

### Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Tests pass on Linux, macOS, Windows
- [ ] GPU tests pass (if applicable)

### Release

- [ ] Create release branch
- [ ] Tag release
- [ ] Build and test package locally
- [ ] Publish to TestPyPI
- [ ] Test installation from TestPyPI
- [ ] Publish to PyPI
- [ ] Create GitHub release
- [ ] Update Docker images

### Post-release

- [ ] Announce on Discourse
- [ ] Update conda-forge recipe
- [ ] Merge release branch to main
- [ ] Increment version for next development cycle
