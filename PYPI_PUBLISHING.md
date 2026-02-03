# PyPI Publishing Guide for hyperct

This guide walks through the complete process of publishing the `hyperct` package to PyPI.

## Prerequisites

### 1. Install Required Tools

```bash
pip install --upgrade pip build twine
```

### 2. Create PyPI Accounts

- **Production PyPI**: Register at https://pypi.org/account/register/
- **Test PyPI** (optional but recommended): Register at https://test.pypi.org/account/register/

### 3. Set Up API Tokens

API tokens are more secure than using your password.

#### Create API Tokens

1. Go to your PyPI account settings
2. Scroll to "API tokens" section
3. Click "Add API token"
4. Give it a name (e.g., "hyperct-publishing")
5. Set scope to "Entire account" or specific to "hyperct" project (after first upload)
6. Copy the token (starts with `pypi-...`)

Repeat for Test PyPI if testing first.

#### Configure Credentials

**Option A: Using ~/.pypirc file (Recommended)**

Create or edit `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...  # Your PyPI token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9yZw...  # Your TestPyPI token
```

**Important**: Set proper permissions to protect your tokens:
```bash
chmod 600 ~/.pypirc
```

**Option B: Using Environment Variables**

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEIcHlwaS5vcmc...
```

Add these to your `~/.bashrc` or `~/.zshrc` for persistence (be careful with security).

## Publishing Workflow

### Step 1: Update Version

Edit `setup.py` and increment the version:

```python
version='0.3.2',  # Update from 0.3.1
```

Follow semantic versioning:
- **Major** (1.0.0): Breaking changes
- **Minor** (0.4.0): New features, backward compatible
- **Patch** (0.3.2): Bug fixes, backward compatible

### Step 2: Update CHANGELOG (Recommended)

Create a `CHANGELOG.md` if it doesn't exist and document changes:

```markdown
## [0.3.2] - 2026-02-03
### Added
- New feature description

### Changed
- What changed

### Fixed
- Bug fixes
```

### Step 3: Commit Changes

```bash
git add setup.py CHANGELOG.md
git commit -m "Bump version to 0.3.2"
git push
```

### Step 4: Test Upload to TestPyPI (Optional but Recommended)

Test the publishing process without affecting production:

```bash
# Using the bash script:
./publish_to_pypi.sh --test-only

# Or using the Python script:
python publish_to_pypi.py --test-only
```

Then test installation from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            hyperct==0.3.2
```

### Step 5: Publish to Production PyPI

Once you've verified everything works:

```bash
# Using the bash script:
./publish_to_pypi.sh

# Or using the Python script:
python publish_to_pypi.py
```

The script will:
1. Check git status is clean
2. Verify version tag doesn't exist
3. Run tests
4. Clean old builds
5. Build distribution packages (wheel and sdist)
6. Check package with twine
7. Upload to PyPI
8. Create git tag

### Step 6: Push Git Tag

```bash
git push origin v0.3.2
```

### Step 7: Verify

Check your package at: https://pypi.org/project/hyperct/

Test installation:
```bash
pip install hyperct==0.3.2
```

## Script Options

Both `publish_to_pypi.sh` and `publish_to_pypi.py` support:

- `--test-only`: Upload to TestPyPI instead of production
- `--skip-tests`: Skip running pytest (not recommended)
- `--skip-tag`: Don't create git tag

Examples:
```bash
# Test upload without running tests
./publish_to_pypi.sh --test-only --skip-tests

# Publish without creating tag (if you'll tag manually)
python publish_to_pypi.py --skip-tag
```

## Manual Publishing (Alternative)

If you prefer to run commands manually:

```bash
# 1. Clean old builds
rm -rf build/ dist/ *.egg-info

# 2. Build distribution
python -m build

# 3. Check distribution
python -m twine check dist/*

# 4. Upload to TestPyPI (optional)
python -m twine upload --repository testpypi dist/*

# 5. Upload to PyPI
python -m twine upload dist/*

# 6. Create and push tag
git tag -a v0.3.2 -m "Release version 0.3.2"
git push origin v0.3.2
```

## Troubleshooting

### "Repository does not allow updating"

If you try to upload the same version twice, PyPI will reject it. You must increment the version number.

### "Invalid or non-existent authentication information"

- Check your token is correct in `~/.pypirc`
- Ensure token starts with `pypi-` (for PyPI) or `pypi-` (for TestPyPI)
- Verify `username = __token__` (not your username)

### "403 Forbidden"

- You may not have permission for the package name
- For first upload, the package name must be available
- Check if the package already exists at https://pypi.org/project/hyperct/

### Tests Fail

```bash
# Run tests manually to see errors
pytest hyperct/tests/ -v

# Skip tests if needed (not recommended)
./publish_to_pypi.sh --skip-tests
```

### Missing Dependencies

```bash
pip install --upgrade pip build twine pytest pytest-cov
```

## Security Best Practices

1. **Never commit tokens**: Don't commit `.pypirc` or files with tokens
2. **Use project-scoped tokens**: After first upload, create project-specific tokens
3. **Rotate tokens**: Regularly update your API tokens
4. **Protect .pypirc**: `chmod 600 ~/.pypirc`
5. **Use 2FA**: Enable two-factor authentication on PyPI

## Additional Resources

- PyPI Help: https://pypi.org/help/
- Python Packaging Guide: https://packaging.python.org/
- Twine Documentation: https://twine.readthedocs.io/
- TestPyPI: https://test.pypi.org/