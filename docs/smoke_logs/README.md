# WSL2 Ubuntu Smoke Logs

These logs were generated on the verified WSL2 Ubuntu / Python 3.12 / CUDA 13.0 path.

- `wsl2_setup_linux.log`: `bash scripts/setup_linux.sh`
- `wsl2_check_artifacts.log`: `python scripts/check_artifacts.py`
- `wsl2_smoke_test.log`: `bash scripts/smoke_test.sh`

The logs are path-sanitized with `<REPO_ROOT>` and `<USER_HOME>` placeholders. They are copied into `05_reproduction/logs/` when `scripts/build_submission_package.py --zip` builds the submission package.
