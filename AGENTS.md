# Parlay Guard scripts

When adding a script that needs settings or credentials, use `src.env_config.env_value` to read the named settings. The helper uses the repository-root `.env` by default and gives explicit process environment variables precedence. Keep credentials out of source code, logs, and Git.
