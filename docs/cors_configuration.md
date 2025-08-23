# CORS Configuration

GenomeVault's API restricts cross-origin requests via the CORS middleware.
Allowed origins are configured through the `GENOMEVAULT_CORS_ORIGINS` environment
variable.

## Environment variable

Set a comma-separated list of origins to allow:

```
export GENOMEVAULT_CORS_ORIGINS="https://example.com,https://app.example.com"
```

If the variable is unset or empty, cross-origin requests are blocked.
Avoid using wildcard origins in production.
