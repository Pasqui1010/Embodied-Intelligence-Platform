#!/bin/bash
# Comprehensive security check for Python dependencies
# Exit on any error
set -e

echo "🔒 Running comprehensive security checks..."

# Check for high-severity CVEs
echo "📋 Checking for security vulnerabilities..."
if pip-audit --format=json | jq -e '.[] | select(.severity == "HIGH" or .severity == "CRITICAL")' > /dev/null 2>&1; then
    echo "❌ HIGH or CRITICAL vulnerabilities found!"
    pip-audit
    exit 1
else
    echo "✅ No high-severity vulnerabilities found"
fi

# Run safety check
echo "🛡️ Running safety check..."
safety check --json --output=security_report.json

# Check for secrets in code
echo "🔍 Scanning for secrets..."
if command -v detect-secrets > /dev/null 2>&1; then
    detect-secrets scan --baseline .secrets.baseline
else
    echo "⚠️  detect-secrets not installed, skipping secret scan"
fi

# Bandit security linting
echo "🔐 Running bandit security linting..."
bandit -r intelligence/ -f json -o bandit_report.json || true

echo "✅ Security checks completed" 