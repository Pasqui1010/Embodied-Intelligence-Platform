#!/bin/bash
# Security check for Python dependencies
pip install --upgrade pip pip-audit safety
pip-audit
safety check 