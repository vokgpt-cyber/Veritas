#!/bin/bash

# EPAM VERITAS - Linux/Mac Start Script
# Verbal Intelligence, Transcription and Summarization v2.0

set -e

echo ""
echo "===================================="
echo "EPAM VERITAS - Starting Application"
echo "===================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

# Check if dependencies are installed
if ! pip show fastapi > /dev/null 2>&1; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
fi

# Run the application
echo ""
echo "Starting EPAM VERITAS API Server..."
echo "Server will be available at: http://127.0.0.1:8000"
echo "API Documentation: http://127.0.0.1:8000/api/docs"
echo ""

uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Failed to start the application"
    exit 1
fi
