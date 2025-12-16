#!/bin/bash
cd /home/kavia/workspace/code-generation/personal-notes-manager-223136-223145/notes_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

