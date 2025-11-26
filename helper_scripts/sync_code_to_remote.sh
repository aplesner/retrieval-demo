#!/bin/bash

# Check if project_variables.sh is sourced
if [ "${PROJECT_NAME}" != "retrieval-demo" ]; then
    echo "project_variables.sh is not sourced"
    exit 1
fi

# Define the remote directory
REMOTE_DIR="${USERNAME}@${REMOTE_SERVER}:${REMOTE_STORAGE_DIR}"

# Use rsync to sync to remote
rsync -avzh --info=progress2 \
    --exclude-from=.ignore_for_code_sync \
    --delete \
    ./ "$REMOTE_DIR"

echo "Code synced to remote!"
