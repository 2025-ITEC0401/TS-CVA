#!/bin/bash
# Sync TS-CVA project to lab server
# Run this script from local machine

set -e

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
else
    echo "Error: .env file not found"
    exit 1
fi

# Configuration
LOCAL_DIR="${LOCAL_PROJECT_DIR}"
SERVER="${SSH_USER}@${SSH_SERVER}"
REMOTE_DIR="${REMOTE_PROJECT_DIR}"
PASSWORD="${SSH_PASSWORD}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Syncing TS-CVA to Lab Server${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if in correct directory
if [ ! -f "$LOCAL_DIR/train.py" ]; then
    echo -e "${RED}Error: Not in TimeCMA project directory${NC}"
    exit 1
fi

cd "$LOCAL_DIR"

# Create/update exclude list
cat > /tmp/rsync_exclude.txt << 'EOF'
__pycache__/
*.pyc
*.pyo
.git/
.DS_Store
*.log
logs/*/*.log
*.swp
*.swo
*~
.vscode/
.idea/
venv/
env/
dataset/
Embeddings/
Results/
EOF

echo -e "${BLUE}Files to sync:${NC}"
echo "  Source: $LOCAL_DIR"
echo "  Destination: $SERVER:$REMOTE_DIR"
echo ""

# Create expect script for rsync
cat > /tmp/sync_expect.exp << EOFEXP
#!/usr/bin/expect -f

set timeout 120
set password "$PASSWORD"

set source [lindex \$argv 0]
set dest [lindex \$argv 1]

spawn rsync -avz --exclude-from=/tmp/rsync_exclude.txt --delete \$source \$dest

expect {
    "password:" {
        send "\$password\r"
        exp_continue
    }
    "yes/no" {
        send "yes\r"
        exp_continue
    }
    eof
}
EOFEXP

chmod +x /tmp/sync_expect.exp

# Sync files
echo -e "${YELLOW}Syncing files...${NC}"
/tmp/sync_expect.exp "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Sync completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Connect to server: ./ssh_connect.sh"
    echo "  2. Setup environment: bash setup_env.sh"
    echo "  3. Activate environment: conda activate TS-CVA"
    echo ""
else
    echo -e "${RED}✗ Sync failed${NC}"
    exit 1
fi

# Cleanup
rm -f /tmp/sync_expect.exp
