#!/bin/bash
# TS-CVA Lab Server SSH Connection Script

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
else
    echo "Error: .env file not found. Please copy .env.example to .env and configure it."
    exit 1
fi

SERVER="${SSH_USER}@${SSH_SERVER}"
PASSWORD="${SSH_PASSWORD}"
PROJECT_DIR="${REMOTE_PROJECT_DIR}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== TS-CVA Lab Server Connection ===${NC}"
echo "Server: $SERVER"
echo "Project Directory: $PROJECT_DIR"
echo ""

# Check if expect is installed
if ! command -v expect &> /dev/null; then
    echo "Error: 'expect' is not installed. Please install it first."
    exit 1
fi

# Create expect script for SSH connection
cat > /tmp/ssh_connect.exp << EOFEXP
#!/usr/bin/expect -f

set timeout 30
set password "${PASSWORD}"

spawn ssh ${SERVER} -t "cd ${PROJECT_DIR} && exec bash -l"

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

interact
EOFEXP

chmod +x /tmp/ssh_connect.exp

# Connect to server
echo -e "${YELLOW}Connecting to server...${NC}"
/tmp/ssh_connect.exp

# Cleanup
rm -f /tmp/ssh_connect.exp
