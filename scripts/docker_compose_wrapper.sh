#!/bin/bash
# Docker Compose Compatibility Wrapper
# Handles different Docker Compose installations automatically

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to detect Docker Compose
detect_compose() {
    # Try Docker Compose v2 (plugin)
    if docker compose version >/dev/null 2>&1; then
        echo "docker compose"
        return 0
    fi
    
    # Try standalone docker-compose v1
    if command -v docker-compose >/dev/null 2>&1; then
        echo "docker-compose"
        return 0
    fi
    
    # Try Docker Desktop's bundled docker-compose
    if [ -f "/Applications/Docker.app/Contents/Resources/bin/docker-compose" ]; then
        echo "/Applications/Docker.app/Contents/Resources/bin/docker-compose"
        return 0
    fi
    
    return 1
}

# Function to check if Docker daemon is running
check_docker_daemon() {
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        log_info "Starting Docker Desktop..."
        
        # Try to start Docker Desktop
        if [ -d "/Applications/Docker.app" ]; then
            open -a Docker
            log_info "Waiting for Docker Desktop to start..."
            
            # Wait up to 60 seconds for Docker to start
            for i in {1..12}; do
                if docker info >/dev/null 2>&1; then
                    log_success "Docker daemon is now running"
                    return 0
                fi
                sleep 5
                echo -n "."
            done
            echo
            log_error "Docker Desktop failed to start within 60 seconds"
            return 1
        else
            log_error "Docker Desktop not found. Please install Docker Desktop."
            return 1
        fi
    fi
    return 0
}

# Function to install Docker Compose if missing
install_compose() {
    log_warning "Docker Compose not found. Attempting to install..."
    
    # Check if we can install via curl
    if command -v curl >/dev/null 2>&1; then
        log_info "Installing Docker Compose standalone..."
        
        # Get latest version
        COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
        
        if [ -z "$COMPOSE_VERSION" ]; then
            log_error "Could not determine latest Docker Compose version"
            return 1
        fi
        
        # Download for macOS ARM64
        COMPOSE_URL="https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-darwin-aarch64"
        COMPOSE_PATH="/usr/local/bin/docker-compose"
        
        log_info "Downloading Docker Compose ${COMPOSE_VERSION}..."
        
        if curl -L "$COMPOSE_URL" -o "$COMPOSE_PATH" && chmod +x "$COMPOSE_PATH"; then
            log_success "Docker Compose installed successfully"
            echo "docker-compose"
            return 0
        else
            log_error "Failed to install Docker Compose"
            return 1
        fi
    else
        log_error "curl not available. Please install Docker Compose manually:"
        echo "  1. Download from: https://docs.docker.com/compose/install/"
        echo "  2. Or install Docker Desktop which includes Compose"
        return 1
    fi
}

# Main function to get compose command
get_compose_command() {
    local compose_cmd
    
    # Check Docker daemon first
    if ! check_docker_daemon; then
        return 1
    fi
    
    # Detect existing compose installation
    if compose_cmd=$(detect_compose); then
        echo "$compose_cmd"
        return 0
    fi
    
    # Try to install if not found
    log_warning "Docker Compose not found"
    if compose_cmd=$(install_compose); then
        echo "$compose_cmd"
        return 0
    fi
    
    return 1
}

# Function to run compose command
run_compose() {
    local compose_cmd
    
    if compose_cmd=$(get_compose_command); then
        log_info "Using: $compose_cmd"
        
        # Change to project root
        cd "$PROJECT_ROOT"
        
        # Execute the command
        if [[ "$compose_cmd" == "docker compose" ]]; then
            docker compose "$@"
        else
            $compose_cmd "$@"
        fi
    else
        log_error "Could not find or install Docker Compose"
        log_info "Please install Docker Desktop or Docker Compose manually"
        return 1
    fi
}

# Function to validate compose files
validate_compose_files() {
    local compose_files=(
        "docker-compose.yml"
        "docker-compose.dev.yml" 
        "docker-compose.demo.yml"
        "docker-compose.obsv.yml"
    )
    
    log_info "Validating Docker Compose files..."
    
    for file in "${compose_files[@]}"; do
        if [ -f "$PROJECT_ROOT/$file" ]; then
            log_info "Checking $file..."
            
            # Basic YAML validation
            if command -v python3 >/dev/null 2>&1; then
                if ! python3 -c "import yaml; yaml.safe_load(open('$PROJECT_ROOT/$file'))" 2>/dev/null; then
                    log_error "Invalid YAML in $file"
                    return 1
                fi
            fi
            
            log_success "$file is valid"
        else
            log_warning "$file not found"
        fi
    done
    
    return 0
}

# Function to show available services
show_services() {
    local compose_file="${1:-docker-compose.dev.yml}"
    
    if [ ! -f "$PROJECT_ROOT/$compose_file" ]; then
        log_error "Compose file not found: $compose_file"
        return 1
    fi
    
    log_info "Services defined in $compose_file:"
    
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "
import yaml
with open('$PROJECT_ROOT/$compose_file') as f:
    data = yaml.safe_load(f)
    if 'services' in data:
        for service in data['services']:
            print(f'  • {service}')
    else:
        print('  No services found')
" 2>/dev/null || log_error "Could not parse compose file"
    else
        # Fallback: simple grep
        grep -A 1000 "^services:" "$PROJECT_ROOT/$compose_file" | grep "^  [a-zA-Z]" | sed 's/^  /  • /' | sed 's/:.*$//'
    fi
}

# Function to setup environment
setup_environment() {
    log_info "Setting up GenomeVault Docker environment..."
    
    # Create data directories
    mkdir -p "$PROJECT_ROOT/data"/{cache,encrypted,input,output,processed,raw}
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/keys"
    mkdir -p "$PROJECT_ROOT/config"
    
    # Create .env file if it doesn't exist
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        log_info "Creating .env file..."
        cat > "$PROJECT_ROOT/.env" << 'EOF'
# GenomeVault Environment Variables
DATABASE_URL=postgresql://genomevault:secure_password@postgres:5432/genomevault
JWT_SECRET_KEY=change-this-secret-key-in-production
API_KEY_SECRET=change-this-api-key-secret
ENABLE_ZK_PROOFS=true
ENABLE_PIR=true
ENABLE_METAL_ACCELERATION=false
LOG_LEVEL=INFO
DEBUG=false
HIPAA_COMPLIANCE=true
AUDIT_LOG_RETENTION_DAYS=2557
EOF
        log_success "Created .env file"
    else
        log_success ".env file already exists"
    fi
    
    log_success "Environment setup complete"
}

# Main script logic
case "${1:-help}" in
    "detect")
        if compose_cmd=$(get_compose_command); then
            echo "Docker Compose found: $compose_cmd"
            $compose_cmd --version 2>/dev/null || echo "Version check failed"
        else
            echo "Docker Compose not found"
            exit 1
        fi
        ;;
    
    "validate")
        validate_compose_files
        ;;
    
    "services")
        show_services "${2:-docker-compose.dev.yml}"
        ;;
    
    "setup")
        setup_environment
        ;;
    
    "dev")
        log_info "Starting development environment..."
        run_compose -f docker-compose.yml -f docker-compose.dev.yml up -d
        ;;
    
    "demo") 
        log_info "Starting demo environment..."
        run_compose -f docker-compose.yml -f docker-compose.demo.yml up -d
        ;;
    
    "monitor")
        log_info "Starting observability stack..."
        run_compose -f docker-compose.yml -f docker-compose.obsv.yml up -d
        ;;
    
    "stop")
        log_info "Stopping all services..."
        run_compose -f docker-compose.yml down
        ;;
    
    "status")
        run_compose -f docker-compose.yml ps
        ;;
    
    "logs")
        service="${2:-api}"
        run_compose -f docker-compose.yml logs -f "$service"
        ;;
    
    "help"|"--help"|"-h")
        echo "GenomeVault Docker Compose Wrapper"
        echo "Usage: $0 [command] [options]"
        echo
        echo "Commands:"
        echo "  detect     - Detect available Docker Compose installation"
        echo "  validate   - Validate Docker Compose files"
        echo "  services   - Show available services in compose file"
        echo "  setup      - Set up environment and directories"
        echo "  dev        - Start development environment"
        echo "  demo       - Start demo environment" 
        echo "  monitor    - Start observability stack"
        echo "  stop       - Stop all services"
        echo "  status     - Show service status"
        echo "  logs       - Follow logs for service (default: api)"
        echo "  help       - Show this help"
        echo
        echo "Examples:"
        echo "  $0 setup          # Set up environment"
        echo "  $0 dev            # Start development services"
        echo "  $0 status         # Check service status"
        echo "  $0 logs api       # Follow API logs"
        echo "  $0 stop           # Stop all services"
        ;;
    
    *)
        # Pass through any other commands to compose
        run_compose "$@"
        ;;
esac