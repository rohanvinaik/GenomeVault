#!/bin/bash
# GenomeVault Clinic Installer - One-button deployment for clinical environments
# Version: 1.0.0
# License: MIT

set -e

echo "🧬 GenomeVault Clinic Installer v1.0"
echo "===================================="
echo ""

# Configuration
INSTALL_DIR="${GENOMEVAULT_HOME:-/opt/genomevault}"
DATA_DIR="${GENOMEVAULT_DATA:-/var/lib/genomevault}"
LOG_DIR="${GENOMEVAULT_LOGS:-/var/log/genomevault}"
CONFIG_DIR="${GENOMEVAULT_CONFIG:-/etc/genomevault}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
LOG_FILE="/tmp/genomevault_install_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

# Helper functions
print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root"
        print_status "Please run as a regular user with sudo privileges"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_deps=()
    
    # Check for Docker
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    else
        docker_version=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        print_success "Docker $docker_version found"
    fi
    
    # Check for Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        if ! docker compose version &> /dev/null; then
            missing_deps+=("docker-compose")
        else
            print_success "Docker Compose plugin found"
        fi
    else
        compose_version=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
        print_success "Docker Compose $compose_version found"
    fi
    
    # Check for Python 3.10+
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version | cut -d' ' -f2)
        major=$(echo $python_version | cut -d'.' -f1)
        minor=$(echo $python_version | cut -d'.' -f2)
        
        if [[ $major -eq 3 && $minor -ge 10 ]]; then
            print_success "Python $python_version found"
        else
            missing_deps+=("python3.10+")
        fi
    else
        missing_deps+=("python3")
    fi
    
    # Check for Git
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    else
        git_version=$(git --version | cut -d' ' -f3)
        print_success "Git $git_version found"
    fi
    
    # Check for curl
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi
    
    # Report missing dependencies
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        echo ""
        echo "Installation instructions:"
        echo "  Docker:         https://docs.docker.com/get-docker/"
        echo "  Python 3.10+:   https://www.python.org/downloads/"
        echo "  Git:            https://git-scm.com/downloads"
        exit 1
    fi
    
    print_success "All prerequisites satisfied"
}

# Detect operating system
detect_os() {
    print_status "Detecting operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$NAME
            VER=$VERSION_ID
            print_success "Detected: $OS $VER"
        else
            OS="Linux"
            print_success "Detected: Generic Linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        OS="macOS"
        VER=$(sw_vers -productVersion)
        print_success "Detected: macOS $VER"
        
        # Check for Homebrew
        if ! command -v brew &> /dev/null; then
            print_warning "Homebrew not found. Installing..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Install system dependencies
install_dependencies() {
    print_status "Installing system dependencies..."
    
    if [[ "$OS" == "macOS" ]]; then
        # macOS dependencies
        brew install cmake node npm
        
        # Install Rust for accelerator
        if ! command -v rustc &> /dev/null; then
            print_status "Installing Rust..."
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source "$HOME/.cargo/env"
        fi
        
    elif [[ "$OS" == "Ubuntu" ]] || [[ "$OS" == "Debian"* ]]; then
        # Ubuntu/Debian dependencies
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            nodejs \
            npm \
            libssl-dev \
            pkg-config \
            postgresql-client
        
        # Install Rust
        if ! command -v rustc &> /dev/null; then
            print_status "Installing Rust..."
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source "$HOME/.cargo/env"
        fi
        
    elif [[ "$OS" == "CentOS"* ]] || [[ "$OS" == "Red Hat"* ]] || [[ "$OS" == "Fedora"* ]]; then
        # RHEL-based dependencies
        sudo yum groupinstall -y "Development Tools"
        sudo yum install -y \
            cmake \
            nodejs \
            npm \
            openssl-devel \
            postgresql
    fi
    
    # Install Circom and SnarkJS
    print_status "Installing Zero-Knowledge toolchain..."
    sudo npm install -g circom snarkjs
    
    print_success "Dependencies installed"
}

# Setup directories
setup_directories() {
    print_status "Setting up directories..."
    
    # Create directories with proper permissions
    sudo mkdir -p "$INSTALL_DIR" "$DATA_DIR" "$LOG_DIR" "$CONFIG_DIR"
    sudo mkdir -p "$DATA_DIR/uploads" "$DATA_DIR/cache" "$DATA_DIR/results"
    sudo mkdir -p "$LOG_DIR/api" "$LOG_DIR/worker" "$LOG_DIR/audit"
    
    # Set ownership
    sudo chown -R $USER:$USER "$INSTALL_DIR" "$DATA_DIR" "$LOG_DIR" "$CONFIG_DIR"
    
    # Set permissions (restricted for HIPAA compliance)
    chmod 750 "$DATA_DIR"
    chmod 750 "$LOG_DIR"
    chmod 755 "$CONFIG_DIR"
    
    print_success "Directories created with proper permissions"
}

# Clone and setup repository
setup_repository() {
    print_status "Setting up GenomeVault repository..."
    
    cd "$INSTALL_DIR"
    
    # Clone if not exists
    if [ ! -d "genomevault" ]; then
        print_status "Cloning repository..."
        git clone https://github.com/genomevault/genomevault.git
    else
        print_status "Repository exists, pulling latest changes..."
        cd genomevault
        git pull origin main
    fi
    
    cd "$INSTALL_DIR/genomevault"
    
    # Create virtual environment
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install GenomeVault with all extras
    print_status "Installing GenomeVault packages..."
    pip install -e ".[all]"
    
    # Install additional clinical requirements
    pip install \
        cryptography \
        pycryptodome \
        python-jose \
        passlib \
        python-multipart \
        aiofiles \
        prometheus-client \
        structlog
    
    print_success "Repository setup complete"
}

# Build Rust accelerator
build_rust_accelerator() {
    print_status "Building Rust accelerator for performance..."
    
    cd "$INSTALL_DIR/genomevault"
    
    if [ -f "build_rust.sh" ]; then
        ./build_rust.sh
        print_success "Rust accelerator built successfully"
    else
        print_warning "Rust build script not found, skipping accelerator"
    fi
}

# Setup ZK toolchain
setup_zk_toolchain() {
    print_status "Setting up Zero-Knowledge proof system..."
    
    cd "$INSTALL_DIR/genomevault"
    
    # Create directories
    mkdir -p zk_circuits/trusted_setup
    mkdir -p zk_circuits/compiled
    
    cd zk_circuits/trusted_setup
    
    # Download trusted setup (Powers of Tau)
    if [ ! -f "powersOfTau28_hez_final_15.ptau" ]; then
        print_status "Downloading trusted setup parameters..."
        curl -L -o powersOfTau28_hez_final_15.ptau \
            https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_15.ptau
    fi
    
    cd "$INSTALL_DIR/genomevault"
    
    # Compile circuits if script exists
    if [ -f "scripts/compile_circuits.sh" ]; then
        print_status "Compiling ZK circuits..."
        ./scripts/compile_circuits.sh
    fi
    
    print_success "ZK toolchain ready"
}

# Generate secure configuration
generate_config() {
    print_status "Generating secure configuration..."
    
    # Generate secure passwords
    DB_PASSWORD=$(openssl rand -base64 32)
    JWT_SECRET=$(openssl rand -base64 64)
    ENCRYPTION_KEY=$(openssl rand -base64 32)
    REDIS_PASSWORD=$(openssl rand -base64 32)
    
    # Create .env file
    cat > "$CONFIG_DIR/genomevault.env" << EOF
# GenomeVault Configuration
# Generated: $(date)

# Environment
GENOMEVAULT_ENV=production
GENOMEVAULT_HOME=$INSTALL_DIR
GENOMEVAULT_DATA=$DATA_DIR
GENOMEVAULT_LOGS=$LOG_DIR

# Database
DATABASE_URL=postgresql://genomevault:${DB_PASSWORD}@localhost:5432/genomevault
REDIS_URL=redis://default:${REDIS_PASSWORD}@localhost:6379

# Security
JWT_SECRET_KEY=${JWT_SECRET}
ENCRYPTION_KEY=${ENCRYPTION_KEY}
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4

# HIPAA Compliance
ENABLE_AUDIT_LOG=true
ENABLE_ENCRYPTION=true
PHI_DETECTION=true
SESSION_TIMEOUT=900

# Performance
ENABLE_RUST_ACCELERATOR=true
CACHE_SIZE_MB=1024
MAX_UPLOAD_SIZE_MB=1000

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO
EOF
    
    # Secure the config file
    chmod 600 "$CONFIG_DIR/genomevault.env"
    
    print_success "Configuration generated"
}

# Setup Docker services
setup_docker_services() {
    print_status "Setting up Docker services..."
    
    cd "$INSTALL_DIR/genomevault"
    
    # Create docker-compose.yml for clinic deployment
    cat > docker-compose.clinic.yml << 'EOF'
version: '3.8'

services:
  genomevault-api:
    build: .
    container_name: genomevault-api
    ports:
      - "8000:8000"
    env_file:
      - /etc/genomevault/genomevault.env
    volumes:
      - ${DATA_DIR}:/data
      - ${LOG_DIR}:/logs
      - /etc/genomevault:/config:ro
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    container_name: genomevault-db
    environment:
      - POSTGRES_DB=genomevault
      - POSTGRES_USER=genomevault
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./deploy/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U genomevault"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: genomevault-cache
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  grafana:
    image: grafana/grafana:latest
    container_name: genomevault-grafana
    ports:
      - "3000:3000"
    volumes:
      - ./deploy/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./deploy/grafana/datasources:/etc/grafana/provisioning/datasources
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=redis-datasource
      - GF_AUTH_ANONYMOUS_ENABLED=false
      - GF_AUTH_BASIC_ENABLED=true
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: genomevault-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./deploy/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: genomevault-proxy
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./deploy/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./deploy/nginx/ssl:/etc/nginx/ssl:ro
      - nginx_logs:/var/log/nginx
    depends_on:
      - genomevault-api
      - grafana
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  grafana_data:
  prometheus_data:
  nginx_logs:
EOF
    
    # Source environment variables
    source "$CONFIG_DIR/genomevault.env"
    export DB_PASSWORD REDIS_PASSWORD
    
    # Start services
    print_status "Starting Docker services..."
    docker-compose -f docker-compose.clinic.yml up -d
    
    print_success "Docker services running"
}

# Setup systemd service for non-Docker deployment
setup_systemd() {
    print_status "Setting up systemd service..."
    
    sudo cat > /etc/systemd/system/genomevault.service << EOF
[Unit]
Description=GenomeVault API Service
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$INSTALL_DIR/genomevault
Environment="PATH=$INSTALL_DIR/genomevault/venv/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=$CONFIG_DIR/genomevault.env
ExecStartPre=$INSTALL_DIR/genomevault/venv/bin/python -m genomevault.cli db upgrade
ExecStart=$INSTALL_DIR/genomevault/venv/bin/uvicorn genomevault.api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10
StandardOutput=append:$LOG_DIR/api/genomevault.log
StandardError=append:$LOG_DIR/api/genomevault_error.log

# Security
PrivateTmp=true
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$DATA_DIR $LOG_DIR

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable genomevault
    
    print_success "Systemd service configured"
}

# Initialize database
initialize_database() {
    print_status "Initializing database..."
    
    cd "$INSTALL_DIR/genomevault"
    source venv/bin/activate
    
    # Wait for PostgreSQL to be ready
    print_status "Waiting for database to be ready..."
    for i in {1..30}; do
        if docker exec genomevault-db pg_isready -U genomevault > /dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    
    # Run migrations
    print_status "Running database migrations..."
    source "$CONFIG_DIR/genomevault.env"
    export DATABASE_URL
    
    python -m alembic upgrade head
    
    # Load initial data if script exists
    if [ -f "scripts/seed_data.py" ]; then
        print_status "Loading initial data..."
        python scripts/seed_data.py
    fi
    
    print_success "Database initialized"
}

# Setup SSL certificates
setup_ssl() {
    print_status "Setting up SSL certificates..."
    
    SSL_DIR="$INSTALL_DIR/genomevault/deploy/nginx/ssl"
    mkdir -p "$SSL_DIR"
    
    # Generate self-signed certificate for testing
    # In production, use Let's Encrypt or proper certificates
    if [ ! -f "$SSL_DIR/cert.pem" ]; then
        print_warning "Generating self-signed certificate (replace with proper cert in production)"
        openssl req -x509 -newkey rsa:4096 -nodes \
            -keyout "$SSL_DIR/key.pem" \
            -out "$SSL_DIR/cert.pem" \
            -days 365 \
            -subj "/C=US/ST=State/L=City/O=Clinic/CN=localhost"
    fi
    
    chmod 600 "$SSL_DIR/key.pem"
    chmod 644 "$SSL_DIR/cert.pem"
    
    print_success "SSL certificates configured"
}

# Create CLI shortcuts
create_cli_shortcuts() {
    print_status "Creating CLI shortcuts..."
    
    # Create wrapper script
    cat > "$INSTALL_DIR/genomevault/bin/genomevault" << EOF
#!/bin/bash
source $INSTALL_DIR/genomevault/venv/bin/activate
source $CONFIG_DIR/genomevault.env
python -m genomevault.cli "\$@"
EOF
    
    chmod +x "$INSTALL_DIR/genomevault/bin/genomevault"
    
    # Create symlink
    sudo ln -sf "$INSTALL_DIR/genomevault/bin/genomevault" /usr/local/bin/genomevault
    
    print_success "CLI shortcuts created"
}

# Run health checks
run_health_checks() {
    print_status "Running health checks..."
    
    # Wait for services to start
    sleep 10
    
    # Check API
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API is healthy"
    else
        print_error "API health check failed"
        docker logs genomevault-api
        exit 1
    fi
    
    # Check database
    if docker exec genomevault-db pg_isready -U genomevault > /dev/null 2>&1; then
        print_success "Database is healthy"
    else
        print_error "Database health check failed"
    fi
    
    # Check Redis
    if docker exec genomevault-cache redis-cli ping > /dev/null 2>&1; then
        print_success "Redis is healthy"
    else
        print_error "Redis health check failed"
    fi
    
    # Check Grafana
    if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        print_success "Grafana is healthy"
    else
        print_warning "Grafana not responding (may still be starting)"
    fi
    
    # Check Prometheus
    if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
        print_success "Prometheus is healthy"
    else
        print_warning "Prometheus not responding"
    fi
}

# Setup backups
setup_backups() {
    print_status "Setting up automated backups..."
    
    # Create backup directory
    BACKUP_DIR="$DATA_DIR/backups"
    mkdir -p "$BACKUP_DIR"
    
    # Create backup script
    cat > "$INSTALL_DIR/genomevault/bin/backup.sh" << 'EOF'
#!/bin/bash
BACKUP_DIR="/var/lib/genomevault/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup database
docker exec genomevault-db pg_dump -U genomevault genomevault | \
    gzip > "$BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz"

# Backup data directory
tar -czf "$BACKUP_DIR/data_backup_$TIMESTAMP.tar.gz" \
    -C /var/lib/genomevault uploads cache

# Keep only last 7 days of backups
find "$BACKUP_DIR" -name "*.gz" -mtime +7 -delete

echo "Backup completed: $TIMESTAMP"
EOF
    
    chmod +x "$INSTALL_DIR/genomevault/bin/backup.sh"
    
    # Add to crontab (daily at 2 AM)
    (crontab -l 2>/dev/null; echo "0 2 * * * $INSTALL_DIR/genomevault/bin/backup.sh") | crontab -
    
    print_success "Automated backups configured"
}

# Print installation summary
print_summary() {
    echo ""
    echo "========================================="
    echo -e "${GREEN}✓ GenomeVault Installation Complete!${NC}"
    echo "========================================="
    echo ""
    echo "Installation Details:"
    echo "  Install Directory: $INSTALL_DIR"
    echo "  Data Directory:    $DATA_DIR"
    echo "  Log Directory:     $LOG_DIR"
    echo "  Config Directory:  $CONFIG_DIR"
    echo ""
    echo "Access URLs:"
    echo "  API:        http://localhost:8000"
    echo "  API Docs:   http://localhost:8000/docs"
    echo "  Grafana:    http://localhost:3000 (admin/admin)"
    echo "  Prometheus: http://localhost:9090"
    echo ""
    echo "Quick Start Commands:"
    echo "  genomevault --help                    # Show CLI help"
    echo "  genomevault demo run --type full      # Run full demo"
    echo "  genomevault hdc encode <file>         # Encode VCF file"
    echo "  genomevault status                    # Check system status"
    echo ""
    echo "Docker Commands:"
    echo "  docker-compose -f docker-compose.clinic.yml ps     # View services"
    echo "  docker-compose -f docker-compose.clinic.yml logs   # View logs"
    echo "  docker-compose -f docker-compose.clinic.yml down   # Stop services"
    echo ""
    echo "Important Files:"
    echo "  Configuration: $CONFIG_DIR/genomevault.env"
    echo "  API Logs:      $LOG_DIR/api/genomevault.log"
    echo "  Install Log:   $LOG_FILE"
    echo ""
    echo "Security Notes:"
    echo "  - Change default passwords in production"
    echo "  - Install proper SSL certificates"
    echo "  - Configure firewall rules"
    echo "  - Enable audit logging for HIPAA compliance"
    echo ""
    echo "Documentation: https://genomevault.io/docs/clinic-guide"
    echo "Support:       support@genomevault.io"
    echo ""
    print_warning "Remember to:"
    echo "  1. Change the Grafana admin password"
    echo "  2. Configure SSL certificates for production"
    echo "  3. Set up regular backups"
    echo "  4. Review security settings"
}

# Cleanup on error
cleanup_on_error() {
    print_error "Installation failed. Check log: $LOG_FILE"
    
    # Optionally stop services
    if [ -f "$INSTALL_DIR/genomevault/docker-compose.clinic.yml" ]; then
        cd "$INSTALL_DIR/genomevault"
        docker-compose -f docker-compose.clinic.yml down 2>/dev/null || true
    fi
    
    exit 1
}

# Trap errors
trap cleanup_on_error ERR

# Main installation flow
main() {
    echo "Starting GenomeVault clinic installation..."
    echo "Installation log: $LOG_FILE"
    echo ""
    
    # Confirm installation
    read -p "This will install GenomeVault on your system. Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
    
    # Run installation steps
    check_root
    check_prerequisites
    detect_os
    install_dependencies
    setup_directories
    setup_repository
    build_rust_accelerator
    setup_zk_toolchain
    generate_config
    setup_ssl
    setup_docker_services
    initialize_database
    setup_systemd
    create_cli_shortcuts
    setup_backups
    run_health_checks
    print_summary
    
    print_success "Installation completed successfully!"
    print_status "Log saved to: $LOG_FILE"
}

# Parse arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h         Show this help message"
        echo "  --uninstall        Remove GenomeVault installation"
        echo "  --update           Update existing installation"
        echo "  --check            Run health checks only"
        echo ""
        echo "Environment Variables:"
        echo "  GENOMEVAULT_HOME   Installation directory (default: /opt/genomevault)"
        echo "  GENOMEVAULT_DATA   Data directory (default: /var/lib/genomevault)"
        echo "  GENOMEVAULT_LOGS   Log directory (default: /var/log/genomevault)"
        exit 0
        ;;
    --uninstall)
        print_warning "Uninstalling GenomeVault..."
        cd "$INSTALL_DIR/genomevault" 2>/dev/null && \
            docker-compose -f docker-compose.clinic.yml down -v
        sudo systemctl stop genomevault 2>/dev/null || true
        sudo systemctl disable genomevault 2>/dev/null || true
        sudo rm -f /etc/systemd/system/genomevault.service
        sudo rm -f /usr/local/bin/genomevault
        print_success "GenomeVault uninstalled"
        exit 0
        ;;
    --update)
        print_status "Updating GenomeVault..."
        cd "$INSTALL_DIR/genomevault"
        git pull origin main
        source venv/bin/activate
        pip install --upgrade -e ".[all]"
        docker-compose -f docker-compose.clinic.yml pull
        docker-compose -f docker-compose.clinic.yml up -d
        print_success "Update complete"
        exit 0
        ;;
    --check)
        run_health_checks
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac