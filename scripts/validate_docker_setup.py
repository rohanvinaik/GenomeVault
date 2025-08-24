#!/usr/bin/env python3
"""
GenomeVault Docker Setup Validator

Validates Docker configuration without requiring Docker daemon to be running.
Checks compose files, environment setup, and provides setup recommendations.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any


class DockerSetupValidator:
    """Validates Docker setup for GenomeVault."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.compose_files = [
            "docker-compose.yml",
            "docker-compose.dev.yml", 
            "docker-compose.demo.yml",
            "docker-compose.obsv.yml"
        ]
        self.issues = []
        self.recommendations = []
        
    def check_docker_installation(self) -> Tuple[bool, str]:
        """Check if Docker is installed."""
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
            return False, "Docker command failed"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, "Docker not found in PATH"
    
    def check_docker_compose(self) -> Tuple[bool, str]:
        """Check Docker Compose availability."""
        # Try docker compose (v2)
        try:
            result = subprocess.run(
                ["docker", "compose", "version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                return True, f"Docker Compose v2: {result.stdout.strip()}"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Try docker-compose (v1)
        try:
            result = subprocess.run(
                ["docker-compose", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                return True, f"Docker Compose v1: {result.stdout.strip()}"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return False, "Docker Compose not found"
    
    def check_compose_files(self) -> Dict[str, bool]:
        """Check if compose files exist and are valid."""
        results = {}
        
        for compose_file in self.compose_files:
            file_path = self.project_root / compose_file
            exists = file_path.exists()
            results[compose_file] = exists
            
            if not exists:
                self.issues.append(f"Missing compose file: {compose_file}")
            else:
                # Try to validate YAML structure (basic check)
                try:
                    import yaml
                    with open(file_path) as f:
                        yaml.safe_load(f)
                except ImportError:
                    self.recommendations.append("Install PyYAML for better compose file validation: pip install pyyaml")
                except yaml.YAMLError as e:
                    self.issues.append(f"Invalid YAML in {compose_file}: {e}")
        
        return results
    
    def check_dockerfiles(self) -> Dict[str, bool]:
        """Check if Dockerfiles exist."""
        dockerfiles = [
            "Dockerfile",
            "Dockerfile.pir", 
            "Dockerfile.prover",
            "docker/api/Dockerfile",
            "docker/pir/Dockerfile",
            "docker/blockchain/Dockerfile"
        ]
        
        results = {}
        for dockerfile in dockerfiles:
            file_path = self.project_root / dockerfile
            results[dockerfile] = file_path.exists()
            if not file_path.exists():
                self.recommendations.append(f"Consider creating {dockerfile} if needed")
        
        return results
    
    def check_environment_files(self) -> Dict[str, Any]:
        """Check environment configuration."""
        env_files = [".env", ".env.example", ".env.template"]
        results = {}
        
        for env_file in env_files:
            file_path = self.project_root / env_file
            results[env_file] = {
                "exists": file_path.exists(),
                "size": file_path.stat().st_size if file_path.exists() else 0
            }
        
        # Check if any .env file exists
        if not any(results[f]["exists"] for f in env_files):
            self.issues.append("No environment file found (.env, .env.example, or .env.template)")
            self.recommendations.append("Create .env file with required environment variables")
        
        return results
    
    def check_volumes_and_data(self) -> Dict[str, Any]:
        """Check data directories and volumes."""
        data_dirs = ["data", "logs", "keys", "config"]
        results = {}
        
        for data_dir in data_dirs:
            dir_path = self.project_root / data_dir
            results[data_dir] = {
                "exists": dir_path.exists(),
                "is_dir": dir_path.is_dir() if dir_path.exists() else False,
                "writable": os.access(dir_path, os.W_OK) if dir_path.exists() else False
            }
            
            if dir_path.exists() and not results[data_dir]["writable"]:
                self.issues.append(f"Data directory {data_dir} is not writable")
        
        return results
    
    def analyze_compose_services(self) -> Dict[str, Any]:
        """Analyze services in compose files."""
        services_analysis = {}
        
        for compose_file in self.compose_files:
            file_path = self.project_root / compose_file
            if not file_path.exists():
                continue
                
            try:
                import yaml
                with open(file_path) as f:
                    compose_data = yaml.safe_load(f)
                
                if "services" in compose_data:
                    services = list(compose_data["services"].keys())
                    services_analysis[compose_file] = {
                        "service_count": len(services),
                        "services": services
                    }
                    
                    # Check for common services
                    expected_services = ["api", "postgres", "redis"]
                    missing_services = [s for s in expected_services if s not in services]
                    if missing_services:
                        self.recommendations.append(
                            f"{compose_file}: Consider adding services: {', '.join(missing_services)}"
                        )
                        
            except ImportError:
                services_analysis[compose_file] = {"error": "PyYAML not available"}
            except Exception as e:
                services_analysis[compose_file] = {"error": str(e)}
        
        return services_analysis
    
    def check_network_ports(self) -> Dict[str, Any]:
        """Check for port conflicts (basic check without Docker running)."""
        common_ports = {
            8000: "API Server",
            8001: "ZK Prover", 
            8002: "PIR Server",
            5432: "PostgreSQL",
            6379: "Redis",
            9090: "Prometheus",
            3000: "Grafana"
        }
        
        port_status = {}
        for port, service in common_ports.items():
            try:
                # Try to check if port is in use
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                port_status[port] = {
                    "service": service,
                    "in_use": result == 0
                }
                
                if result == 0:
                    self.recommendations.append(f"Port {port} ({service}) appears to be in use")
                    
            except Exception:
                port_status[port] = {
                    "service": service, 
                    "in_use": False,
                    "error": "Could not check port"
                }
        
        return port_status
    
    def generate_setup_script(self) -> str:
        """Generate a setup script for Docker environment."""
        script_lines = [
            "#!/bin/bash",
            "# GenomeVault Docker Setup Script",
            "# Generated automatically by validate_docker_setup.py",
            "",
            "set -e",
            "",
            "echo '🐳 Setting up GenomeVault Docker environment...'",
            "",
            "# Check Docker installation",
            "if ! command -v docker &> /dev/null; then",
            "    echo '❌ Docker not found. Please install Docker Desktop.'",
            "    echo 'Visit: https://docs.docker.com/get-docker/'",
            "    exit 1",
            "fi",
            "",
            "# Check Docker Compose", 
            "if ! docker compose version &> /dev/null && ! docker-compose --version &> /dev/null; then",
            "    echo '❌ Docker Compose not found. Please install Docker Compose.'",
            "    exit 1",
            "fi",
            "",
            "# Create data directories",
            "mkdir -p data/{cache,encrypted,input,output,processed,raw}",
            "mkdir -p logs",
            "mkdir -p keys", 
            "mkdir -p config",
            "",
            "# Create .env file if it doesn't exist",
            "if [ ! -f .env ]; then",
            "    echo '📝 Creating .env file...'",
            "    cat > .env << 'EOF'",
            "# GenomeVault Environment Variables",
            "DATABASE_URL=postgresql://genomevault:secure_password@postgres:5432/genomevault",
            "JWT_SECRET_KEY=change-this-secret-key-in-production",
            "API_KEY_SECRET=change-this-api-key-secret", 
            "ENABLE_ZK_PROOFS=true",
            "ENABLE_PIR=true",
            "LOG_LEVEL=INFO",
            "DEBUG=false",
            "EOF",
            "    echo '✅ Created .env file'",
            "else",
            "    echo '✅ .env file already exists'",
            "fi",
            "",
            "# Pull required images", 
            "echo '📦 Pulling Docker images...'",
            "docker compose -f docker-compose.dev.yml pull",
            "",
            "# Build custom images",
            "echo '🔨 Building GenomeVault images...'", 
            "docker compose -f docker-compose.dev.yml build",
            "",
            "echo '✅ Setup complete!'",
            "echo ''",
            "echo 'To start services:'",
            "echo '  docker compose -f docker-compose.dev.yml up -d'",
            "echo ''", 
            "echo 'To check status:'",
            "echo '  docker compose -f docker-compose.dev.yml ps'",
            "echo ''",
            "echo 'To view logs:'",
            "echo '  docker compose -f docker-compose.dev.yml logs -f api'",
        ]
        
        return "\n".join(script_lines)
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete validation."""
        print("🐳 GenomeVault Docker Setup Validator")
        print("=" * 50)
        
        results = {}
        
        # Check Docker installation
        docker_ok, docker_version = self.check_docker_installation()
        results["docker"] = {"installed": docker_ok, "version": docker_version}
        print(f"Docker: {'✅' if docker_ok else '❌'} {docker_version}")
        
        # Check Docker Compose
        compose_ok, compose_version = self.check_docker_compose()
        results["compose"] = {"available": compose_ok, "version": compose_version}
        print(f"Compose: {'✅' if compose_ok else '❌'} {compose_version}")
        
        # Check compose files
        compose_files = self.check_compose_files()
        results["compose_files"] = compose_files
        print(f"Compose Files: {sum(compose_files.values())}/{len(compose_files)} found")
        
        # Check Dockerfiles
        dockerfiles = self.check_dockerfiles()  
        results["dockerfiles"] = dockerfiles
        print(f"Dockerfiles: {sum(dockerfiles.values())}/{len(dockerfiles)} found")
        
        # Check environment
        env_files = self.check_environment_files()
        results["environment"] = env_files
        env_exists = any(f["exists"] for f in env_files.values())
        print(f"Environment: {'✅' if env_exists else '❌'} {'Found' if env_exists else 'Missing'}")
        
        # Check data directories
        volumes = self.check_volumes_and_data()
        results["volumes"] = volumes
        print(f"Data Directories: {sum(1 for v in volumes.values() if v['exists'])}/{len(volumes)} exist")
        
        # Analyze services
        services = self.analyze_compose_services()
        results["services"] = services
        total_services = sum(s.get("service_count", 0) for s in services.values() if isinstance(s, dict))
        print(f"Services Defined: {total_services} across {len(services)} compose files")
        
        # Check ports
        ports = self.check_network_ports()
        results["ports"] = ports
        busy_ports = sum(1 for p in ports.values() if p.get("in_use", False))
        print(f"Port Conflicts: {busy_ports}/{len(ports)} ports in use")
        
        print("\n" + "=" * 50)
        
        # Print issues
        if self.issues:
            print("❌ Issues Found:")
            for issue in self.issues:
                print(f"  - {issue}")
            print()
        
        # Print recommendations  
        if self.recommendations:
            print("💡 Recommendations:")
            for rec in self.recommendations:
                print(f"  - {rec}")
            print()
        
        # Overall status
        critical_ok = docker_ok and compose_ok and env_exists
        if critical_ok:
            print("✅ Docker setup is ready for GenomeVault!")
            if not any(results["compose_files"].values()):
                print("⚠️  No compose files found - services won't be available")
        else:
            print("❌ Docker setup needs attention before running GenomeVault")
        
        results["validation"] = {
            "passed": critical_ok,
            "issues": self.issues,
            "recommendations": self.recommendations
        }
        
        return results


if __name__ == "__main__":
    validator = DockerSetupValidator()
    results = validator.run_validation()
    
    # Generate setup script
    setup_script = validator.generate_setup_script()
    script_path = validator.project_root / "scripts" / "setup_docker.sh"
    
    with open(script_path, "w") as f:
        f.write(setup_script)
    os.chmod(script_path, 0o755)
    
    print(f"\n📝 Generated setup script: {script_path}")
    print("Run ./scripts/setup_docker.sh to automatically set up the Docker environment")
    
    # Exit with appropriate code
    sys.exit(0 if results["validation"]["passed"] else 1)