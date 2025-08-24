#!/usr/bin/env python3
"""
Docker Debug and Status Script for GenomeVault

Comprehensive debugging tool for Docker Compose issues.
Provides detailed diagnostics and fixes common problems.
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time


class DockerDebugger:
    """Debug Docker and Docker Compose issues."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.issues = []
        self.solutions = []
        self.compose_files = [
            "docker-compose.yml",
            "docker-compose.dev.yml",
            "docker-compose.demo.yml", 
            "docker-compose.obsv.yml"
        ]
    
    def log_info(self, message: str):
        """Log info message."""
        print(f"ℹ️  {message}")
    
    def log_success(self, message: str):
        """Log success message."""
        print(f"✅ {message}")
    
    def log_warning(self, message: str):
        """Log warning message."""
        print(f"⚠️  {message}")
    
    def log_error(self, message: str):
        """Log error message."""
        print(f"❌ {message}")
    
    def run_command(self, cmd: List[str], timeout: int = 30) -> Tuple[bool, str, str]:
        """Run command and return success, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=self.project_root
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)
    
    def check_docker_installation(self) -> Dict[str, Any]:
        """Check Docker installation."""
        self.log_info("Checking Docker installation...")
        
        result = {"installed": False, "version": None, "daemon_running": False}
        
        # Check if docker command exists
        if not shutil.which("docker"):
            self.issues.append("Docker command not found in PATH")
            self.solutions.append("Install Docker Desktop: https://docs.docker.com/get-docker/")
            return result
        
        # Get Docker version
        success, stdout, stderr = self.run_command(["docker", "--version"])
        if success:
            result["installed"] = True
            result["version"] = stdout.strip()
            self.log_success(f"Docker installed: {result['version']}")
        else:
            self.issues.append(f"Docker version check failed: {stderr}")
            return result
        
        # Check if daemon is running
        success, stdout, stderr = self.run_command(["docker", "info"])
        if success:
            result["daemon_running"] = True
            self.log_success("Docker daemon is running")
        else:
            result["daemon_running"] = False
            self.log_error("Docker daemon is not running")
            self.issues.append("Docker daemon not accessible")
            self.solutions.append("Start Docker Desktop or run 'sudo systemctl start docker'")
        
        return result
    
    def check_compose_availability(self) -> Dict[str, Any]:
        """Check Docker Compose availability and version."""
        self.log_info("Checking Docker Compose...")
        
        result = {"available": False, "version": None, "type": None}
        
        # Try Docker Compose v2 (plugin)
        success, stdout, stderr = self.run_command(["docker", "compose", "version"])
        if success:
            result["available"] = True
            result["version"] = stdout.strip()
            result["type"] = "docker-compose-plugin"
            self.log_success(f"Docker Compose v2 (plugin): {result['version']}")
            return result
        
        # Try standalone docker-compose
        if shutil.which("docker-compose"):
            success, stdout, stderr = self.run_command(["docker-compose", "--version"])
            if success:
                result["available"] = True
                result["version"] = stdout.strip()
                result["type"] = "docker-compose-standalone"
                self.log_success(f"Docker Compose v1 (standalone): {result['version']}")
                return result
        
        self.issues.append("Docker Compose not found")
        self.solutions.append("Install Docker Compose or update Docker Desktop")
        return result
    
    def check_compose_files(self) -> Dict[str, Any]:
        """Check Docker Compose files for syntax and common issues."""
        self.log_info("Checking Docker Compose files...")
        
        results = {}
        
        for compose_file in self.compose_files:
            file_path = self.project_root / compose_file
            file_result = {"exists": False, "valid": False, "issues": []}
            
            if file_path.exists():
                file_result["exists"] = True
                
                # Check YAML syntax
                try:
                    import yaml
                    with open(file_path) as f:
                        data = yaml.safe_load(f)
                    
                    file_result["valid"] = True
                    self.log_success(f"{compose_file} is valid YAML")
                    
                    # Check for common issues
                    if "services" not in data:
                        file_result["issues"].append("No services defined")
                    
                    # Check for version
                    if "version" not in data:
                        file_result["issues"].append("No version specified")
                    
                    # Check for volume mounts that might fail
                    if "services" in data:
                        for service_name, service in data["services"].items():
                            if "volumes" in service:
                                for volume in service["volumes"]:
                                    if isinstance(volume, str) and volume.startswith("./"):
                                        # Relative path volume
                                        local_path = self.project_root / volume.split(":")[0][2:]
                                        if not local_path.exists():
                                            file_result["issues"].append(
                                                f"Service {service_name}: Volume path {volume.split(':')[0]} doesn't exist"
                                            )
                    
                except ImportError:
                    file_result["issues"].append("PyYAML not available for validation")
                except yaml.YAMLError as e:
                    file_result["valid"] = False
                    file_result["issues"].append(f"YAML syntax error: {e}")
                    self.log_error(f"{compose_file} has YAML errors")
                except Exception as e:
                    file_result["issues"].append(f"Validation error: {e}")
            else:
                self.log_warning(f"{compose_file} not found")
            
            results[compose_file] = file_result
        
        return results
    
    def check_ports(self) -> Dict[str, Any]:
        """Check for port conflicts."""
        self.log_info("Checking for port conflicts...")
        
        common_ports = {
            8000: "GenomeVault API",
            8001: "ZK Prover API", 
            8002: "PIR Server",
            5432: "PostgreSQL",
            6379: "Redis",
            9090: "Prometheus",
            3000: "Grafana",
            5050: "PgAdmin",
            8081: "Redis Commander"
        }
        
        port_status = {}
        
        for port, service in common_ports.items():
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                in_use = result == 0
                port_status[port] = {
                    "service": service,
                    "in_use": in_use,
                    "status": "❌ IN USE" if in_use else "✅ Available"
                }
                
                if in_use:
                    self.issues.append(f"Port {port} ({service}) is already in use")
                    self.solutions.append(f"Stop service using port {port} or change port in compose file")
                
            except Exception as e:
                port_status[port] = {
                    "service": service,
                    "in_use": False,
                    "status": f"❓ Could not check: {e}"
                }
        
        return port_status
    
    def check_volumes_and_data(self) -> Dict[str, Any]:
        """Check data directories and volume mounts."""
        self.log_info("Checking volumes and data directories...")
        
        required_dirs = [
            "data",
            "data/cache",
            "data/encrypted", 
            "data/input",
            "data/output",
            "data/processed",
            "data/raw",
            "logs",
            "keys",
            "config"
        ]
        
        results = {}
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            dir_result = {
                "exists": dir_path.exists(),
                "writable": False,
                "size_mb": 0
            }
            
            if dir_result["exists"]:
                # Check if writable
                try:
                    test_file = dir_path / ".write_test"
                    test_file.touch()
                    test_file.unlink()
                    dir_result["writable"] = True
                except:
                    dir_result["writable"] = False
                    self.issues.append(f"Directory {dir_name} is not writable")
                    self.solutions.append(f"Fix permissions: chmod 755 {dir_path}")
                
                # Get size
                try:
                    total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                    dir_result["size_mb"] = total_size / (1024 * 1024)
                except:
                    pass
            
            results[dir_name] = dir_result
        
        return results
    
    def check_environment(self) -> Dict[str, Any]:
        """Check environment configuration."""
        self.log_info("Checking environment configuration...")
        
        env_file = self.project_root / ".env"
        result = {
            "env_file_exists": env_file.exists(),
            "required_vars": {},
            "issues": []
        }
        
        required_vars = [
            "DATABASE_URL",
            "JWT_SECRET_KEY",
            "API_KEY_SECRET"
        ]
        
        if result["env_file_exists"]:
            try:
                with open(env_file) as f:
                    env_content = f.read()
                
                for var in required_vars:
                    if f"{var}=" in env_content:
                        # Extract value
                        for line in env_content.split('\n'):
                            if line.startswith(f"{var}="):
                                value = line.split('=', 1)[1].strip()
                                result["required_vars"][var] = {
                                    "present": True,
                                    "value_length": len(value),
                                    "is_default": value in [
                                        "change-this-secret-key-in-production",
                                        "change-this-api-key-secret"
                                    ]
                                }
                                break
                    else:
                        result["required_vars"][var] = {"present": False}
                        result["issues"].append(f"Missing environment variable: {var}")
                
            except Exception as e:
                result["issues"].append(f"Error reading .env file: {e}")
        else:
            result["issues"].append(".env file not found")
            self.solutions.append("Run './scripts/docker_compose_wrapper.sh setup' to create .env file")
        
        return result
    
    def test_docker_operations(self) -> Dict[str, Any]:
        """Test basic Docker operations."""
        self.log_info("Testing Docker operations...")
        
        results = {}
        
        # Test image pull
        test_image = "hello-world"
        self.log_info(f"Testing image pull: {test_image}")
        success, stdout, stderr = self.run_command(["docker", "pull", test_image])
        results["image_pull"] = {
            "success": success,
            "output": stdout if success else stderr
        }
        
        if success:
            self.log_success("Image pull successful")
            
            # Test container run
            self.log_info("Testing container run...")
            success, stdout, stderr = self.run_command(["docker", "run", "--rm", test_image])
            results["container_run"] = {
                "success": success,
                "output": stdout if success else stderr
            }
            
            if success:
                self.log_success("Container run successful")
            else:
                self.log_error(f"Container run failed: {stderr}")
        else:
            self.log_error(f"Image pull failed: {stderr}")
        
        return results
    
    def diagnose_compose_issues(self, compose_file: str = "docker-compose.dev.yml") -> Dict[str, Any]:
        """Diagnose specific compose file issues."""
        self.log_info(f"Diagnosing {compose_file}...")
        
        results = {"validation": {}, "config": {}, "services": {}}
        
        # Validate compose file
        success, stdout, stderr = self.run_command([
            "docker", "compose", "-f", compose_file, "config"
        ])
        
        results["validation"] = {
            "success": success,
            "output": stdout if success else stderr
        }
        
        if success:
            self.log_success(f"{compose_file} validation passed")
            
            # Try dry-run
            success, stdout, stderr = self.run_command([
                "docker", "compose", "-f", compose_file, "up", "--dry-run"
            ])
            
            results["config"] = {
                "success": success,
                "output": stdout if success else stderr
            }
            
            if success:
                self.log_success("Dry-run successful")
            else:
                self.log_error(f"Dry-run failed: {stderr}")
        else:
            self.log_error(f"{compose_file} validation failed: {stderr}")
            self.issues.append(f"Compose file validation failed: {stderr}")
        
        return results
    
    def generate_fix_script(self) -> str:
        """Generate a script to fix identified issues."""
        script_lines = [
            "#!/bin/bash",
            "# GenomeVault Docker Fix Script",
            "# Generated automatically by docker_debug.py",
            "",
            "set -e",
            "",
            "echo '🔧 Fixing GenomeVault Docker issues...'",
            ""
        ]
        
        # Create missing directories
        script_lines.extend([
            "# Create missing data directories",
            "mkdir -p data/{cache,encrypted,input,output,processed,raw}",
            "mkdir -p logs keys config",
            ""
        ])
        
        # Fix permissions
        script_lines.extend([
            "# Fix directory permissions",
            "chmod 755 data logs keys config",
            "chmod -R 755 data/*",
            ""
        ])
        
        # Create .env if missing
        script_lines.extend([
            "# Create .env file if missing",
            "if [ ! -f .env ]; then",
            "  ./scripts/docker_compose_wrapper.sh setup",
            "fi",
            ""
        ])
        
        # Start Docker if needed
        script_lines.extend([
            "# Start Docker Desktop if not running",
            "if ! docker info >/dev/null 2>&1; then",
            "  echo 'Starting Docker Desktop...'", 
            "  open -a Docker",
            "  sleep 10",
            "fi",
            ""
        ])
        
        # Add specific solutions
        for solution in self.solutions:
            script_lines.append(f"# {solution}")
        
        script_lines.extend([
            "",
            "echo '✅ Fix script complete!'",
            "echo 'You can now try:'",
            "echo '  ./scripts/docker_compose_wrapper.sh dev'",
            "echo '  ./scripts/docker_compose_wrapper.sh status'"
        ])
        
        return "\n".join(script_lines)
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run all diagnostic checks."""
        print("🐳 GenomeVault Docker Comprehensive Debug")
        print("=" * 60)
        
        results = {}
        
        # Check Docker installation
        results["docker"] = self.check_docker_installation()
        
        # Check Compose availability
        results["compose"] = self.check_compose_availability()
        
        # Check compose files
        results["compose_files"] = self.check_compose_files()
        
        # Check ports
        results["ports"] = self.check_ports()
        
        # Check volumes and data
        results["volumes"] = self.check_volumes_and_data()
        
        # Check environment
        results["environment"] = self.check_environment()
        
        # Test Docker operations if daemon is running
        if results["docker"]["daemon_running"]:
            results["docker_test"] = self.test_docker_operations()
        
        # Diagnose compose issues if compose is available
        if results["compose"]["available"]:
            results["compose_diagnosis"] = self.diagnose_compose_issues()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)
        
        if self.issues:
            print("❌ Issues Found:")
            for issue in self.issues:
                print(f"  • {issue}")
            print()
        
        if self.solutions:
            print("💡 Recommended Solutions:")
            for solution in self.solutions:
                print(f"  • {solution}")
            print()
        
        # Overall status
        docker_ok = results["docker"]["installed"] and results["docker"]["daemon_running"]
        compose_ok = results["compose"]["available"]
        
        if docker_ok and compose_ok:
            print("✅ Docker setup appears to be working!")
            print("You can try: ./scripts/docker_compose_wrapper.sh dev")
        else:
            print("❌ Docker setup needs attention")
            print("Run the generated fix script or follow the solutions above")
        
        # Generate fix script
        fix_script = self.generate_fix_script()
        fix_script_path = self.project_root / "scripts" / "docker_fix.sh"
        
        with open(fix_script_path, "w") as f:
            f.write(fix_script)
        os.chmod(fix_script_path, 0o755)
        
        print(f"\n📝 Generated fix script: {fix_script_path}")
        print("Run: ./scripts/docker_fix.sh")
        
        return results


if __name__ == "__main__":
    debugger = DockerDebugger()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "compose-check":
            file_name = sys.argv[2] if len(sys.argv) > 2 else "docker-compose.dev.yml"
            results = debugger.diagnose_compose_issues(file_name)
            print(json.dumps(results, indent=2))
        
        elif command == "ports":
            results = debugger.check_ports()
            print(json.dumps(results, indent=2))
        
        elif command == "fix":
            # Just generate and run fix script
            fix_script = debugger.generate_fix_script()
            print(fix_script)
        
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    else:
        # Run comprehensive check
        results = debugger.run_comprehensive_check()
        
        # Save detailed results
        results_file = debugger.project_root / "docker_debug_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📋 Detailed results saved to: {results_file}")