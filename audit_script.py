#!/usr/bin/env python3
"""
GenomeVault 3.0 Comprehensive Codebase Audit Script
====================================================
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict
import ast
import re

@dataclass
class AuditResult:
    """Container for audit results"""
    category: str
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    file_path: str
    line_number: int
    message: str
    suggestion: str = ""

class GenomeVaultAuditor:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.genomevault_dir = self.project_root / "genomevault"
        self.results: List[AuditResult] = []
        
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run all audit checks and return comprehensive results"""
        print("🔍 Starting GenomeVault 3.0 Comprehensive Audit...")
        print("=" * 60)
        
        # Core audits
        self.audit_architecture()
        self.audit_imports_and_dependencies()
        self.audit_security_patterns()
        self.audit_code_quality()
        self.audit_test_coverage()
        self.audit_documentation()
        self.audit_compliance()
        
        return self.generate_audit_report()
    
    def audit_architecture(self):
        """Audit system architecture and module organization"""
        print("🏗️  Auditing Architecture...")
        
        # Check if all required modules exist
        required_modules = [
            "local_processing", "hypervector_transform", "zk_proofs", 
            "pir", "blockchain", "api", "utils", "core"
        ]
        
        for module in required_modules:
            module_path = self.genomevault_dir / module
            if not module_path.exists():
                self.results.append(AuditResult(
                    category="Architecture",
                    severity="critical",
                    file_path=str(module_path),
                    line_number=0,
                    message=f"Required module '{module}' is missing",
                    suggestion=f"Create {module} module with proper structure"
                ))
            elif not (module_path / "__init__.py").exists():
                self.results.append(AuditResult(
                    category="Architecture",
                    severity="medium",
                    file_path=str(module_path),
                    line_number=0,
                    message=f"Module '{module}' missing __init__.py",
                    suggestion="Add __init__.py to make it a proper Python package"
                ))
    
    def audit_imports_and_dependencies(self):
        """Audit import structure and dependencies"""
        print("📦 Auditing Imports and Dependencies...")
        
        for py_file in self.genomevault_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for dangerous imports
                dangerous_patterns = [
                    ('pickle', 'Use safer alternatives like json'),
                    ('eval', 'Avoid eval() for security reasons'),
                    ('exec', 'Avoid exec() for security reasons'),
                ]
                
                for pattern, message in dangerous_patterns:
                    if pattern in content:
                        self.results.append(AuditResult(
                            category="Security",
                            severity="high",
                            file_path=str(py_file),
                            line_number=0,
                            message=f"Dangerous import/usage: {pattern} - {message}",
                            suggestion="Use secure alternatives"
                        ))
                            
            except Exception as e:
                self.results.append(AuditResult(
                    category="Import Analysis",
                    severity="low",
                    file_path=str(py_file),
                    line_number=0,
                    message=f"Could not analyze imports: {e}",
                    suggestion="Check file format and encoding"
                ))
    
    def audit_security_patterns(self):
        """Audit for security anti-patterns and vulnerabilities"""
        print("🔒 Auditing Security Patterns...")
        
        security_patterns = {
            r'password\s*=\s*["\'][^"\']+["\']': 'Hardcoded password detected',
            r'secret\s*=\s*["\'][^"\']+["\']': 'Hardcoded secret detected',
            r'api_key\s*=\s*["\'][^"\']+["\']': 'Hardcoded API key detected',
            r'shell\s*=\s*True': 'Shell injection risk',
            r'random\.random\(\)': 'Use secrets module for cryptographic randomness'
        }
        
        for py_file in self.genomevault_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for line_no, line in enumerate(lines, 1):
                    for pattern, message in security_patterns.items():
                        if re.search(pattern, line, re.IGNORECASE):
                            self.results.append(AuditResult(
                                category="Security",
                                severity="high",
                                file_path=str(py_file),
                                line_number=line_no,
                                message=message,
                                suggestion="Review and use secure alternatives"
                            ))
                            
            except Exception as e:
                pass  # Skip files that can't be read
    
    def audit_code_quality(self):
        """Audit code quality metrics"""
        print("✨ Auditing Code Quality...")
        
        for py_file in self.genomevault_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                # Check line length
                long_lines = 0
                for line_no, line in enumerate(lines, 1):
                    if len(line) > 100:
                        long_lines += 1
                
                if long_lines > 10:  # More than 10 long lines
                    self.results.append(AuditResult(
                        category="Code Style",
                        severity="low",
                        file_path=str(py_file),
                        line_number=0,
                        message=f"File has {long_lines} lines exceeding 100 characters",
                        suggestion="Break long lines for better readability"
                    ))
                
                # Parse AST for complexity analysis
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Count parameters
                            if len(node.args.args) > 5:
                                self.results.append(AuditResult(
                                    category="Code Quality",
                                    severity="medium",
                                    file_path=str(py_file),
                                    line_number=node.lineno,
                                    message=f"Function '{node.name}' has too many parameters ({len(node.args.args)})",
                                    suggestion="Consider using dataclasses or reducing parameters"
                                ))
                            
                            # Check for missing docstring
                            if not ast.get_docstring(node):
                                self.results.append(AuditResult(
                                    category="Documentation",
                                    severity="low",
                                    file_path=str(py_file),
                                    line_number=node.lineno,
                                    message=f"Function '{node.name}' missing docstring",
                                    suggestion="Add descriptive docstring"
                                ))
                        
                        elif isinstance(node, ast.ClassDef):
                            # Check for missing docstring
                            if not ast.get_docstring(node):
                                self.results.append(AuditResult(
                                    category="Documentation",
                                    severity="low",
                                    file_path=str(py_file),
                                    line_number=node.lineno,
                                    message=f"Class '{node.name}' missing docstring",
                                    suggestion="Add descriptive docstring"
                                ))
                            
                            # Count methods
                            method_count = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
                            if method_count > 20:
                                self.results.append(AuditResult(
                                    category="Code Quality",
                                    severity="medium",
                                    file_path=str(py_file),
                                    line_number=node.lineno,
                                    message=f"Class '{node.name}' has too many methods ({method_count})",
                                    suggestion="Consider splitting into smaller classes"
                                ))
                except SyntaxError:
                    self.results.append(AuditResult(
                        category="Syntax",
                        severity="critical",
                        file_path=str(py_file),
                        line_number=0,
                        message="Syntax error in file",
                        suggestion="Fix syntax errors"
                    ))
                        
            except Exception as e:
                pass  # Skip files that can't be analyzed
    
    def audit_test_coverage(self):
        """Audit test coverage and test quality"""
        print("🧪 Auditing Test Coverage...")
        
        tests_dir = self.project_root / "tests"
        if not tests_dir.exists():
            self.results.append(AuditResult(
                category="Testing",
                severity="critical",
                file_path=str(tests_dir),
                line_number=0,
                message="Tests directory missing",
                suggestion="Create comprehensive test suite"
            ))
            return
        
        # Check test file coverage
        source_files = list(self.genomevault_dir.rglob("*.py"))
        test_files = list(tests_dir.rglob("test_*.py"))
        
        coverage_ratio = len(test_files) / max(len(source_files), 1)
        if coverage_ratio < 0.5:
            self.results.append(AuditResult(
                category="Testing",
                severity="high",
                file_path=str(tests_dir),
                line_number=0,
                message=f"Low test file coverage ({coverage_ratio:.2%})",
                suggestion="Add more test files to reach at least 50% coverage"
            ))
        
        # Check for specific test patterns
        required_test_categories = [
            "test_security", "test_privacy", "test_zk_proofs", 
            "test_pir", "test_hypervector"
        ]
        
        existing_tests = [f.stem for f in tests_dir.rglob("test_*.py")]
        
        for required_test in required_test_categories:
            if not any(required_test in test for test in existing_tests):
                self.results.append(AuditResult(
                    category="Testing",
                    severity="medium",
                    file_path=str(tests_dir),
                    line_number=0,
                    message=f"Missing critical test category: {required_test}",
                    suggestion=f"Add {required_test}.py with comprehensive tests"
                ))
    
    def audit_documentation(self):
        """Audit documentation completeness"""
        print("📚 Auditing Documentation...")
        
        required_docs = [
            "README.md", "CONTRIBUTING.md", "LICENSE", 
            "INSTALL.md", "docs/api.md", "docs/architecture.md"
        ]
        
        for doc in required_docs:
            doc_path = self.project_root / doc
            if not doc_path.exists():
                self.results.append(AuditResult(
                    category="Documentation",
                    severity="medium",
                    file_path=str(doc_path),
                    line_number=0,
                    message=f"Missing required documentation: {doc}",
                    suggestion=f"Create {doc} with comprehensive information"
                ))
    
    def audit_compliance(self):
        """Audit regulatory compliance patterns"""
        print("⚖️  Auditing Compliance...")
        
        # Check for HIPAA compliance patterns
        hipaa_keywords = [
            "hipaa", "phi", "protected_health_information", 
            "patient_data", "medical_record"
        ]
        
        compliance_found = False
        for py_file in self.genomevault_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for keyword in hipaa_keywords:
                    if keyword in content:
                        compliance_found = True
                        break
                        
            except Exception:
                continue
        
        if not compliance_found:
            self.results.append(AuditResult(
                category="Compliance",
                severity="medium",
                file_path="genomevault/",
                line_number=0,
                message="No HIPAA compliance patterns detected",
                suggestion="Implement HIPAA compliance checks and documentation"
            ))
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        print("\n📊 Generating Audit Report...")
        
        # Categorize results
        by_category = defaultdict(list)
        by_severity = defaultdict(list)
        
        for result in self.results:
            by_category[result.category].append(result)
            by_severity[result.severity].append(result)
        
        # Calculate metrics
        total_issues = len(self.results)
        critical_issues = len(by_severity['critical'])
        high_issues = len(by_severity['high'])
        medium_issues = len(by_severity['medium'])
        low_issues = len(by_severity['low'])
        
        # Calculate health score
        if critical_issues > 0:
            health = "CRITICAL"
        elif high_issues > 10:
            health = "POOR"
        elif high_issues > 5:
            health = "FAIR"
        elif total_issues > 50:
            health = "GOOD"
        else:
            health = "EXCELLENT"
        
        # Generate summary
        summary = {
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "high_issues": high_issues,
            "medium_issues": medium_issues,
            "low_issues": low_issues,
            "by_category": {cat: len(issues) for cat, issues in by_category.items()},
            "overall_health": health
        }
        
        # Generate recommendations
        recommendations = []
        if by_category["Security"]:
            recommendations.append("🔒 Address security vulnerabilities immediately")
        if by_category["Architecture"]:
            recommendations.append("🏗️ Refactor architecture issues before scaling")
        if by_category["Testing"]:
            recommendations.append("🧪 Implement comprehensive test suite")
        if by_category["Documentation"]:
            recommendations.append("📚 Complete missing documentation")
        if by_category["Compliance"]:
            recommendations.append("⚖️ Implement compliance frameworks")
        
        next_steps = [
            "1. Fix all critical and high-severity issues",
            "2. Set up continuous integration with linting",
            "3. Implement automated security scanning",
            "4. Add comprehensive test coverage (target: 80%+)",
            "5. Complete documentation for all public APIs",
            "6. Set up code review process for all changes",
            "7. Implement HIPAA compliance validation",
            "8. Add performance monitoring and benchmarks"
        ]
        
        return {
            "summary": summary,
            "results": [vars(r) for r in self.results],
            "recommendations": recommendations,
            "next_steps": next_steps
        }

def main():
    """Main audit execution"""
    project_root = "/Users/rohanvinaik/genomevault"
    
    if not os.path.exists(project_root):
        print(f"❌ Project root not found: {project_root}")
        sys.exit(1)
    
    auditor = GenomeVaultAuditor(project_root)
    report = auditor.run_comprehensive_audit()
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 AUDIT SUMMARY")
    print("="*60)
    print(f"Total Issues: {report['summary']['total_issues']}")
    print(f"Critical: {report['summary']['critical_issues']}")
    print(f"High: {report['summary']['high_issues']}")
    print(f"Medium: {report['summary']['medium_issues']}")
    print(f"Low: {report['summary']['low_issues']}")
    print(f"Overall Health: {report['summary']['overall_health']}")
    
    print(f"\n📊 Issues by Category:")
    for category, count in report['summary']['by_category'].items():
        print(f"  {category}: {count}")
    
    print(f"\n💡 Key Recommendations:")
    for rec in report['recommendations']:
        print(f"  {rec}")
    
    print(f"\n🚀 Next Steps:")
    for step in report['next_steps']:
        print(f"  {step}")
    
    # Save detailed report
    with open(f"{project_root}/audit_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📝 Detailed report saved to: {project_root}/audit_report.json")
    
    return report

if __name__ == "__main__":
    main()
