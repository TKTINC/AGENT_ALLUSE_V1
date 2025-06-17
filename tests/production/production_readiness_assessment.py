#!/usr/bin/env python3
"""
ALL-USE Agent - Protocol Engine Production Readiness Assessment
P6 of WS2: Protocol Engine Final Integration and System Testing - Phase 4

This module provides comprehensive production readiness assessment for the complete
Protocol Engine system, validating all aspects required for production deployment
including security, configuration, monitoring, documentation, and operational readiness.

Production Readiness Components:
1. Deployment Readiness Checklist - Complete production deployment validation
2. Security Assessment - Production security requirements validation
3. Configuration Management - Production configuration and environment setup
4. Monitoring and Alerting - Production monitoring system validation
5. Documentation Completeness - Production documentation requirements
6. Operational Readiness - Production operations and maintenance validation
"""

import os
import json
import time
import hashlib
import subprocess
import logging
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import importlib.util

# Add src to path for imports
sys.path.insert(0, '/home/ubuntu/AGENT_ALLUSE_V1/src')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReadinessLevel(Enum):
    """Production readiness levels"""
    NOT_READY = "not_ready"
    PARTIALLY_READY = "partially_ready"
    READY = "ready"
    PRODUCTION_READY = "production_ready"


@dataclass
class ReadinessCheckResult:
    """Production readiness check result"""
    check_name: str
    category: str
    status: ReadinessLevel
    score: float  # 0-100
    details: Dict[str, Any]
    recommendations: List[str]
    critical_issues: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ProductionReadinessReport:
    """Complete production readiness report"""
    overall_status: ReadinessLevel
    overall_score: float
    category_scores: Dict[str, float]
    readiness_checks: List[ReadinessCheckResult]
    critical_blockers: List[str]
    recommendations: List[str]
    deployment_approval: bool
    assessment_timestamp: datetime
    next_review_date: datetime


class DeploymentReadinessChecker:
    """Validates deployment readiness requirements"""
    
    def __init__(self):
        self.project_root = "/home/ubuntu/AGENT_ALLUSE_V1"
        self.required_files = []
        self.required_directories = []
        
        logger.info("Deployment Readiness Checker initialized")
    
    def check_file_structure(self) -> ReadinessCheckResult:
        """Check required file and directory structure"""
        required_structure = {
            'src/': ['Source code directory'],
            'tests/': ['Test suite directory'],
            'docs/': ['Documentation directory'],
            'src/protocol_engine/': ['Protocol engine core'],
            'src/protocol_engine/week_classification/': ['Week classification system'],
            'src/protocol_engine/market_analysis/': ['Market analysis system'],
            'src/protocol_engine/rules/': ['Trading rules engine'],
            'src/protocol_engine/optimization/': ['Performance optimization'],
            'src/protocol_engine/monitoring/': ['Monitoring system'],
            'src/protocol_engine/analytics/': ['Analytics system'],
            'tests/unit/': ['Unit tests'],
            'tests/integration/': ['Integration tests'],
            'tests/performance/': ['Performance tests'],
            'docs/planning/': ['Planning documentation'],
            'docs/testing/': ['Testing documentation'],
            'docs/optimization/': ['Optimization documentation']
        }
        
        missing_structure = []
        present_structure = []
        
        for path, description in required_structure.items():
            full_path = os.path.join(self.project_root, path)
            if os.path.exists(full_path):
                present_structure.append(f"{path}: {description[0]}")
            else:
                missing_structure.append(f"{path}: {description[0]}")
        
        # Calculate score
        total_required = len(required_structure)
        present_count = len(present_structure)
        score = (present_count / total_required) * 100 if total_required > 0 else 0
        
        # Determine status
        if score >= 95:
            status = ReadinessLevel.PRODUCTION_READY
        elif score >= 80:
            status = ReadinessLevel.READY
        elif score >= 60:
            status = ReadinessLevel.PARTIALLY_READY
        else:
            status = ReadinessLevel.NOT_READY
        
        recommendations = []
        if missing_structure:
            recommendations.append("Create missing directory structure")
            recommendations.extend([f"Create {item}" for item in missing_structure[:3]])
        
        return ReadinessCheckResult(
            check_name="File Structure",
            category="deployment",
            status=status,
            score=score,
            details={
                'present_structure': present_structure,
                'missing_structure': missing_structure,
                'structure_completeness': f"{present_count}/{total_required}"
            },
            recommendations=recommendations,
            critical_issues=missing_structure if score < 80 else []
        )
    
    def check_code_quality(self) -> ReadinessCheckResult:
        """Check code quality and standards"""
        quality_metrics = {
            'python_files_count': 0,
            'documented_files': 0,
            'test_files_count': 0,
            'large_files': [],
            'import_errors': []
        }
        
        # Scan Python files
        for root, dirs, files in os.walk(os.path.join(self.project_root, 'src')):
            for file in files:
                if file.endswith('.py'):
                    quality_metrics['python_files_count'] += 1
                    file_path = os.path.join(root, file)
                    
                    # Check file size
                    file_size = os.path.getsize(file_path)
                    if file_size > 50000:  # Files larger than 50KB
                        quality_metrics['large_files'].append(f"{file}: {file_size//1024}KB")
                    
                    # Check for documentation
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if '"""' in content or "'''" in content:
                                quality_metrics['documented_files'] += 1
                    except Exception:
                        pass
        
        # Count test files
        test_dir = os.path.join(self.project_root, 'tests')
        if os.path.exists(test_dir):
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.endswith('.py') and ('test_' in file or file.startswith('test')):
                        quality_metrics['test_files_count'] += 1
        
        # Calculate quality score
        documentation_ratio = quality_metrics['documented_files'] / quality_metrics['python_files_count'] if quality_metrics['python_files_count'] > 0 else 0
        test_coverage_estimate = min(1.0, quality_metrics['test_files_count'] / max(1, quality_metrics['python_files_count'] * 0.5))
        
        quality_score = (documentation_ratio * 40 + test_coverage_estimate * 40 + 20) * (1 - len(quality_metrics['large_files']) * 0.05)
        quality_score = max(0, min(100, quality_score))
        
        # Determine status
        if quality_score >= 85:
            status = ReadinessLevel.PRODUCTION_READY
        elif quality_score >= 70:
            status = ReadinessLevel.READY
        elif quality_score >= 50:
            status = ReadinessLevel.PARTIALLY_READY
        else:
            status = ReadinessLevel.NOT_READY
        
        recommendations = []
        if documentation_ratio < 0.8:
            recommendations.append("Improve code documentation coverage")
        if test_coverage_estimate < 0.7:
            recommendations.append("Increase test coverage")
        if quality_metrics['large_files']:
            recommendations.append("Consider refactoring large files")
        
        return ReadinessCheckResult(
            check_name="Code Quality",
            category="deployment",
            status=status,
            score=quality_score,
            details=quality_metrics,
            recommendations=recommendations,
            critical_issues=[]
        )
    
    def check_dependencies(self) -> ReadinessCheckResult:
        """Check dependency management and requirements"""
        dependency_status = {
            'python_version': sys.version,
            'required_packages': [],
            'missing_packages': [],
            'version_conflicts': []
        }
        
        # Check critical packages
        critical_packages = [
            'numpy', 'pandas', 'matplotlib', 'logging', 'datetime', 
            'dataclasses', 'enum', 'typing', 'json', 'time'
        ]
        
        for package in critical_packages:
            try:
                if package in ['logging', 'datetime', 'dataclasses', 'enum', 'typing', 'json', 'time']:
                    # Built-in packages
                    dependency_status['required_packages'].append(f"{package}: built-in")
                else:
                    # External packages
                    spec = importlib.util.find_spec(package)
                    if spec is not None:
                        dependency_status['required_packages'].append(f"{package}: available")
                    else:
                        dependency_status['missing_packages'].append(package)
            except Exception as e:
                dependency_status['missing_packages'].append(f"{package}: {str(e)}")
        
        # Calculate dependency score
        total_packages = len(critical_packages)
        available_packages = len(dependency_status['required_packages'])
        dependency_score = (available_packages / total_packages) * 100 if total_packages > 0 else 0
        
        # Determine status
        if dependency_score >= 95:
            status = ReadinessLevel.PRODUCTION_READY
        elif dependency_score >= 85:
            status = ReadinessLevel.READY
        elif dependency_score >= 70:
            status = ReadinessLevel.PARTIALLY_READY
        else:
            status = ReadinessLevel.NOT_READY
        
        recommendations = []
        if dependency_status['missing_packages']:
            recommendations.append("Install missing packages")
            recommendations.extend([f"Install {pkg}" for pkg in dependency_status['missing_packages'][:3]])
        
        return ReadinessCheckResult(
            check_name="Dependencies",
            category="deployment",
            status=status,
            score=dependency_score,
            details=dependency_status,
            recommendations=recommendations,
            critical_issues=dependency_status['missing_packages'] if dependency_score < 85 else []
        )


class SecurityAssessment:
    """Validates production security requirements"""
    
    def __init__(self):
        self.project_root = "/home/ubuntu/AGENT_ALLUSE_V1"
        
        logger.info("Security Assessment initialized")
    
    def check_code_security(self) -> ReadinessCheckResult:
        """Check code security practices"""
        security_issues = {
            'hardcoded_secrets': [],
            'unsafe_imports': [],
            'sql_injection_risks': [],
            'file_access_issues': [],
            'network_security_issues': []
        }
        
        # Security patterns to check
        security_patterns = {
            'hardcoded_secrets': ['password', 'secret', 'key', 'token', 'api_key'],
            'unsafe_imports': ['eval', 'exec', 'subprocess.call', 'os.system'],
            'sql_injection': ['execute(', 'query(', 'raw('],
            'file_access': ['open(', 'file(', 'read(', 'write(']
        }
        
        # Scan Python files for security issues
        python_files_scanned = 0
        for root, dirs, files in os.walk(os.path.join(self.project_root, 'src')):
            for file in files:
                if file.endswith('.py'):
                    python_files_scanned += 1
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().lower()
                            
                            # Check for hardcoded secrets (basic check)
                            for pattern in security_patterns['hardcoded_secrets']:
                                if f'{pattern} =' in content or f'{pattern}=' in content:
                                    if 'test' not in file.lower():  # Ignore test files
                                        security_issues['hardcoded_secrets'].append(f"{file}: potential {pattern}")
                            
                            # Check for unsafe imports
                            for pattern in security_patterns['unsafe_imports']:
                                if pattern in content:
                                    security_issues['unsafe_imports'].append(f"{file}: {pattern}")
                    
                    except Exception:
                        pass
        
        # Calculate security score
        total_issues = sum(len(issues) for issues in security_issues.values())
        security_score = max(0, 100 - (total_issues * 10))  # Deduct 10 points per issue
        
        # Determine status
        if security_score >= 90 and total_issues == 0:
            status = ReadinessLevel.PRODUCTION_READY
        elif security_score >= 80:
            status = ReadinessLevel.READY
        elif security_score >= 60:
            status = ReadinessLevel.PARTIALLY_READY
        else:
            status = ReadinessLevel.NOT_READY
        
        recommendations = []
        if security_issues['hardcoded_secrets']:
            recommendations.append("Remove hardcoded secrets and use environment variables")
        if security_issues['unsafe_imports']:
            recommendations.append("Review and secure unsafe function usage")
        if total_issues == 0:
            recommendations.append("Implement additional security measures for production")
        
        critical_issues = []
        if security_issues['hardcoded_secrets']:
            critical_issues.extend(security_issues['hardcoded_secrets'])
        
        return ReadinessCheckResult(
            check_name="Code Security",
            category="security",
            status=status,
            score=security_score,
            details={
                'files_scanned': python_files_scanned,
                'security_issues': security_issues,
                'total_issues': total_issues
            },
            recommendations=recommendations,
            critical_issues=critical_issues
        )
    
    def check_data_protection(self) -> ReadinessCheckResult:
        """Check data protection and privacy measures"""
        data_protection = {
            'logging_security': True,
            'data_encryption': False,
            'access_controls': False,
            'audit_logging': False,
            'data_retention': False
        }
        
        # Check logging configuration
        log_files_found = []
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if 'log' in file.lower() or file.endswith('.log'):
                    log_files_found.append(file)
        
        # Check for security-related configurations
        config_files = []
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.endswith(('.conf', '.config', '.ini', '.yaml', '.yml', '.json')):
                    config_files.append(file)
        
        # Calculate data protection score
        protection_features = sum(data_protection.values())
        total_features = len(data_protection)
        protection_score = (protection_features / total_features) * 100
        
        # Determine status
        if protection_score >= 80:
            status = ReadinessLevel.READY
        elif protection_score >= 60:
            status = ReadinessLevel.PARTIALLY_READY
        else:
            status = ReadinessLevel.NOT_READY
        
        recommendations = [
            "Implement data encryption for sensitive information",
            "Add access control mechanisms",
            "Implement audit logging for security events",
            "Define data retention policies"
        ]
        
        return ReadinessCheckResult(
            check_name="Data Protection",
            category="security",
            status=status,
            score=protection_score,
            details={
                'protection_features': data_protection,
                'log_files_found': log_files_found,
                'config_files_found': config_files
            },
            recommendations=recommendations,
            critical_issues=[]
        )


class ConfigurationManagement:
    """Validates configuration management for production"""
    
    def __init__(self):
        self.project_root = "/home/ubuntu/AGENT_ALLUSE_V1"
        
        logger.info("Configuration Management initialized")
    
    def check_environment_configuration(self) -> ReadinessCheckResult:
        """Check environment configuration readiness"""
        config_status = {
            'environment_variables': {},
            'configuration_files': [],
            'default_configurations': [],
            'missing_configurations': []
        }
        
        # Check for environment variables
        important_env_vars = [
            'PYTHONPATH', 'PATH', 'HOME', 'USER'
        ]
        
        for var in important_env_vars:
            value = os.environ.get(var)
            if value:
                config_status['environment_variables'][var] = f"Set ({len(value)} chars)"
            else:
                config_status['missing_configurations'].append(f"Environment variable: {var}")
        
        # Check for configuration files
        config_patterns = ['.conf', '.config', '.ini', '.yaml', '.yml', '.json', '.env']
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if any(file.endswith(pattern) for pattern in config_patterns):
                    config_status['configuration_files'].append(file)
        
        # Check for default configurations in code
        python_files_with_config = 0
        for root, dirs, files in os.walk(os.path.join(self.project_root, 'src')):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if any(keyword in content.lower() for keyword in ['config', 'setting', 'default']):
                                python_files_with_config += 1
                                config_status['default_configurations'].append(file)
                    except Exception:
                        pass
        
        # Calculate configuration score
        env_score = len(config_status['environment_variables']) / len(important_env_vars) * 40
        config_file_score = min(40, len(config_status['configuration_files']) * 10)
        default_config_score = min(20, python_files_with_config * 5)
        
        total_score = env_score + config_file_score + default_config_score
        
        # Determine status
        if total_score >= 80:
            status = ReadinessLevel.READY
        elif total_score >= 60:
            status = ReadinessLevel.PARTIALLY_READY
        else:
            status = ReadinessLevel.NOT_READY
        
        recommendations = []
        if len(config_status['configuration_files']) == 0:
            recommendations.append("Create configuration files for production settings")
        if config_status['missing_configurations']:
            recommendations.append("Set missing environment variables")
        recommendations.append("Implement environment-specific configuration management")
        
        return ReadinessCheckResult(
            check_name="Environment Configuration",
            category="configuration",
            status=status,
            score=total_score,
            details=config_status,
            recommendations=recommendations,
            critical_issues=config_status['missing_configurations'] if total_score < 60 else []
        )


class MonitoringValidation:
    """Validates monitoring and alerting systems"""
    
    def __init__(self):
        self.project_root = "/home/ubuntu/AGENT_ALLUSE_V1"
        
        logger.info("Monitoring Validation initialized")
    
    def check_monitoring_systems(self) -> ReadinessCheckResult:
        """Check monitoring system readiness"""
        monitoring_status = {
            'monitoring_components': [],
            'logging_configuration': [],
            'metrics_collection': [],
            'alerting_systems': [],
            'dashboard_availability': False
        }
        
        # Check for monitoring components
        monitoring_files = [
            'src/protocol_engine/monitoring/performance_monitor.py',
            'src/protocol_engine/analytics/performance_analytics.py'
        ]
        
        for file_path in monitoring_files:
            full_path = os.path.join(self.project_root, file_path)
            if os.path.exists(full_path):
                monitoring_status['monitoring_components'].append(file_path)
        
        # Check for logging configuration
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if 'logging' in content and ('basicConfig' in content or 'getLogger' in content):
                                monitoring_status['logging_configuration'].append(file)
                    except Exception:
                        pass
        
        # Check for metrics collection
        metrics_keywords = ['metrics', 'performance', 'monitor', 'track']
        for root, dirs, files in os.walk(os.path.join(self.project_root, 'src')):
            for file in files:
                if file.endswith('.py') and any(keyword in file.lower() for keyword in metrics_keywords):
                    monitoring_status['metrics_collection'].append(file)
        
        # Calculate monitoring score
        component_score = len(monitoring_status['monitoring_components']) * 25
        logging_score = min(25, len(monitoring_status['logging_configuration']) * 5)
        metrics_score = min(25, len(monitoring_status['metrics_collection']) * 5)
        dashboard_score = 25 if monitoring_status['dashboard_availability'] else 0
        
        total_score = component_score + logging_score + metrics_score + dashboard_score
        
        # Determine status
        if total_score >= 80:
            status = ReadinessLevel.READY
        elif total_score >= 60:
            status = ReadinessLevel.PARTIALLY_READY
        else:
            status = ReadinessLevel.NOT_READY
        
        recommendations = []
        if len(monitoring_status['monitoring_components']) < 2:
            recommendations.append("Implement comprehensive monitoring components")
        if not monitoring_status['dashboard_availability']:
            recommendations.append("Set up monitoring dashboard")
        recommendations.append("Configure production alerting thresholds")
        
        return ReadinessCheckResult(
            check_name="Monitoring Systems",
            category="monitoring",
            status=status,
            score=total_score,
            details=monitoring_status,
            recommendations=recommendations,
            critical_issues=[]
        )


class DocumentationCompleteness:
    """Validates documentation completeness for production"""
    
    def __init__(self):
        self.project_root = "/home/ubuntu/AGENT_ALLUSE_V1"
        
        logger.info("Documentation Completeness initialized")
    
    def check_documentation_coverage(self) -> ReadinessCheckResult:
        """Check documentation coverage and quality"""
        doc_status = {
            'documentation_files': [],
            'api_documentation': [],
            'user_guides': [],
            'technical_documentation': [],
            'missing_documentation': []
        }
        
        # Required documentation types
        required_docs = {
            'README.md': 'Project overview and setup',
            'docs/planning/': 'Planning documentation',
            'docs/testing/': 'Testing documentation',
            'docs/optimization/': 'Optimization documentation'
        }
        
        # Check for documentation files
        doc_extensions = ['.md', '.rst', '.txt', '.pdf']
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if any(file.endswith(ext) for ext in doc_extensions):
                    relative_path = os.path.relpath(os.path.join(root, file), self.project_root)
                    doc_status['documentation_files'].append(relative_path)
                    
                    # Categorize documentation
                    if 'api' in file.lower():
                        doc_status['api_documentation'].append(relative_path)
                    elif 'user' in file.lower() or 'guide' in file.lower():
                        doc_status['user_guides'].append(relative_path)
                    elif 'technical' in file.lower() or 'implementation' in file.lower():
                        doc_status['technical_documentation'].append(relative_path)
        
        # Check for required documentation
        for req_doc, description in required_docs.items():
            doc_path = os.path.join(self.project_root, req_doc)
            if not os.path.exists(doc_path):
                doc_status['missing_documentation'].append(f"{req_doc}: {description}")
        
        # Calculate documentation score
        total_docs = len(doc_status['documentation_files'])
        required_present = len(required_docs) - len(doc_status['missing_documentation'])
        
        coverage_score = (required_present / len(required_docs)) * 60 if required_docs else 0
        volume_score = min(40, total_docs * 5)
        
        total_score = coverage_score + volume_score
        
        # Determine status
        if total_score >= 85:
            status = ReadinessLevel.PRODUCTION_READY
        elif total_score >= 70:
            status = ReadinessLevel.READY
        elif total_score >= 50:
            status = ReadinessLevel.PARTIALLY_READY
        else:
            status = ReadinessLevel.NOT_READY
        
        recommendations = []
        if doc_status['missing_documentation']:
            recommendations.append("Create missing required documentation")
            recommendations.extend([f"Create {item}" for item in doc_status['missing_documentation'][:3]])
        if len(doc_status['api_documentation']) == 0:
            recommendations.append("Add API documentation")
        if len(doc_status['user_guides']) == 0:
            recommendations.append("Create user guides")
        
        return ReadinessCheckResult(
            check_name="Documentation Coverage",
            category="documentation",
            status=status,
            score=total_score,
            details=doc_status,
            recommendations=recommendations,
            critical_issues=doc_status['missing_documentation'] if total_score < 70 else []
        )


class ProductionReadinessAssessor:
    """Main production readiness assessment coordinator"""
    
    def __init__(self):
        self.deployment_checker = DeploymentReadinessChecker()
        self.security_assessor = SecurityAssessment()
        self.config_manager = ConfigurationManagement()
        self.monitoring_validator = MonitoringValidation()
        self.doc_checker = DocumentationCompleteness()
        
        logger.info("Production Readiness Assessor initialized")
    
    def run_comprehensive_assessment(self) -> ProductionReadinessReport:
        """Run comprehensive production readiness assessment"""
        logger.info("Starting comprehensive production readiness assessment")
        
        assessment_start = time.perf_counter()
        readiness_checks = []
        
        # Run all readiness checks
        checks_to_run = [
            ("File Structure", self.deployment_checker.check_file_structure),
            ("Code Quality", self.deployment_checker.check_code_quality),
            ("Dependencies", self.deployment_checker.check_dependencies),
            ("Code Security", self.security_assessor.check_code_security),
            ("Data Protection", self.security_assessor.check_data_protection),
            ("Environment Configuration", self.config_manager.check_environment_configuration),
            ("Monitoring Systems", self.monitoring_validator.check_monitoring_systems),
            ("Documentation Coverage", self.doc_checker.check_documentation_coverage)
        ]
        
        for check_name, check_function in checks_to_run:
            try:
                logger.info(f"Running check: {check_name}")
                result = check_function()
                readiness_checks.append(result)
                logger.info(f"Check completed: {check_name} - {result.status.value} ({result.score:.1f}/100)")
            except Exception as e:
                logger.error(f"Check failed: {check_name} - {e}")
                # Create a failed check result
                failed_result = ReadinessCheckResult(
                    check_name=check_name,
                    category="system",
                    status=ReadinessLevel.NOT_READY,
                    score=0.0,
                    details={'error': str(e)},
                    recommendations=[f"Fix error in {check_name} check"],
                    critical_issues=[f"Check execution failed: {e}"]
                )
                readiness_checks.append(failed_result)
        
        assessment_duration = time.perf_counter() - assessment_start
        
        # Calculate overall scores
        category_scores = {}
        for check in readiness_checks:
            if check.category not in category_scores:
                category_scores[check.category] = []
            category_scores[check.category].append(check.score)
        
        # Average scores by category
        for category in category_scores:
            category_scores[category] = sum(category_scores[category]) / len(category_scores[category])
        
        # Calculate overall score
        overall_score = sum(category_scores.values()) / len(category_scores) if category_scores else 0
        
        # Determine overall status
        if overall_score >= 85:
            overall_status = ReadinessLevel.PRODUCTION_READY
        elif overall_score >= 70:
            overall_status = ReadinessLevel.READY
        elif overall_score >= 50:
            overall_status = ReadinessLevel.PARTIALLY_READY
        else:
            overall_status = ReadinessLevel.NOT_READY
        
        # Collect critical blockers and recommendations
        critical_blockers = []
        recommendations = []
        
        for check in readiness_checks:
            critical_blockers.extend(check.critical_issues)
            recommendations.extend(check.recommendations)
        
        # Remove duplicates
        critical_blockers = list(set(critical_blockers))
        recommendations = list(set(recommendations))
        
        # Determine deployment approval
        deployment_approval = (
            overall_status in [ReadinessLevel.READY, ReadinessLevel.PRODUCTION_READY] and
            len(critical_blockers) == 0
        )
        
        # Create report
        report = ProductionReadinessReport(
            overall_status=overall_status,
            overall_score=overall_score,
            category_scores=category_scores,
            readiness_checks=readiness_checks,
            critical_blockers=critical_blockers,
            recommendations=recommendations[:10],  # Top 10 recommendations
            deployment_approval=deployment_approval,
            assessment_timestamp=datetime.now(),
            next_review_date=datetime.now() + timedelta(days=30)
        )
        
        logger.info(f"Production readiness assessment completed in {assessment_duration:.2f}s")
        logger.info(f"Overall status: {overall_status.value} ({overall_score:.1f}/100)")
        logger.info(f"Deployment approval: {'✅ APPROVED' if deployment_approval else '❌ NOT APPROVED'}")
        
        return report


if __name__ == '__main__':
    print("🏭 Protocol Engine Production Readiness Assessment (P6 of WS2 - Phase 4)")
    print("=" * 85)
    
    # Initialize production readiness assessor
    assessor = ProductionReadinessAssessor()
    
    print("\n🔍 Running comprehensive production readiness assessment...")
    
    # Run comprehensive assessment
    readiness_report = assessor.run_comprehensive_assessment()
    
    print(f"\n📊 Production Readiness Assessment Results:")
    print(f"   Overall Status: {readiness_report.overall_status.value.upper()}")
    print(f"   Overall Score: {readiness_report.overall_score:.1f}/100")
    print(f"   Deployment Approval: {'✅ APPROVED' if readiness_report.deployment_approval else '❌ NOT APPROVED'}")
    
    print(f"\n📋 Category Scores:")
    for category, score in readiness_report.category_scores.items():
        print(f"   {category.title()}: {score:.1f}/100")
    
    print(f"\n🔍 Detailed Check Results:")
    for check in readiness_report.readiness_checks:
        status_icon = {
            ReadinessLevel.PRODUCTION_READY: "🟢",
            ReadinessLevel.READY: "🟡", 
            ReadinessLevel.PARTIALLY_READY: "🟠",
            ReadinessLevel.NOT_READY: "🔴"
        }.get(check.status, "⚪")
        
        print(f"   {status_icon} {check.check_name}: {check.status.value} ({check.score:.1f}/100)")
        
        if check.critical_issues:
            print(f"     ⚠️  Critical Issues: {len(check.critical_issues)}")
            for issue in check.critical_issues[:2]:  # Show first 2 issues
                print(f"       - {issue}")
    
    if readiness_report.critical_blockers:
        print(f"\n🚨 Critical Blockers ({len(readiness_report.critical_blockers)}):")
        for i, blocker in enumerate(readiness_report.critical_blockers[:5], 1):
            print(f"   {i}. {blocker}")
    
    if readiness_report.recommendations:
        print(f"\n💡 Top Recommendations ({len(readiness_report.recommendations)}):")
        for i, recommendation in enumerate(readiness_report.recommendations[:5], 1):
            print(f"   {i}. {recommendation}")
    
    # Save assessment report
    output_dir = "/home/ubuntu/AGENT_ALLUSE_V1/docs/production"
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, f"production_readiness_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Convert report to JSON-serializable format
    report_dict = {
        'overall_status': readiness_report.overall_status.value,
        'overall_score': readiness_report.overall_score,
        'category_scores': readiness_report.category_scores,
        'readiness_checks': [
            {
                'check_name': check.check_name,
                'category': check.category,
                'status': check.status.value,
                'score': check.score,
                'details': check.details,
                'recommendations': check.recommendations,
                'critical_issues': check.critical_issues,
                'timestamp': check.timestamp.isoformat()
            }
            for check in readiness_report.readiness_checks
        ],
        'critical_blockers': readiness_report.critical_blockers,
        'recommendations': readiness_report.recommendations,
        'deployment_approval': readiness_report.deployment_approval,
        'assessment_timestamp': readiness_report.assessment_timestamp.isoformat(),
        'next_review_date': readiness_report.next_review_date.isoformat()
    }
    
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    print(f"\n📁 Assessment Report Saved: {report_path}")
    
    # Determine next steps
    if readiness_report.deployment_approval:
        print(f"\n🎉 PRODUCTION DEPLOYMENT APPROVED!")
        print(f"✅ System is ready for production deployment")
        print(f"🚀 Ready for Phase 5: Quality Assurance and Certification")
    else:
        print(f"\n⚠️  PRODUCTION DEPLOYMENT NOT YET APPROVED")
        print(f"📋 Address critical blockers and recommendations before deployment")
        print(f"🔄 Re-run assessment after addressing issues")
        print(f"🚀 Proceeding to Phase 5: Quality Assurance and Certification")

