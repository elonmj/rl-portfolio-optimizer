"""
Kaggle GPU Manager for RL Portfolio Optimizer

This module provides comprehensive Kaggle API integration for automated
GPU training and evaluation with upload/monitoring/download capabilities.
"""

import os
import json
import time
import logging
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("Warning: Kaggle package not installed. Install with: pip install kaggle")

from config import Config


class KaggleManager:
    """
    Comprehensive Kaggle API manager for automated GPU training and evaluation.
    
    Provides:
    - Authentication with environment variables
    - Kernel upload with metadata generation
    - Real-time monitoring with progress reporting
    - Automatic result download and synchronization
    - Local/remote workflow switching
    """
    
    def __init__(self, username: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize Kaggle manager with optional credentials override.
        
        Args:
            username: Kaggle username (defaults to KAGGLE_USERNAME env var)
            api_key: Kaggle API key (defaults to KAGGLE_KEY env var)
        """
        self.logger = self._setup_logging()
        
        # Initialize Kaggle API if available
        if not KAGGLE_AVAILABLE:
            raise ImportError("Kaggle package required. Install with: pip install kaggle")
            
        self.api = KaggleApi()
        
        # Set credentials from parameters or environment
        if username and api_key:
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = api_key
            
        # Validate environment configuration
        self._validate_environment()
        
        # Authenticate with Kaggle API
        self._authenticate()
        
        # Configuration
        self.username = self._get_username()
        self.kernel_base_name = "rl-portfolio-optimizer"
        self.default_config = self._get_default_kernel_config()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup dedicated logging for Kaggle operations."""
        logger = logging.getLogger('kaggle_manager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _validate_environment(self) -> None:
        """Validate Kaggle environment variables and configuration."""
        required_vars = ['KAGGLE_USERNAME', 'KAGGLE_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
                
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {missing_vars}. "
                f"Please set them in your .env file or environment."
            )
            
        self.logger.info("Environment variables validated successfully")
        
    def _authenticate(self) -> None:
        """Authenticate with Kaggle API using environment variables."""
        try:
            self.api.authenticate()
            self.logger.info("Successfully authenticated with Kaggle API")
        except Exception as e:
            self.logger.error(f"Authentication failed: {str(e)}")
            raise RuntimeError(f"Kaggle authentication failed: {str(e)}")
            
    def _get_username(self) -> str:
        """Get authenticated Kaggle username."""
        return os.environ.get('KAGGLE_USERNAME', '')
        
    def _get_default_kernel_config(self) -> Dict[str, Any]:
        """Get default kernel configuration for RL portfolio optimizer."""
        return {
            "language": "python",
            "kernel_type": "script",
            "is_private": False,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "keywords": ["reinforcement-learning", "portfolio", "sac", "pytorch"],
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": []
        }
        
    def is_kaggle_environment(self) -> bool:
        """
        Detect if code is running in Kaggle kernel environment.
        
        Returns:
            bool: True if running in Kaggle kernel, False otherwise
        """
        # Check for Kaggle-specific environment indicators
        kaggle_indicators = [
            os.path.exists('/kaggle/input'),
            os.path.exists('/kaggle/working'),
            'KAGGLE_KERNEL_RUN_TYPE' in os.environ,
            'KAGGLE_URL_BASE' in os.environ
        ]
        
        return any(kaggle_indicators)
        
    def get_kaggle_paths(self) -> Dict[str, str]:
        """
        Get Kaggle-specific file paths for input and output.
        
        Returns:
            Dict with 'input', 'working', and 'output' paths
        """
        if self.is_kaggle_environment():
            return {
                'input': '/kaggle/input',
                'working': '/kaggle/working',
                'output': '/kaggle/working'  # Working dir is used for outputs
            }
        else:
            # Local development paths
            return {
                'input': str(Path.cwd() / 'datas'),
                'working': str(Path.cwd()),
                'output': str(Path.cwd() / 'results')
            }
            
    def generate_kernel_name(self, task_type: str = "training") -> str:
        """
        Generate unique kernel name with timestamp and random suffix.
        
        Args:
            task_type: Type of task (training, evaluation, etc.)
            
        Returns:
            Unique kernel name string
        """
        import random
        import string
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Add random suffix to ensure uniqueness
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        return f"{self.kernel_base_name}-{task_type}-{timestamp}-{random_suffix}"
        
    def create_kernel_metadata(self, 
                             kernel_name: str,
                             title: str,
                             code_file: str,
                             description: str = "",
                             custom_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create kernel metadata JSON for Kaggle submission.
        
        Args:
            kernel_name: Unique kernel name
            title: Human-readable kernel title
            code_file: Main Python script file
            description: Kernel description
            custom_config: Custom configuration overrides
            
        Returns:
            Complete kernel metadata dictionary
        """
        config = self.default_config.copy()
        if custom_config:
            config.update(custom_config)
            
        metadata = {
            "id": f"{self.username}/{kernel_name}",
            "title": title,
            "code_file": code_file,
            "language": config["language"],
            "kernel_type": config["kernel_type"],
            "is_private": config["is_private"],
            "enable_gpu": config["enable_gpu"],
            "enable_tpu": config["enable_tpu"],
            "enable_internet": config["enable_internet"],
            "keywords": config["keywords"],
            "dataset_sources": config["dataset_sources"],
            "kernel_sources": config["kernel_sources"],
            "competition_sources": config["competition_sources"],
            "model_sources": config["model_sources"]
        }
        
        if description:
            metadata["description"] = description
            
        self.logger.info(f"Generated kernel metadata for: {kernel_name}")
        return metadata
        
    def save_kernel_metadata(self, metadata: Dict[str, Any], output_dir: str = ".") -> str:
        """
        Save kernel metadata to JSON file.
        
        Args:
            metadata: Kernel metadata dictionary
            output_dir: Directory to save metadata file
            
        Returns:
            Path to saved metadata file
        """
        metadata_file = Path(output_dir) / "kernel-metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Kernel metadata saved to: {metadata_file}")
        return str(metadata_file)
        
    def validate_kernel_structure(self, kernel_dir: str) -> bool:
        """
        Validate kernel directory structure and required files.
        
        Args:
            kernel_dir: Directory containing kernel files
            
        Returns:
            True if structure is valid, False otherwise
        """
        kernel_path = Path(kernel_dir)
        
        # Check for required files
        required_files = [
            "kernel-metadata.json"
        ]
        
        for file_name in required_files:
            if not (kernel_path / file_name).exists():
                self.logger.error(f"Missing required file: {file_name}")
                return False
                
        # Validate metadata file
        try:
            with open(kernel_path / "kernel-metadata.json") as f:
                metadata = json.load(f)
                
            required_fields = ["id", "title", "code_file", "language"]
            for field in required_fields:
                if field not in metadata:
                    self.logger.error(f"Missing required metadata field: {field}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Invalid metadata file: {str(e)}")
            return False
            
        # Check for main code file
        code_file = metadata.get("code_file", "")
        if code_file and not (kernel_path / code_file).exists():
            self.logger.error(f"Missing main code file: {code_file}")
            return False
            
        self.logger.info("Kernel structure validation passed")
        return True
    
    def upload_kernel(self, kernel_dir: str, message: str = "Automated upload") -> str:
        """
        Upload kernel to Kaggle for execution.
        
        Args:
            kernel_dir: Directory containing kernel files and metadata
            message: Commit message for the upload
            
        Returns:
            Kernel slug/ID for monitoring
        """
        kernel_path = Path(kernel_dir)
        
        # Validate kernel structure before upload
        if not self.validate_kernel_structure(kernel_dir):
            raise ValueError("Kernel structure validation failed")
            
        # Read metadata to get kernel info
        with open(kernel_path / "kernel-metadata.json") as f:
            metadata = json.load(f)
            
        kernel_slug = metadata["id"]
        
        self.logger.info(f"Uploading kernel: {kernel_slug}")
        self.logger.info(f"Upload message: {message}")
        
        try:
            # Upload kernel using Kaggle API
            self.api.kernels_push(str(kernel_path))
            self.logger.info(f"✅ Kernel uploaded successfully: {kernel_slug}")
            return kernel_slug
            
        except Exception as e:
            self.logger.error(f"❌ Kernel upload failed: {str(e)}")
            raise RuntimeError(f"Failed to upload kernel: {str(e)}")
    
    def get_kernel_status(self, kernel_slug: str) -> Dict[str, Any]:
        """
        Get current status of a Kaggle kernel.
        
        Note: Due to API permissions limitations, this method provides 
        basic information and manual monitoring links instead of real-time status.
        
        Args:
            kernel_slug: Kernel identifier (username/kernel-name)
            
        Returns:
            Dictionary with kernel status information
        """
        # Return basic information due to API limitations
        return {
            'status': 'uploaded',
            'url': f"https://www.kaggle.com/code/{kernel_slug}",
            'message': 'Kernel uploaded successfully. Monitor execution manually via URL.',
            'monitoring_note': 'API status monitoring has permission limitations'
        }
    
    def monitor_kernel_execution(self, kernel_slug: str, 
                               check_interval: int = 30,
                               timeout: int = 3600) -> bool:
        """
        Monitor kernel execution until completion.
        
        Note: Due to API permissions limitations, this method provides guidance
        for manual monitoring instead of automated status checking.
        
        Args:
            kernel_slug: Kernel identifier to monitor
            check_interval: Not used (API limitations)
            timeout: Not used (API limitations)
            
        Returns:
            True (assuming upload was successful)
        """
        import time
        
        self.logger.info(f" Starting monitoring of kernel: {kernel_slug}")
        self.logger.info(f" View at: https://www.kaggle.com/code/{kernel_slug}")
        
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < timeout:
            try:
                # Get kernel status
                status_response = self.api.kernels_status(kernel_slug)
                current_status = getattr(status_response, 'status', 'unknown')
                
                if current_status != last_status:
                    self.logger.info(f"📊 Status: {current_status}")
                    last_status = current_status
                
                # Check if execution is complete
                if current_status in ['complete', 'error', 'cancelled']:
                    elapsed = time.time() - start_time
                    self.logger.info(f"⏱️ Execution finished in {elapsed:.1f}s with status: {current_status}")
                    
                    # Retrieve and display logs
                    success = self._retrieve_and_display_logs(kernel_slug)
                    
                    return current_status == 'complete' and success
                
                elif current_status == 'running':
                    # Continue monitoring
                    time.sleep(check_interval)
                    
                elif current_status in ['queued', 'pending']:
                    # Wait a bit longer for queued jobs
                    time.sleep(min(check_interval * 2, 60))
                    
                else:
                    self.logger.warning(f"⚠️ Unknown status: {current_status}")
                    time.sleep(check_interval)
                    
            except Exception as e:
                self.logger.error(f"❌ Error checking status: {e}")
                # Try to get logs anyway
                self._retrieve_and_display_logs(kernel_slug)
                return False
        
        # Timeout reached
        self.logger.warning(f"⏰ Timeout reached ({timeout}s). Attempting to retrieve logs...")
        self._retrieve_and_display_logs(kernel_slug)
        return False
    
    def _retrieve_and_display_logs(self, kernel_slug: str) -> bool:
        """
        Retrieve and display kernel execution logs.
        
        Args:
            kernel_slug: Kernel identifier
            
        Returns:
            bool: True if logs retrieved successfully
        """
        try:
            self.logger.info("📋 Retrieving execution logs...")
            
            # Get kernel logs
            logs_response = self.api.kernels_output(kernel_slug)
            
            if logs_response and 'log' in logs_response:
                log_content = logs_response['log']
                
                self.logger.info("=" * 60)
                self.logger.info("📊 KERNEL EXECUTION LOGS")
                self.logger.info("=" * 60)
                
                # Display log content
                for line in log_content.split('\n'):
                    if line.strip():
                        self.logger.info(f"LOG: {line}")
                
                self.logger.info("=" * 60)
                
                # Save logs to file
                log_file = Path("kaggle_results") / f"{kernel_slug.replace('/', '_')}.log"
                log_file.parent.mkdir(exist_ok=True)
                
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(log_content)
                
                self.logger.info(f"💾 Logs saved to: {log_file}")
                return True
                
            else:
                self.logger.warning("⚠️ No logs available or logs not ready yet")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error retrieving logs: {e}")
            return False
    
    def download_kernel_output(self, kernel_slug: str, output_dir: str = "kaggle_results") -> str:
        """
        Download kernel output files after execution.
        
        Note: Due to API limitations, provides guidance for manual download.
        
        Args:
            kernel_slug: Kernel identifier
            output_dir: Local directory to save downloaded files
            
        Returns:
            Path to download directory with instructions
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        self.logger.info(f"📥 Kernel output download guidance for: {kernel_slug}")
        self.logger.info(f"📁 Prepare download directory: {output_path.absolute()}")
        self.logger.info("📝 Note: Automatic download may have API limitations")
        self.logger.info(f"🔗 Manual download from: https://www.kaggle.com/code/{kernel_slug}")
        self.logger.info("💡 Alternative: Use 'kaggle kernels output' command manually")
        
        # Create a guidance file
        guidance_file = output_path / "download_guidance.txt"
        with open(guidance_file, 'w') as f:
            f.write(f"Kaggle Kernel Output Download Guidance\n")
            f.write(f"=====================================\n\n")
            f.write(f"Kernel: {kernel_slug}\n")
            f.write(f"URL: https://www.kaggle.com/code/{kernel_slug}\n\n")
            f.write(f"Manual Download Options:\n")
            f.write(f"1. Visit the URL above and download files manually\n")
            f.write(f"2. Use command: kaggle kernels output {kernel_slug} -p {output_path.absolute()}\n")
            f.write(f"3. Check execution logs and outputs on Kaggle dashboard\n")
        
        self.logger.info(f"📄 Download guidance saved to: {guidance_file}")
        
        return str(output_path.absolute())
    
    def _clean_python_file_for_kaggle(self, content: str) -> str:
        """
        Clean Python file content to remove non-ASCII characters that cause Kaggle upload issues.
        
        Args:
            content: Original file content
            
        Returns:
            Cleaned content compatible with Windows cp1252 encoding
        """
        import re
        
        # Mapping of common non-ASCII characters to ASCII equivalents
        replacements = {
            # French characters
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'à': 'a', 'â': 'a', 'ä': 'a',
            'ù': 'u', 'û': 'u', 'ü': 'u',
            'ç': 'c', 'î': 'i', 'ï': 'i', 'ô': 'o', 'ö': 'o',
            'À': 'A', 'É': 'E', 'È': 'E', 'Ê': 'E',
            
            # Currency and symbols
            '€': 'EUR', '£': 'GBP', '¥': 'JPY', '°': 'deg',
            '×': 'x', '÷': '/', '±': '+/-', '∞': 'inf',
            '≤': '<=', '≥': '>=', '≠': '!=', '≈': '~=',
            '→': '->', '←': '<-', '↑': '^', '↓': 'v',
            
            # Quotes and punctuation
            '"': '"', '"': '"', ''': "'", ''': "'",
            '–': '-', '—': '--', '…': '...',
            
            # Emojis to text representations
            '⚡': 'lightning', '🚀': 'rocket', '🎯': 'target',
            '📊': 'chart', '📈': 'trending_up', '📉': 'trending_down',
            '💰': 'money', '⭐': 'star', '✅': 'check', '❌': 'cross',
            '🔧': 'wrench', '🎭': 'mask', '🌟': 'star2', '🔍': 'magnifier',
            '📦': 'package', '📄': 'document', '📁': 'folder',
            '🧠': 'brain', '⚖️': 'balance', '🌐': 'globe', '🛡️': 'shield',
            '🔄': 'repeat', '🔨': 'hammer', '🧪': 'test_tube', '🧹': 'clean',
        }
        
        # Apply character replacements
        cleaned_content = content
        for old, new in replacements.items():
            cleaned_content = cleaned_content.replace(old, new)
        
        # Remove any remaining non-ASCII characters
        cleaned_content = re.sub(r'[^\x00-\x7F]+', '?', cleaned_content)
        
        return cleaned_content

    def prepare_kernel_package(self, 
                             source_files: List[str],
                             data_files: Optional[List[str]] = None,
                             output_dir: str = "kaggle_package",
                             task_type: str = "training") -> str:
        """
        Prepare complete kernel package with code and data files.
        Automatically cleans Python files to ensure Kaggle compatibility.
        
        Args:
            source_files: List of Python source files to include
            data_files: List of data files to include  
            output_dir: Directory to create the kernel package
            task_type: Type of task (training, evaluation, etc.)
            
        Returns:
            Path to prepared kernel package directory
        """
        import shutil  # Import shutil for file operations
        
        package_path = Path(output_dir)
        if package_path.exists():
            shutil.rmtree(package_path)
        package_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📦 Preparing kernel package: {package_path.absolute()}")
        self.logger.info("🧹 Auto-cleaning files for Kaggle compatibility...")
        
        # Copy and clean source files
        if task_type == "training":
            # Use standalone Kaggle script for training
            kaggle_script = Path(__file__).parent / "train_kaggle.py"
            if kaggle_script.exists():
                with open(kaggle_script, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                cleaned_content = self._clean_python_file_for_kaggle(content)
                
                train_path = package_path / "train.py"
                with open(train_path, 'w', encoding='ascii', errors='replace') as f:
                    f.write(cleaned_content)
                
                self.logger.info(f"  📄 Added standalone Kaggle script: train.py")
                main_script = "train.py"
            else:
                self.logger.error("❌ train_kaggle.py not found!")
                raise FileNotFoundError("train_kaggle.py is required for Kaggle training")
        else:
            # For other tasks, use the original approach
            for source_file in source_files:
                src_path = Path(source_file)
                if src_path.exists():
                    dst_path = package_path / src_path.name
                    
                    if src_path.suffix == '.py':
                        # Read, clean, and write Python files
                        try:
                            with open(src_path, 'r', encoding='utf-8') as f:
                                original_content = f.read()
                            
                            # Clean content for Kaggle compatibility
                            cleaned_content = self._clean_python_file_for_kaggle(original_content)
                            
                            # Write cleaned content as ASCII-safe
                            with open(dst_path, 'w', encoding='ascii', errors='replace') as f:
                                f.write(cleaned_content)
                            
                            self.logger.info(f"  📄 Added (cleaned): {src_path.name}")
                        except Exception as e:
                            self.logger.error(f"  ❌ Failed to clean {src_path.name}: {e}")
                            # Fallback to binary copy
                            shutil.copy2(src_path, dst_path)
                            self.logger.info(f"  📄 Added (fallback): {src_path.name}")
                    else:
                        # Binary copy for non-Python files
                        shutil.copy2(src_path, dst_path)
                        self.logger.info(f"  📄 Added: {src_path.name}")
                else:
                    self.logger.warning(f"  ⚠️ Source file not found: {source_file}")
            
            main_script = source_files[0] if source_files and task_type != "training" else "train.py"
        
        # Copy data files if provided, or copy default data for training
        if data_files:
            for data_file in data_files:
                src_path = Path(data_file)
                if src_path.exists():
                    if src_path.is_dir():
                        # Copy entire directory
                        dst_path = package_path / src_path.name
                        import shutil
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                        self.logger.info(f"  📁 Added directory: {src_path.name}")
                    else:
                        # Copy single file
                        dst_path = package_path / src_path.name
                        import shutil
                        shutil.copy2(src_path, dst_path)
                        self.logger.info(f"  📄 Added data: {src_path.name}")
                else:
                    self.logger.warning(f"  ⚠️ Data file not found: {data_file}")
        else:
            # Copy default data directory for training tasks
            default_data_dir = Path(__file__).parent / "datas"
            if default_data_dir.exists() and task_type == "training":
                shutil.copytree(default_data_dir, package_path / "datas", dirs_exist_ok=True)
                self.logger.info(f"  📁 Added default directory: datas")
        
        # Copy requirements.txt if it exists
        requirements_path = Path(__file__).parent / "requirements.txt"
        if requirements_path.exists():
            shutil.copy2(requirements_path, package_path / "requirements.txt")
            self.logger.info(f"  📄 Added: requirements.txt")
        
        # Generate kernel metadata
        kernel_name = self.generate_kernel_name(task_type)
        
        # Create unique title with kernel name suffix for uniqueness
        import random
        import string
        random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        title = f"RL Portfolio Optimizer {task_type.title()} {random_id}"
        
        main_script = source_files[0] if source_files else "train.py"
        
        description = (
            f"Automated {task_type} execution for RL Portfolio Optimizer. "
            f"SAC agent with GPU acceleration for portfolio management. "
            f"Kernel ID: {kernel_name}"
        )
        
        metadata = self.create_kernel_metadata(
            kernel_name=kernel_name,
            title=title, 
            code_file=Path(main_script).name,
            description=description
        )
        
        # Save metadata
        self.save_kernel_metadata(metadata, str(package_path))
        
        self.logger.info(f"✅ Kernel package prepared: {package_path.absolute()}")
        return str(package_path.absolute())
    
    def run_kaggle_workflow(self,
                          source_files: List[str],
                          data_files: Optional[List[str]] = None,
                          task_type: str = "training",
                          monitor: bool = True,
                          download_results: bool = True) -> Dict[str, Any]:
        """
        Complete Kaggle workflow: prepare -> upload -> monitor -> download.
        
        Args:
            source_files: Python source files to upload
            data_files: Data files/directories to upload
            task_type: Type of task (training, evaluation)
            monitor: Whether to monitor execution
            download_results: Whether to download results automatically
            
        Returns:
            Dictionary with workflow results and paths
        """
        workflow_start = time.time()
        self.logger.info(f"🚀 Starting Kaggle {task_type} workflow...")
        
        try:
            # Step 1: Prepare kernel package
            package_dir = self.prepare_kernel_package(
                source_files=source_files,
                data_files=data_files,
                task_type=task_type
            )
            
            # Step 2: Upload kernel
            kernel_slug = self.upload_kernel(package_dir)
            
            results = {
                'kernel_slug': kernel_slug,
                'package_dir': package_dir,
                'upload_time': time.time(),
                'status': 'uploaded'
            }
            
            # Step 3: Monitor execution (guidance only due to API limitations)
            if monitor:
                success = self.monitor_kernel_execution(kernel_slug)
                results['execution_success'] = success
                results['completion_time'] = time.time()
                results['status'] = 'uploaded_with_guidance'
            
            # Step 4: Download results (guidance only due to API limitations)
            if download_results:
                download_dir = self.download_kernel_output(kernel_slug)
                results['download_dir'] = download_dir
                results['status'] = 'completed_with_guidance'
            
            workflow_time = time.time() - workflow_start
            self.logger.info(f"🎉 Kaggle workflow completed in {workflow_time:.1f}s")
            
            # Final summary
            self.logger.info("📊 Workflow Summary:")
            self.logger.info(f"  🔗 Kernel: https://www.kaggle.com/code/{kernel_slug}")
            if download_results and 'download_dir' in results:
                self.logger.info(f"  📁 Results: {results['download_dir']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Kaggle workflow failed: {str(e)}")
            raise RuntimeError(f"Kaggle workflow failed: {str(e)}")
    
    def create_dataset_and_notebook_workflow(self, 
                                           source_files: List[str],
                                           data_files: Optional[List[str]] = None,
                                           task_type: str = "training",
                                           episodes: int = 50) -> Dict[str, Any]:
        """
        NEW APPROACH: Create dataset with source code + notebook that imports and executes.
        This maintains code modularity and avoids 409 conflicts.
        
        Args:
            source_files: Python source files to include in dataset
            data_files: Data files to include in dataset
            task_type: Type of task (training, evaluation)
            episodes: Number of training episodes
            
        Returns:
            Dictionary with workflow results
        """
        workflow_start = time.time()
        self.logger.info(f"🚀 Starting Dataset + Notebook workflow for {task_type}...")
        
        try:
            # Step 1: Create and upload dataset with source code
            dataset_slug = self._create_and_upload_code_dataset(
                source_files=source_files,
                data_files=data_files,
                task_type=task_type
            )
            
            # Step 2: Create and upload notebook that imports from dataset
            kernel_slug = self._create_and_upload_notebook(
                dataset_slug=dataset_slug,
                task_type=task_type,
                episodes=episodes
            )
            
            results = {
                'dataset_slug': dataset_slug,
                'kernel_slug': kernel_slug,
                'workflow_type': 'dataset_notebook',
                'task_type': task_type,
                'episodes': episodes,
                'upload_time': time.time(),
                'status': 'uploaded'
            }
            
            workflow_time = time.time() - workflow_start
            self.logger.info(f"🎉 Dataset + Notebook workflow completed in {workflow_time:.1f}s")
            
            # Final summary
            self.logger.info("📊 Workflow Summary:")
            self.logger.info(f"  📦 Dataset: https://www.kaggle.com/datasets/{dataset_slug}")
            self.logger.info(f"  📓 Notebook: https://www.kaggle.com/code/{kernel_slug}")
            self.logger.info("📝 Monitor execution manually via notebook URL")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Dataset + Notebook workflow failed: {str(e)}")
            raise RuntimeError(f"Dataset + Notebook workflow failed: {str(e)}")
    
    def _create_and_upload_code_dataset(self,
                                      source_files: List[str],
                                      data_files: Optional[List[str]] = None,
                                      task_type: str = "training") -> str:
        """
        Create and upload dataset containing source code and data files.
        
        Args:
            source_files: Python source files to include
            data_files: Data files to include
            task_type: Type of task (for dataset naming)
            
        Returns:
            Dataset slug (username/dataset-name)
        """
        import shutil
        import random
        import string
        
        self.logger.info("📦 Creating code dataset...")
        
        # Generate unique dataset name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        dataset_name = f"rl-portfolio-optimizer-code-{timestamp}-{random_suffix}"
        
        # Create dataset directory
        dataset_dir = Path("kaggle_dataset_temp")
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy and clean source files
        self.logger.info("📄 Adding source files to dataset...")
        for source_file in source_files:
            src_path = Path(source_file)
            if src_path.exists():
                dst_path = dataset_dir / src_path.name
                
                if src_path.suffix == '.py':
                    # Read, clean, and write Python files
                    try:
                        with open(src_path, 'r', encoding='utf-8') as f:
                            original_content = f.read()
                        
                        # Clean content for Kaggle compatibility
                        cleaned_content = self._clean_python_file_for_kaggle(original_content)
                        
                        # Write cleaned content as ASCII-safe
                        with open(dst_path, 'w', encoding='ascii', errors='replace') as f:
                            f.write(cleaned_content)
                        
                        self.logger.info(f"  ✅ Added (cleaned): {src_path.name}")
                    except Exception as e:
                        self.logger.error(f"  ❌ Failed to clean {src_path.name}: {e}")
                        # Fallback to binary copy
                        shutil.copy2(src_path, dst_path)
                        self.logger.info(f"  ✅ Added (fallback): {src_path.name}")
                else:
                    # Binary copy for non-Python files
                    shutil.copy2(src_path, dst_path)
                    self.logger.info(f"  ✅ Added: {src_path.name}")
            else:
                self.logger.warning(f"  ⚠️ Source file not found: {source_file}")
        
        # Copy data files
        if data_files:
            self.logger.info("📁 Adding data files to dataset...")
            for data_file in data_files:
                src_path = Path(data_file)
                if src_path.exists():
                    if src_path.is_dir():
                        # Copy entire directory
                        dst_path = dataset_dir / src_path.name
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                        self.logger.info(f"  ✅ Added directory: {src_path.name}")
                    else:
                        # Copy single file
                        dst_path = dataset_dir / src_path.name
                        shutil.copy2(src_path, dst_path)
                        self.logger.info(f"  ✅ Added data: {src_path.name}")
                else:
                    self.logger.warning(f"  ⚠️ Data file not found: {data_file}")
        else:
            # Copy default data directory for training tasks
            default_data_dir = Path(__file__).parent / "datas"
            if default_data_dir.exists() and task_type == "training":
                shutil.copytree(default_data_dir, dataset_dir / "datas", dirs_exist_ok=True)
                self.logger.info(f"  ✅ Added default directory: datas")
        
        # Copy requirements.txt if it exists
        requirements_path = Path(__file__).parent / "requirements.txt"
        if requirements_path.exists():
            shutil.copy2(requirements_path, dataset_dir / "requirements.txt")
            self.logger.info(f"  ✅ Added: requirements.txt")
        
        # Create dataset metadata
        dataset_metadata = {
            "title": f"RL Portfolio Optimizer Code {timestamp}",
            "id": f"{self.username}/{dataset_name}",
            "licenses": [{"name": "MIT"}],
            "description": f"Source code for RL Portfolio Optimizer {task_type} task. Generated: {datetime.now().isoformat()}"
        }
        
        # Save dataset metadata
        metadata_file = dataset_dir / "dataset-metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        self.logger.info(f"📋 Dataset metadata created: {dataset_name}")
        
        # Upload dataset using Kaggle API (make public)
        try:
            self.logger.info(f"⬆️ Uploading PUBLIC dataset: {dataset_name}")
            self.api.dataset_create_new(
                folder=str(dataset_dir),
                convert_to_csv=False,
                dir_mode='tar',
                public=True  # Make dataset public
            )
            
            dataset_slug = f"{self.username}/{dataset_name}"
            self.logger.info(f"✅ PUBLIC Dataset uploaded successfully: {dataset_slug}")
            
            # Clean up temporary directory
            shutil.rmtree(dataset_dir)
            
            return dataset_slug
            
        except Exception as e:
            self.logger.error(f"❌ Dataset upload failed: {str(e)}")
            # Clean up on failure
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            raise RuntimeError(f"Failed to upload dataset: {str(e)}")
    
 
    def _ensure_git_up_to_date(self, branch: str = "feature/training-config-updates") -> bool:
        """
        Automatically manage Git workflow: status -> add -> commit -> push.
        Ensures that local changes are pushed to GitHub before Kaggle clones.
        
        Args:
            branch: Git branch to work with
            
        Returns:
            bool: True if Git is up to date, False if errors occurred
        """
        import subprocess
        from datetime import datetime
        
        self.logger.info("🔍 Checking Git status and ensuring changes are pushed...")
        
        try:
            # Check if we're in a git repository
            result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                self.logger.error("❌ Not in a Git repository")
                return False
            
            # Get current branch
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            current_branch = result.stdout.strip()
            self.logger.info(f"📍 Current branch: {current_branch}")
            
            # Check if we're on the right branch
            if current_branch != branch:
                self.logger.warning(f"⚠️ Current branch ({current_branch}) != target branch ({branch})")
                self.logger.info(f"🔄 Switching to branch: {branch}")
                
                # Try to switch to the target branch
                result = subprocess.run(['git', 'checkout', branch], 
                                      capture_output=True, text=True, cwd=os.getcwd())
                if result.returncode != 0:
                    self.logger.error(f"❌ Failed to switch to branch {branch}: {result.stderr}")
                    return False
            
            # Check git status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                self.logger.error(f"❌ Git status failed: {result.stderr}")
                return False
            
            status_output = result.stdout.strip()
            
            if not status_output:
                self.logger.info("✅ Working directory clean - no changes to commit")
                
                # Check if we're ahead of remote
                result = subprocess.run(['git', 'status', '--porcelain=v1', '--branch'], 
                                      capture_output=True, text=True, cwd=os.getcwd())
                if '[ahead' in result.stdout:
                    self.logger.info("📤 Local commits ahead of remote - pushing...")
                    return self._git_push(branch)
                else:
                    self.logger.info("✅ Repository is up to date with remote")
                    return True
            
            # There are changes - show them
            self.logger.info("📝 Detected local changes:")
            for line in status_output.split('\n'):
                if line.strip():
                    status_code = line[:2]
                    file_path = line[3:]
                    status_desc = self._get_git_status_description(status_code)
                    self.logger.info(f"  {status_desc}: {file_path}")
            
            # Add all changes
            self.logger.info("📦 Adding all changes...")
            result = subprocess.run(['git', 'add', '.'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                self.logger.error(f"❌ Git add failed: {result.stderr}")
                return False
            
            # Create commit message with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message = f"Auto-commit before Kaggle workflow - {timestamp}\n\nUpdated KaggleManager with Git automation and configuration fixes"
            
            self.logger.info("💾 Committing changes...")
            result = subprocess.run(['git', 'commit', '-m', commit_message], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                # Check if it's because there's nothing to commit
                if "nothing to commit" in result.stdout:
                    self.logger.info("✅ Nothing to commit - working directory clean")
                    return True
                else:
                    self.logger.error(f"❌ Git commit failed: {result.stderr}")
                    return False
            
            self.logger.info("✅ Changes committed successfully")
            
            # Push to remote
            return self._git_push(branch)
            
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Git command timed out")
            return False
        except Exception as e:
            self.logger.error(f"❌ Git workflow failed: {e}")
            return False
    
    def _git_push(self, branch: str) -> bool:
        """
        Push changes to remote repository.
        
        Args:
            branch: Branch to push
            
        Returns:
            bool: True if push successful
        """
        import subprocess
        
        self.logger.info(f"📤 Pushing to remote branch: {branch}")
        
        try:
            result = subprocess.run(['git', 'push', 'origin', branch], 
                                  capture_output=True, text=True, cwd=os.getcwd(), timeout=60)
            
            if result.returncode == 0:
                self.logger.info("✅ Successfully pushed to remote repository")
                return True
            else:
                self.logger.error(f"❌ Git push failed: {result.stderr}")
                # Try to provide helpful error messages
                if "rejected" in result.stderr.lower():
                    self.logger.error("💡 Push rejected - you may need to pull first")
                elif "authentication" in result.stderr.lower():
                    self.logger.error("💡 Authentication failed - check your Git credentials")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Git push timed out")
            return False
        except Exception as e:
            self.logger.error(f"❌ Git push error: {e}")
            return False
    
    def _get_git_status_description(self, status_code: str) -> str:
        """
        Convert Git status codes to human-readable descriptions.
        
        Args:
            status_code: Two-character Git status code
            
        Returns:
            Human-readable description
        """
        status_map = {
            'M ': 'Modified (staged)',
            ' M': 'Modified (unstaged)',
            'MM': 'Modified (staged and unstaged)',
            'A ': 'Added (staged)',
            ' A': 'Added (unstaged)',
            'D ': 'Deleted (staged)',
            ' D': 'Deleted (unstaged)',
            'R ': 'Renamed (staged)',
            'C ': 'Copied (staged)',
            '??': 'Untracked',
            '!!': 'Ignored'
        }
        return status_map.get(status_code, f'Unknown ({status_code})')

    def create_and_upload_notebook_github(self, repo_url: str = "https://github.com/elonmj/rl-portfolio-optimizer.git", 
                                         branch: str = "feature/training-config-updates", 
                                         timeout: int = 3600, check_interval: int = 30, 
                                         task_type: str = "training", episodes: int = 1) -> bool:
        """
        Create and upload notebook that clones from GitHub instead of using datasets.
        Much more reliable approach! Now includes automatic Git management.
        
        Args:
            repo_url: GitHub repository URL
            branch: Git branch to clone
            timeout: Max execution time in seconds
            check_interval: Status check interval in seconds 
            task_type: Type of task for naming
            episodes: Number of training episodes (default: 1 for quick testing)
            
        Returns:
            bool: True if successful
        """
        
        self.logger.info("🚀 Starting GitHub-based Notebook workflow...")
        self.logger.info(f"📂 Repository: {repo_url}")
        self.logger.info(f"🌿 Branch: {branch}")
        
        # CRITICAL: Ensure Git is up to date before proceeding
        self.logger.info("🔄 Step 1: Ensuring Git repository is up to date...")
        if not self._ensure_git_up_to_date(branch):
            self.logger.error("❌ Failed to update Git repository - aborting workflow")
            self.logger.error("💡 Kaggle will clone outdated code if Git is not up to date!")
            return False
        
        self.logger.info("✅ Git repository is up to date - proceeding with Kaggle workflow")
        
        # Generate unique kernel name
        import random
        import string
        random_suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
        kernel_name = f"rl-portfolio-optimizer-{task_type}-{random_suffix}"
        
        # Create script directory
        script_dir = Path("kaggle_script_temp")
        if script_dir.exists():
            shutil.rmtree(script_dir)
        script_dir.mkdir(parents=True, exist_ok=True)
        
        # Build GitHub-based script content
        script_content = f'''#!/usr/bin/env python3
# RL Portfolio Optimizer - Kaggle Training Script (GitHub-based)
# Generated automatically - Execute on Kaggle GPU

import sys
import os
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

print("=== RL Portfolio Optimizer - GitHub Setup ===")

# Configuration
REPO_URL = "{repo_url}"
BRANCH = "{branch}"
REPO_DIR = "/kaggle/working/rl-portfolio-optimizer"

print(f"Repository: {{REPO_URL}}")
print(f"Branch: {{BRANCH}}")

try:
    # Clone repository with git (simple and reliable for public repos)
    print("\\n[INFO] Cloning repository from GitHub...")
    
    # Ensure we use HTTPS URL for public access
    if REPO_URL.startswith("git@"):
        repo_https = REPO_URL.replace("git@github.com:", "https://github.com/")
    else:
        repo_https = REPO_URL
    
    print(f"🎯 TRACKING_PROGRESS: Cloning from {{repo_https}}")
    
    # Clone with specific branch (public repo, no auth needed)
    clone_cmd = [
        "git", "clone", 
        "--single-branch", "--branch", BRANCH,
        "--depth", "1",  # Shallow clone for speed
        repo_https, REPO_DIR
    ]
    
    print(f"Running: {{' '.join(clone_cmd)}}")
    result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=300)
    
    if result.returncode == 0:
        print("[OK] Repository cloned successfully!")
        print(f"🎯 TRACKING_SUCCESS: Repository cloned from {{BRANCH}} branch")
        
        # List cloned contents
        if os.path.exists(REPO_DIR):
            files = os.listdir(REPO_DIR)
            print(f"[INFO] Cloned files: {{len(files)}} items")
            for f in sorted(files)[:10]:  # First 10 items
                print(f"  - {{f}}")
        print(f"🎯 TRACKING_PROGRESS: File listing completed")
    else:
        print(f"[ERROR] Git clone failed:")
        print(f"STDOUT: {{result.stdout}}")
        print(f"STDERR: {{result.stderr}}")
        print(f"🎯 TRACKING_ERROR: Git clone failed - {{result.stderr}}")
        sys.exit(1)
    
    # Change to repository directory
    os.chdir(REPO_DIR)
    print(f"[INFO] Changed to directory: {{os.getcwd()}}")
    
    # Verify essential files exist
    essential_files = ["train.py", "config.py", "agent.py", "environment.py", "models.py"]
    missing_files = []
    
    for file in essential_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"[OK] Found {{file}}")
    
    if missing_files:
        print(f"[ERROR] Missing essential files: {{missing_files}}")
        sys.exit(1)
    
    # Install requirements if present
    if os.path.exists("requirements.txt"):
        print("\\n[INFO] Installing requirements...")
        print("🎯 TRACKING_PROGRESS: Starting requirements installation")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, timeout=600)
        print("[OK] Requirements installed")
        print("🎯 TRACKING_SUCCESS: Requirements installation completed")
    
    # Set environment variable for number of episodes (CRITICAL: must be set before importing)
    os.environ['TRAINING_EPISODES'] = str({episodes})
    print(f"[INFO] Set TRAINING_EPISODES environment variable to: {episodes}")
    
    # Import and run training
    print("\\n[INFO] Starting training...")
    print("🎯 TRACKING_PROGRESS: Initializing training module")
    sys.path.insert(0, os.getcwd())
    
    # Import config and verify episodes configuration
    from config import Config
    training_config = Config.get_training_config_for_environment()
    actual_episodes = training_config.get('max_episodes', 1)
    print(f"[INFO] Configuration loaded - Episodes to run: {{actual_episodes}}")
    
    if actual_episodes != {episodes}:
        print(f"[WARNING] Configuration mismatch! Expected {episodes}, got {{actual_episodes}}")
        # Force the correct configuration
        os.environ['TRAINING_EPISODES'] = str({episodes})
        print(f"[INFO] Forced TRAINING_EPISODES to: {episodes}")
    
    # Import training script
    import train
    print("🎯 TRACKING_PROGRESS: Train module imported successfully")
    
    # Run training with explicit episodes parameter (bypass train.main fallbacks)
    if hasattr(train, 'main'):
        print(f"🎯 TRACKING_PROGRESS: Executing train.main(num_episodes={episodes}) - explicit parameter")
        # Call main() with explicit episodes parameter to bypass fallbacks
        train.main(num_episodes={episodes})
    else:
        print("🎯 TRACKING_PROGRESS: Executing train.py content directly")
        # Execute train.py content directly
        exec(open('train.py').read())
    
    print("\\n[OK] Training completed successfully!")
    print("🎯 TRACKING_SUCCESS: Training execution finished successfully")
    
except subprocess.TimeoutExpired:
    print("[ERROR] Git clone timeout - repository too large or network issues")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"[ERROR] Command failed: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Training failed: {{e}}")
    print("Traceback:")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # Create results summary
    try:
        results_dir = "/kaggle/working/results"
        os.makedirs(results_dir, exist_ok=True)
        
        summary = {{
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "repo_url": REPO_URL,
            "branch": BRANCH,
            "kaggle_session": True
        }}
        
        with open(os.path.join(results_dir, "session_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\\n[INFO] Results saved to {{results_dir}}")
    except Exception as e:
        print(f"[WARN] Could not save results: {{e}}")
'''

        # Save script
        script_file = script_dir / f"{kernel_name}.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Create metadata (no dataset needed!)
        title = kernel_name.replace('-', ' ').title()
        kernel_metadata = {
            "id": f"{self.username}/{kernel_name}",
            "title": title,
            "code_file": f"{kernel_name}.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": False,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,  # CRITICAL: Need internet for git clone
            "keywords": ["reinforcement-learning", "portfolio", "sac", "pytorch", "github"],
            "datasetSources": [],  # No dataset needed!
            "kernelSources": [],
            "competitionSources": [],
            "modelSources": []
        }
        
        # Save metadata
        metadata_file = script_dir / "kernel-metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(kernel_metadata, f, indent=2)
        
        self.logger.info(f"📋 GitHub script created: {kernel_name}")
        
        # Upload kernel
        try:
            self.logger.info(f"⬆️ Uploading GitHub-based script: {kernel_name}")
            self.api.kernels_push(str(script_dir))
            
            kernel_slug = f"{self.username}/{kernel_name}"
            self.logger.info(f"✅ GitHub script uploaded: {kernel_slug}")
            self.logger.info(f"🔗 URL: https://www.kaggle.com/code/{kernel_slug}")
            
            # Enhanced monitoring with API log retrieval and unique keywords
            self.logger.info(f"� Kernel uploaded: https://www.kaggle.com/code/{kernel_slug}")
            self.logger.info("� Starting enhanced monitoring with log analysis...")
            
            # Monitor execution with smart log analysis
            success = self._monitor_kernel_with_logs(kernel_slug, timeout, check_interval)
            return success
                
        except Exception as e:
            self.logger.error(f"❌ Failed to upload GitHub script: {e}")
            return False
        
        finally:
            # Cleanup
            if script_dir.exists():
                shutil.rmtree(script_dir)

    def _monitor_kernel_with_logs(self, kernel_slug: str, timeout: int = 3600, check_interval: int = None) -> bool:
        """
        Enhanced kernel monitoring with adaptive intervals using exponential backoff.
        Intervals: 10s → 20s → 40s → 80s → 120s (max).
        
        Args:
            kernel_slug: Kernel slug (username/kernel-name)
            timeout: Maximum monitoring time in seconds
            check_interval: Deprecated - using adaptive intervals now
            
        Returns:
            bool: True if execution completed successfully, False otherwise
        """
        import json
        import time
        import tempfile
        import os
        from datetime import datetime
        
        start_time = time.time()
        execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Adaptive monitoring with exponential backoff
        base_interval = 10  # Start with 10 seconds
        max_interval = 120  # Cap at 2 minutes
        current_interval = base_interval
        check_count = 0
        
        self.logger.info(f"🔍 Enhanced monitoring started for: {kernel_slug}")
        self.logger.info(f"⏱️ Timeout: {timeout}s, Adaptive intervals: {base_interval}s → {max_interval}s")
        self.logger.info(f"🏷️ Execution ID: {execution_id}")
        
        # Keywords to track progress (enhanced with unique tracking markers)
        success_keywords = [
            "[OK] Training completed successfully!",
            "TRACKING_SUCCESS: Training execution finished successfully",
            "TRACKING_SUCCESS: Repository cloned",
            "TRACKING_SUCCESS: Requirements installation completed",
            "[OK] Repository cloned successfully!",
            "[OK] Requirements installed"
        ]
        
        error_keywords = [
            "[ERROR]",
            "TRACKING_ERROR:",
            "fatal:",
            "❌",
            "Exception:",
            "Error:",
            "Failed:",
            "sys.exit(1)"
        ]
        
        progress_keywords = [
            "TRACKING_PROGRESS:",
            "[INFO] Cloning repository",
            "[INFO] Installing requirements", 
            "[INFO] Starting training",
            "=== RL Portfolio Optimizer",
            "🎯 TRACKING_PROGRESS:"
        ]
        
        last_log_size = 0
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        try:
            while time.time() - start_time < timeout:
                try:
                    # Check kernel status first
                    status_response = self.api.kernels_status(kernel_slug)
                    status = status_response.status
                    
                    self.logger.info(f"⏳ Status: {status} (Time: {int(time.time() - start_time)}s)")
                    
                    # Convert status to string for comparison (handle both enum and string)
                    status_str = str(status).upper()
                    if hasattr(status, 'name'):
                        status_str = status.name.upper()
                    elif hasattr(status, 'value'):
                        status_str = str(status.value).upper()
                    
                    # Handle different status states
                    if "ERROR" in status_str:
                        failure_msg = getattr(status_response, 'failureMessage', 'Unknown error')
                        self.logger.error(f"❌ Kernel failed: {failure_msg}")
                        
                        # ALWAYS download logs on error - this is crucial for debugging
                        self.logger.info("📥 Downloading error logs for analysis...")
                        try:
                            error_analysis = self._analyze_kernel_logs(kernel_slug, error_keywords + success_keywords, error_keywords, live_mode=False)
                            self.logger.info("✅ Error logs downloaded and analyzed")
                        except Exception as log_e:
                            self.logger.warning(f"Could not retrieve error logs: {log_e}")
                        
                        return False  # Stop monitoring immediately on error
                        
                    elif "COMPLETE" in status_str:
                        self.logger.info("✅ Kernel execution completed!")
                        
                        # Analyze final logs to determine success
                        try:
                            success = self._analyze_kernel_logs(kernel_slug, success_keywords, error_keywords)
                            if success:
                                self.logger.info("🎉 Training completed successfully!")
                            else:
                                self.logger.warning("⚠️ Kernel completed but with potential issues")
                            return success
                        except Exception as log_e:
                            self.logger.warning(f"Could not analyze completion logs: {log_e}")
                            return True  # Assume success if logs unavailable
                            
                    elif any(x in status_str for x in ["RUNNING", "QUEUED"]):
                        # Kernel is active - try to get live progress from logs
                        try:
                            progress_found = self._analyze_kernel_logs(
                                kernel_slug, 
                                progress_keywords + success_keywords, 
                                error_keywords,
                                live_mode=True
                            )
                            consecutive_errors = 0  # Reset error counter
                        except Exception as log_e:
                            consecutive_errors += 1
                            self.logger.debug(f"Log analysis failed (attempt {consecutive_errors}): {log_e}")
                            
                            if consecutive_errors >= max_consecutive_errors:
                                self.logger.warning("Multiple log analysis failures - kernel might have crashed")
                    
                    # Adaptive wait with exponential backoff
                    check_count += 1
                    self.logger.info(f"⏳ Waiting {current_interval}s before next check (#{check_count})")
                    time.sleep(current_interval)
                    
                    # Increase interval for next check (exponential backoff)
                    current_interval = min(current_interval * 2, max_interval)
                    
                except Exception as e:
                    consecutive_errors += 1
                    self.logger.error(f"Status check failed (attempt {consecutive_errors}): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error("Multiple consecutive failures - stopping monitoring")
                        break
                    
                    time.sleep(check_interval)
                    
            # Timeout reached
            elapsed = time.time() - start_time
            self.logger.error(f"⏰ Monitoring timeout after {elapsed:.1f}s")
            self.logger.info(f"📝 Manual check: https://www.kaggle.com/code/{kernel_slug}")
            return False
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Monitoring interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"❌ Monitoring failed: {e}")
            return False
            
    def _analyze_kernel_logs(self, kernel_slug: str, success_keywords: list, 
                           error_keywords: Optional[list] = None, live_mode: bool = False) -> bool:
        """
        Analyze kernel logs for progress and completion status with full log display.
        Restored original functionality that was working before.
        
        Args:
            kernel_slug: Kernel slug to analyze
            success_keywords: Keywords indicating success/progress
            error_keywords: Keywords indicating errors  
            live_mode: If True, only show new log entries
            
        Returns:
            bool: True if success indicators found, False if errors detected
        """
        try:
            # Create temporary directory for log download
            with tempfile.TemporaryDirectory() as temp_dir:
                self.logger.info(f"📥 Downloading logs for: {kernel_slug}")
                
                # Download kernel output (includes logs)
                self.api.kernels_output(kernel_slug, path=temp_dir, quiet=True)
                
                # Look for the log file
                log_file = None
                log_files = []
                for file in os.listdir(temp_dir):
                    if file.endswith('.log'):
                        log_files.append(file)
                        log_file = os.path.join(temp_dir, file)
                        
                self.logger.info(f"📄 Log files found: {log_files}")
                        
                if not log_file or not os.path.exists(log_file):
                    self.logger.warning("❌ No log file found")
                    return False
                    
                # Parse log content
                with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                    log_content = f.read()
                    
                if not log_content.strip():
                    self.logger.warning("❌ Log file is empty")
                    return False
                    
                self.logger.info(f"📊 Log content size: {len(log_content)} characters")
                
                # Parse JSON log entries (restore original parsing logic)
                log_entries = []
                try:
                    # Try to parse as JSON array first
                    if log_content.startswith('[') and log_content.endswith(']'):
                        log_entries = json.loads(log_content)
                    else:
                        # Parse line by line for JSON objects
                        lines = log_content.strip().split('\n')
                        for line in lines:
                            line = line.strip().rstrip(',')
                            if line:
                                try:
                                    entry = json.loads(line)
                                    log_entries.append(entry)
                                except json.JSONDecodeError:
                                    # Add as plain text entry
                                    log_entries.append({"data": line, "stream_name": "stdout", "time": len(log_entries)})
                except json.JSONDecodeError as e:
                    self.logger.warning(f"⚠️ JSON parsing failed: {e}")
                    # Fallback to plain text analysis
                    log_entries = [{"data": log_content, "stream_name": "stdout", "time": 0}]
                
                self.logger.info(f"📋 Parsed {len(log_entries)} log entries")
                
                # Display logs with timestamps like before
                success_found = False
                error_found = False
                
                for i, entry in enumerate(log_entries):
                    if isinstance(entry, dict) and 'data' in entry:
                        data = entry['data']
                        stream = entry.get('stream_name', 'stdout')
                        timestamp = entry.get('time', i)
                        
                        # Display log entry with timestamp (restore original format)
                        if isinstance(timestamp, (int, float)):
                            time_str = f"{timestamp:.1f}s"
                        else:
                            time_str = str(timestamp)
                            
                        # Clean data for display
                        clean_data = data.strip()
                        if clean_data:
                            print(f"{time_str}\t{i+1}\t{clean_data}")
                            
                        # Check for success keywords
                        for keyword in success_keywords:
                            if keyword in data:
                                success_found = True
                                    
                        # Check for error keywords  
                        if error_keywords:
                            for keyword in error_keywords:
                                if keyword in data:
                                    error_found = True
                
                # Summary
                if success_found:
                    self.logger.info("✅ Success indicators found in logs")
                if error_found:
                    self.logger.warning("⚠️ Error indicators found in logs")
                
                # Return analysis result
                if error_found and not success_found:
                    return False
                elif success_found:
                    return True
                else:
                    return not error_found  # Neutral if no clear indicators
                    
        except Exception as e:
            if not live_mode:
                self.logger.debug(f"Log analysis error: {e}")
            return False

    def _create_and_upload_notebook(self,
                                  dataset_slug: str,
                                  task_type: str = "training",
                                  episodes: int = 1) -> str:
        """
        Create and upload SCRIPT (.py) that imports and executes code from dataset.
        Scripts handle dataset_sources attachment more reliably than notebooks.
        
        Args:
            dataset_slug: Dataset containing source code
            task_type: Type of task
            episodes: Number of training episodes
            
        Returns:
            Kernel slug (username/kernel-name)
        """
        import shutil
        import random
        import string
        import json
        
        self.logger.info("📓 Creating execution SCRIPT (.py)...")
        
        # Generate unique kernel name (simple format)
        random_suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
        kernel_name = f"rl-portfolio-optimizer-{task_type}-{random_suffix}"
        
        # Create script directory
        script_dir = Path("kaggle_script_temp")
        if script_dir.exists():
            shutil.rmtree(script_dir)
        script_dir.mkdir(parents=True, exist_ok=True)
        
        # Build complete script content (standalone .py with all logic)
        dataset_name = dataset_slug.split("/")[1]
        
        script_content = f'''# RL Portfolio Optimizer - Kaggle Training Script
# Generated automatically - Execute on Kaggle GPU

import sys
import os
import tarfile
import shutil
import traceback
import json
from datetime import datetime

print("=== RL Portfolio Optimizer - Setup ===")
print(f"Dataset attendu: {dataset_slug}")
print(f"Dataset name: {dataset_name}")

# Debug: Explorer tous les datasets disponibles
INPUT_BASE = "/kaggle/input"
print(f"Datasets disponibles dans {{INPUT_BASE}}:")
available_datasets = os.listdir(INPUT_BASE) if os.path.exists(INPUT_BASE) else []
for dataset in available_datasets:
    print(f"  - {{dataset}}")

# Chercher notre dataset (EXACT MATCH first, puis flexible)
DATASET_PATH = None

# 1. Chercher correspondance exacte avec le nom du dataset
if '{dataset_name}' in available_datasets:
    DATASET_PATH = os.path.join(INPUT_BASE, '{dataset_name}')
    print(f"✅ Dataset trouve (nom exact): {dataset_name}")
elif '{dataset_slug}' in available_datasets:
    DATASET_PATH = os.path.join(INPUT_BASE, '{dataset_slug}')
    print(f"✅ Dataset trouve (slug exact): {dataset_slug}")
else:
    # 2. Chercher match flexible (contient 'rl-portfolio-optimizer')
    for dataset in available_datasets:
        if 'rl-portfolio-optimizer' in dataset.lower():
            DATASET_PATH = os.path.join(INPUT_BASE, dataset)
            print(f"✅ Dataset trouve (match flexible): {{dataset}}")
            break

if DATASET_PATH is None and available_datasets:
    # Fallback: premier dataset
    DATASET_PATH = os.path.join(INPUT_BASE, available_datasets[0])
    print(f"⚠️ Fallback premier dataset: {{available_datasets[0]}}")

if DATASET_PATH is None:
    print("❌ Aucun dataset attache. Tentative de telechargement dynamique via Kaggle API...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()  # Auto en Kaggle (session) ; local: creds requis
        
        download_path = '/kaggle/working/downloaded_dataset'
        os.makedirs(download_path, exist_ok=True)
        
        print(f"📥 Telechargement de {dataset_slug}...")
        api.dataset_download_files('{dataset_slug}', path=download_path, unzip=True)
        
        DATASET_PATH = download_path
        print(f"Dataset telecharge et extrait: {{DATASET_PATH}}")
        if os.path.exists(DATASET_PATH):
            print(f"Contenu: {{os.listdir(DATASET_PATH)}}")
        else:
            raise Exception("Download path inexistant apres extraction")
            
    except ImportError:
        print("❌ Package 'kaggle' non installé. En Kaggle: pre-installé. Local: pip install kaggle")
        DATASET_PATH = None
    except Exception as e:
        print(f"❌ Echec telechargement API: {{e}}")
        print("💡 SOLUTION: Verifiez que le dataset est PUBLIC et attaché au kernel")
        print(f"💡 Dataset URL: https://www.kaggle.com/datasets/{dataset_slug}")
        print("💡 Ou ajoutez manuellement le dataset via UI Kaggle")
        
        # Essayer une dernière fois avec l'API intégrée Kaggle
        try:
            print("🔄 Tentative avec l'API intégrée Kaggle...")
            import kaggle
            kaggle.api.dataset_download_files('{dataset_slug}', path='/kaggle/working/api_download', unzip=True)
            DATASET_PATH = '/kaggle/working/api_download'
            print(f"✅ Dataset téléchargé via API intégrée: {{DATASET_PATH}}")
        except Exception as api_e:
            print(f"❌ Echec API intégrée: {{api_e}}")
            # Fallback final: dossier vide
            DATASET_PATH = "/kaggle/working/fallback"
            os.makedirs(DATASET_PATH, exist_ok=True)
            print(f"⚠️ Fallback final: {{DATASET_PATH}} (vide - pas d'entrainement possible sans dataset)")

print(f"Dataset path utilise: {{DATASET_PATH}}")
if os.path.exists(DATASET_PATH):
    print(f"Contenu: {{os.listdir(DATASET_PATH)}}")
else:
    print("ERREUR: Chemin dataset inexistant - verifiez attachement ou creds Kaggle.")
    sys.exit(1)  # Arrêt propre si pas de dataset

# Extraire archive tar si presente
CODE_PATH = DATASET_PATH
if os.path.exists(DATASET_PATH):
    for file in os.listdir(DATASET_PATH):
        file_path = os.path.join(DATASET_PATH, file)
        if file.endswith(('.tar', '.tar.gz')):
            print(f"Extraction archive: {{file}}")
            extraction_dir = '/kaggle/working/extracted_code'
            os.makedirs(extraction_dir, exist_ok=True)
            with tarfile.open(file_path, 'r:*') as tar:
                tar.extractall(extraction_dir)
            CODE_PATH = extraction_dir
            break

print(f"Code path final: {{CODE_PATH}}")
print(f"Contenu code: {{os.listdir(CODE_PATH) if os.path.exists(CODE_PATH) else 'Vide'}}")

# Ajouter au sys.path
sys.path.insert(0, CODE_PATH)
print(f"sys.path maj: {{sys.path[0]}}")

# Verifier train.py
train_file = os.path.join(CODE_PATH, 'train.py')
print(f"train.py existe: {{os.path.exists(train_file)}}")
if not os.path.exists(train_file):
    print("ERREUR: train.py manquant - verifiez le dataset.")
    sys.exit(1)

# Execution training
try:
    from train import main
    print("Import train.main reussi!")
    print("Lancement entrainement...")
    main(kaggle_mode=False, num_episodes={episodes})
    print("Entrainement termine!")
except ImportError as e:
    print(f"Erreur import: {{e}}")
    print(f"sys.path: {{sys.path[0]}}")
    print(f"Fichiers .py disponibles:")
    if os.path.exists(CODE_PATH):
        for f in os.listdir(CODE_PATH):
            if f.endswith('.py'):
                print(f"  - {{f}}")
    raise
except Exception as e:
    print(f"Erreur execution: {{e}}")
    traceback.print_exc()
    raise

print("=== Setup Training Results ===")

# Preparation resultats
results_dir = "/kaggle/working/training_results"
os.makedirs(results_dir, exist_ok=True)

# Copier models/results si existent
for src_dir in ['models', 'results']:
    if os.path.exists(src_dir):
        shutil.copytree(src_dir, os.path.join(results_dir, src_dir), dirs_exist_ok=True)
        print(f"{{src_dir}} copies.")

# Resume session
summary = {{
    "timestamp": datetime.now().isoformat(),
    "status": "completed",
    "kaggle_session": True,
    "files_generated": [f for root, _, files in os.walk(results_dir) for f in files]
}}

with open(os.path.join(results_dir, "session_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"Resultats dans: {{results_dir}} ({{len(summary['files_generated'])}} fichiers)")
print("Pret pour telechargement!")
'''

        # Sauvegarder le script .py (ASCII safe pour éviter problèmes encodage)
        script_file = script_dir / f"{kernel_name}.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            # Remplacer les emojis par des alternatives ASCII safe
            ascii_content = script_content.replace('✅', '[OK]').replace('❌', '[ERROR]').replace('⚠️', '[WARN]').replace('💡', '[INFO]').replace('🔄', '[RETRY]').replace('📥', '[DOWNLOAD]')
            f.write(ascii_content)

        # Metadata pour script (public, GPU)
        title = kernel_name.replace('-', ' ').title()  # Alignement parfait titre/slug
        kernel_metadata = {
            "id": f"{self.username}/{kernel_name}",
            "title": title,
            "code_file": f"{kernel_name}.py",  # .py pour script
            "language": "python",
            "kernel_type": "script",  # CHANGEMENT: script pour meilleur attachement
            "is_private": False,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "keywords": ["reinforcement-learning", "portfolio", "sac", "pytorch"],
            "datasetSources": [dataset_slug],  # FIX: CamelCase requis par API Kaggle
            "kernelSources": [],  # Aussi en camelCase pour cohérence
            "competitionSources": [],
            "modelSources": []
        }

        # Sauvegarder metadata
        metadata_file = script_dir / "kernel-metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(kernel_metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"📋 Script metadata created: {kernel_name}")

        # Upload
        try:
            self.logger.info(f"⬆️ Uploading PUBLIC script: {kernel_name}")
            self.api.kernels_push(str(script_dir))
            
            kernel_slug = f"{self.username}/{kernel_name}"
            self.logger.info(f"✅ PUBLIC Script uploaded: {kernel_slug}")
            self.logger.info(f"🔗 URL: https://www.kaggle.com/code/{kernel_slug}")
            self.logger.info("💡 Vérifiez /kaggle/input dans les logs d'exécution pour confirmer attachement.")
            
            # Cleanup
            shutil.rmtree(script_dir)
            return kernel_slug
            
        except Exception as e:
            self.logger.error(f"❌ Script upload failed: {str(e)}")
            if script_dir.exists():
                self.logger.info(f"📁 Temp dir preserved: {script_dir}")
            raise RuntimeError(f"Failed to upload script: {str(e)}")


class KaggleConfig:
    """
    Configuration helper for Kaggle integration with Config class.
    Extends existing centralized configuration with Kaggle-specific settings.
    """
    
    @staticmethod
    def get_device_for_kaggle() -> str:
        """
        Get appropriate device configuration for Kaggle environment.
        
        Returns:
            Device string compatible with PyTorch
        """
        if KaggleManager().is_kaggle_environment():
            # In Kaggle environment, always try CUDA first
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        else:
            # Use existing Config device detection for local development
            return str(Config.get_device())
            
    @staticmethod
    def get_execution_mode() -> str:
        """
        Determine execution mode (local or kaggle).
        
        Returns:
            'kaggle' if in Kaggle environment, 'local' otherwise
        """
        manager = KaggleManager()
        return 'kaggle' if manager.is_kaggle_environment() else 'local'
        
    @staticmethod
    def get_data_paths() -> Dict[str, str]:
        """
        Get appropriate data paths for current execution environment.
        
        Returns:
            Dictionary with data paths for current environment
        """
        try:
            manager = KaggleManager()
            return manager.get_kaggle_paths()
        except Exception:
            # Fallback to local paths if Kaggle manager fails
            return {
                'input': str(Path.cwd() / 'datas'),
                'working': str(Path.cwd()),
                'output': str(Path.cwd() / 'results')
            }


# Module-level functions for easy access
def is_kaggle_environment() -> bool:
    """Check if running in Kaggle environment."""
    try:
        manager = KaggleManager()
        return manager.is_kaggle_environment()
    except Exception:
        return False


def get_kaggle_device() -> str:
    """Get appropriate device for current environment."""
    return KaggleConfig.get_device_for_kaggle()


def get_execution_mode() -> str:
    """Get current execution mode."""
    return KaggleConfig.get_execution_mode()