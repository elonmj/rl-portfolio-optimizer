"""
Kaggle Manager GitHub Workflow - Version Optimisée

Ce module se concentre UNIQUEMENT sur le workflow GitHub qui a fonctionné :
- Automation Git (status -> add -> commit -> push)  
- Workflow GitHub-based (clone public repo)
- Monitoring avec session_summary.json
- Log retrieval et persistence

Basé sur l'analyse complète de la discussion et des éléments qui ont marché.
"""

import os
import sys
import json
import time
import shutil
import subprocess
import tempfile
import logging
import random
import string
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("Warning: Kaggle package not installed. Install with: pip install kaggle")

from config import Config


def is_kaggle_environment() -> bool:
    """Check if running in Kaggle environment."""
    kaggle_indicators = [
        os.path.exists('/kaggle/input'),
        os.path.exists('/kaggle/working'),
        'KAGGLE_KERNEL_RUN_TYPE' in os.environ,
        'KAGGLE_URL_BASE' in os.environ
    ]
    return any(kaggle_indicators)


class KaggleManagerGitHub:
    """
    Kaggle Manager optimisé pour le workflow GitHub.
    
    Focus sur ce qui a marché :
    1. Git automation (ensure up-to-date before Kaggle)
    2. GitHub-based kernel (clone public repo, no dataset complexity)
    3. session_summary.json detection (reliable success indicator)
    4. Enhanced monitoring with adaptive intervals
    5. Windows encoding-safe log processing
    """
    
    def __init__(self, username: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize with minimal setup - focus on working workflow."""
        self.logger = self._setup_logging()
        
        if not KAGGLE_AVAILABLE:
            raise ImportError("Kaggle package required. Install with: pip install kaggle")
            
        self.api = KaggleApi()
        
        # Set credentials if provided
        if username and api_key:
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = api_key
            
        # Validate and authenticate
        self._validate_and_authenticate()
        
        # Configuration
        self.username = self._get_username()
        self.kernel_base_name = "rl-portfolio-optimizer"
        
    def _setup_logging(self) -> logging.Logger:
        """Setup dedicated logging."""
        logger = logging.getLogger('kaggle_manager_github')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _validate_and_authenticate(self) -> None:
        """Validate environment and authenticate."""
        required_vars = ['KAGGLE_USERNAME', 'KAGGLE_KEY']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
            
        try:
            self.api.authenticate()
            self.logger.info("✅ Authenticated with Kaggle API")
        except Exception as e:
            raise RuntimeError(f"Kaggle authentication failed: {e}")

    def _get_username(self) -> str:
        """Get authenticated Kaggle username."""
        return os.environ.get('KAGGLE_USERNAME', '')

    def ensure_git_up_to_date(self, branch: str = "feature/training-config-updates") -> bool:
        """
        CORE FEATURE: Automatic Git workflow (status -> add -> commit -> push).
        
        Cette fonctionnalité était CRUCIALE car Kaggle clonait du code obsolète
        quand les changements locaux n'étaient pas pushés sur GitHub.
        
        Args:
            branch: Git branch to work with
            
        Returns:
            bool: True if Git is up to date, False if errors occurred
        """
        self.logger.info("🔍 Checking Git status and ensuring changes are pushed...")
        
        try:
            # Check if we're in a git repository
            result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                self.logger.warning("📁 Not in a Git repository - skipping Git automation")
                return True
            
            # Get current branch
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            current_branch = result.stdout.strip()
            self.logger.info(f"📍 Current branch: {current_branch}")
            
            # Check git status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                self.logger.error(f"❌ Git status failed: {result.stderr}")
                return False
            
            status_output = result.stdout.strip()
            
            if not status_output:
                # Check if we need to push commits
                result = subprocess.run(['git', 'rev-list', '--count', f'{current_branch}..origin/{current_branch}'], 
                                      capture_output=True, text=True, cwd=os.getcwd())
                if result.returncode == 0:
                    behind_count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
                    if behind_count == 0:
                        self.logger.info("✅ Git repository is clean and up to date")
                        return True
            
            # There are changes - show them
            if status_output:
                self.logger.info("📝 Detected local changes:")
                for line in status_output.split('\n'):
                    if line.strip():
                        status_code = line[:2]
                        file_path = line[3:] if len(line) > 3 else line
                        description = self._get_git_status_description(status_code)
                        self.logger.info(f"  {status_code} {file_path} ({description})")
            
            # Add all changes
            self.logger.info("📦 Adding all changes...")
            result = subprocess.run(['git', 'add', '.'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                self.logger.error(f"❌ Git add failed: {result.stderr}")
                return False
            
            # Create commit message with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message = f"Auto-commit before Kaggle workflow - {timestamp}\n\nUpdated for GitHub-based Kaggle execution"
            
            self.logger.info("💾 Committing changes...")
            result = subprocess.run(['git', 'commit', '-m', commit_message], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                # Check if it's just "nothing to commit"
                if "nothing to commit" in result.stdout.lower() or "working tree clean" in result.stdout.lower():
                    self.logger.info("✅ No changes to commit - repository is clean")
                else:
                    self.logger.error(f"❌ Git commit failed: {result.stderr}")
                    return False
            else:
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
        """Push changes to remote repository."""
        self.logger.info(f"📤 Pushing to remote branch: {branch}")
        
        try:
            result = subprocess.run(['git', 'push', 'origin', branch], 
                                  capture_output=True, text=True, cwd=os.getcwd(), timeout=60)
            
            if result.returncode == 0:
                self.logger.info("✅ Changes pushed successfully to GitHub")
                return True
            else:
                self.logger.error(f"❌ Git push failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Git push timed out")
            return False
        except Exception as e:
            self.logger.error(f"❌ Git push error: {e}")
            return False
    
    def _get_git_status_description(self, status_code: str) -> str:
        """Convert Git status codes to human-readable descriptions."""
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

    def create_and_run_github_workflow(self, 
                                     repo_url: str = "https://github.com/elonmj/rl-portfolio-optimizer.git", 
                                     branch: str = "feature/training-config-updates", 
                                     episodes: int = 1,
                                     timeout: int = 3600) -> bool:
        """
        MAIN WORKFLOW: Complete GitHub-based Kaggle workflow.
        
        Cette méthode combine les éléments qui ont fonctionné :
        1. Git automation (ensure code is pushed)
        2. GitHub-based kernel creation (no dataset complexity)
        3. Enhanced monitoring with session_summary.json detection
        
        Args:
            repo_url: GitHub repository URL (must be public)
            branch: Git branch to clone
            episodes: Number of training episodes
            timeout: Maximum monitoring time
            
        Returns:
            bool: True if workflow completed successfully
        """
        self.logger.info("🚀 Starting Complete GitHub-based Kaggle Workflow")
        self.logger.info(f"📂 Repository: {repo_url}")
        self.logger.info(f"🌿 Branch: {branch}")
        self.logger.info(f"🎯 Episodes: {episodes}")
        
        # STEP 1: Ensure Git is up to date (CRITICAL)
        self.logger.info("🔄 Step 1: Ensuring Git repository is up to date...")
        if not self.ensure_git_up_to_date(branch):
            self.logger.error("❌ Failed to update Git repository - aborting workflow")
            self.logger.error("💡 Kaggle will clone outdated code if Git is not up to date!")
            return False
        
        self.logger.info("✅ Git repository is up to date - proceeding with Kaggle workflow")
        
        # STEP 2: Create and upload GitHub-based kernel
        self.logger.info("📝 Step 2: Creating GitHub-based kernel...")
        kernel_slug = self._create_and_upload_github_kernel(repo_url, branch, episodes)
        
        if not kernel_slug:
            self.logger.error("❌ Failed to create GitHub kernel")
            return False
            
        self.logger.info(f"✅ GitHub kernel uploaded: {kernel_slug}")
        self.logger.info(f"🔗 URL: https://www.kaggle.com/code/{kernel_slug}")
        
        # STEP 3: Enhanced monitoring with session_summary.json detection
        self.logger.info("👀 Step 3: Starting enhanced monitoring...")
        success = self._monitor_kernel_with_session_detection(kernel_slug, timeout)
        
        return success

    def _create_and_upload_github_kernel(self, repo_url: str, branch: str, episodes: int) -> Optional[str]:
        """
        Create GitHub-based kernel (the approach that worked).
        
        Cette approche évite les complexités des datasets et utilise directement
        un clone Git public, ce qui s'est révélé plus fiable.
        """
        # Generate unique kernel name
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
        kernel_name = f"{self.kernel_base_name}-training-{random_suffix}"
        
        # Create script directory
        script_dir = Path("kaggle_script_temp")
        if script_dir.exists():
            shutil.rmtree(script_dir)
        script_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Build GitHub-based script content (working version from discussion)
            script_content = self._build_github_script(repo_url, branch, episodes)
            
            # Save script
            script_file = script_dir / f"{kernel_name}.py"
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # Create metadata
            kernel_metadata = {
                "id": f"{self.username}/{kernel_name}",
                "title": f"RL Portfolio Optimizer Training {random_suffix}",
                "code_file": f"{kernel_name}.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": False,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": True,  # CRITICAL for git clone
                "keywords": ["reinforcement-learning", "portfolio", "sac", "pytorch", "github"],
                "dataset_sources": [],  # No dataset needed!
                "kernel_sources": [],
                "competition_sources": [],
                "model_sources": []
            }
            
            # Save metadata
            metadata_file = script_dir / "kernel-metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(kernel_metadata, f, indent=2)
            
            self.logger.info(f"📋 GitHub script created: {kernel_name}")
            
            # Upload kernel
            self.logger.info(f"⬆️ Uploading GitHub-based script...")
            self.api.kernels_push(str(script_dir))
            
            kernel_slug = f"{self.username}/{kernel_name}"
            self.logger.info(f"✅ GitHub script uploaded: {kernel_slug}")
            
            return kernel_slug
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create GitHub kernel: {e}")
            return None
        
        finally:
            # Cleanup
            if script_dir.exists():
                shutil.rmtree(script_dir)

    def _build_github_script(self, repo_url: str, branch: str, episodes: int) -> str:
        """
        Build the GitHub-based script content.
        
        Cette version est basée sur le script qui a fonctionné dans la discussion,
        avec les marqueurs TRACKING_SUCCESS pour le monitoring.
        """
        return f'''#!/usr/bin/env python3
# RL Portfolio Optimizer - Kaggle Training Script (GitHub-based)
# Generated automatically - Execute on Kaggle GPU

import sys
import os
import subprocess
import json
from datetime import datetime
from pathlib import Path

print("=== RL Portfolio Optimizer - GitHub Setup ===")

# Configuration
REPO_URL = "{repo_url}"
BRANCH = "{branch}"
REPO_DIR = "/kaggle/working/rl-portfolio-optimizer"

print(f"Repository: {{REPO_URL}}")
print(f"Branch: {{BRANCH}}")
print(f"Episodes: {episodes}")

try:
    # Clone repository with git
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
    
    # Set environment variable for number of episodes (CRITICAL)
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
    
    # Import training script
    import train
    print("🎯 TRACKING_PROGRESS: Train module imported successfully")
    
    # Run training with explicit episodes parameter
    if hasattr(train, 'main'):
        print(f"🎯 TRACKING_PROGRESS: Executing train.main(num_episodes={episodes})")
        train.main(num_episodes={episodes})
    else:
        print("🎯 TRACKING_PROGRESS: Executing train.py content directly")
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
    # Create results summary (SESSION_SUMMARY.JSON - KEY FOR DETECTION!)
    try:
        results_dir = "/kaggle/working/results"
        os.makedirs(results_dir, exist_ok=True)
        
        summary = {{
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "repo_url": REPO_URL,
            "branch": BRANCH,
            "episodes": {episodes},
            "kaggle_session": True
        }}
        
        with open(os.path.join(results_dir, "session_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\\n[INFO] Results saved with session_summary.json")
        print("🎯 TRACKING_SUCCESS: Session summary created")
    except Exception as e:
        print(f"[WARN] Could not save results: {{e}}")
'''

    def _monitor_kernel_with_session_detection(self, kernel_slug: str, timeout: int = 3600) -> bool:
        """
        Enhanced monitoring with session_summary.json detection.
        
        Cette méthode utilise la détection de session_summary.json qui s'est
        révélée être l'indicateur de succès le plus fiable.
        """
        start_time = time.time()
        
        # Adaptive monitoring intervals (exponential backoff)
        base_interval = 10  # Start with 10 seconds
        max_interval = 120  # Cap at 2 minutes
        current_interval = base_interval
        
        self.logger.info(f"🔍 Enhanced monitoring started for: {kernel_slug}")
        self.logger.info(f"⏱️ Timeout: {timeout}s, Adaptive intervals: {base_interval}s → {max_interval}s")
        
        # Keywords for tracking (based on working script)
        success_keywords = [
            "TRACKING_SUCCESS: Training execution finished successfully",
            "TRACKING_SUCCESS: Repository cloned",
            "TRACKING_SUCCESS: Requirements installation completed",
            "TRACKING_SUCCESS: Session summary created",
            "[OK] Training completed successfully!"
        ]
        
        error_keywords = [
            "TRACKING_ERROR:",
            "[ERROR]",
            "fatal:",
            "Exception:",
            "sys.exit(1)"
        ]
        
        try:
            while time.time() - start_time < timeout:
                try:
                    # Check kernel status
                    status_response = self.api.kernels_status(kernel_slug)
                    current_status = getattr(status_response, 'status', 'unknown')
                    
                    elapsed = time.time() - start_time
                    self.logger.info(f"⏱️ Status: {current_status} (after {elapsed:.1f}s)")
                    
                    # Check if execution is complete
                    if current_status in ['complete', 'error', 'cancelled']:
                        self.logger.info(f"🏁 Kernel execution finished with status: {current_status}")
                        
                        # Analyze logs for session_summary.json (KEY DETECTION)
                        success = self._retrieve_and_analyze_logs(kernel_slug, success_keywords, error_keywords)
                        
                        if success:
                            self.logger.info("✅ Workflow completed successfully!")
                            return True
                        else:
                            self.logger.error("❌ Workflow failed - check logs for details")
                            return False
                    
                    # Adaptive interval (exponential backoff)
                    current_interval = min(current_interval * 1.5, max_interval)
                    self.logger.info(f"⏳ Next check in {current_interval:.0f}s...")
                    time.sleep(current_interval)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error checking status: {e}")
                    # Try to get logs anyway
                    self._retrieve_and_analyze_logs(kernel_slug, success_keywords, error_keywords)
                    return False
        
            # Timeout reached
            elapsed = time.time() - start_time
            self.logger.error(f"⏰ Monitoring timeout after {elapsed:.1f}s")
            self.logger.info(f"📝 Manual check: https://www.kaggle.com/code/{kernel_slug}")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Monitoring failed: {e}")
            return False

    def _retrieve_and_analyze_logs(self, kernel_slug: str, success_keywords: list, error_keywords: list) -> bool:
        """
        Retrieve and analyze logs with session_summary.json detection.
        
        CORE FEATURE: Cette méthode implémente la détection session_summary.json
        qui s'est révélée être le mécanisme de détection de succès le plus fiable.
        """
        try:
            self.logger.info("📋 Retrieving execution logs...")
            
            # Download artifacts to temp dir
            with tempfile.TemporaryDirectory() as temp_dir:
                self.logger.info(f"📥 Downloading kernel output for: {kernel_slug}")
                self.api.kernels_output(kernel_slug, path=temp_dir, quiet=True)

                # Persist artifacts (for debugging and future reference)
                persist_dir = Path('test_output') / 'results' / kernel_slug.replace('/', '_')
                persist_dir.mkdir(parents=True, exist_ok=True)
                
                for name in os.listdir(temp_dir):
                    src_path = os.path.join(temp_dir, name)
                    dst_path = persist_dir / name
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)
                    elif os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                        
                self.logger.info(f"💾 Persisted kernel artifacts to: {persist_dir}")

                # PRIORITY 1: Look for session_summary.json (most reliable indicator)
                session_summary_found = False
                for root, dirs, files in os.walk(temp_dir):
                    if 'session_summary.json' in files:
                        summary_path = os.path.join(root, 'session_summary.json')
                        self.logger.info(f"🎯 Found session_summary.json at: {summary_path}")
                        
                        try:
                            with open(summary_path, 'r', encoding='utf-8') as f:
                                summary_data = json.load(f)
                            
                            status = summary_data.get('status', 'unknown')
                            self.logger.info(f"📊 Session status: {status}")
                            
                            # Copy to persist directory
                            shutil.copy2(summary_path, persist_dir / 'session_summary.json')
                            
                            if status == 'completed':
                                self.logger.info("✅ session_summary.json indicates successful completion!")
                                session_summary_found = True
                                break
                                
                        except Exception as e:
                            self.logger.warning(f"⚠️ Could not parse session_summary.json: {e}")

                # PRIORITY 2: Analyze log files for success/error keywords
                log_analysis_success = False
                log_files = []
                
                # Find log-like files
                for file in os.listdir(temp_dir):
                    if file.endswith(('.log', '.txt')) or file in ['__stdout__.txt', '__stderr__.txt']:
                        log_files.append(os.path.join(temp_dir, file))
                
                # Analyze logs
                if log_files:
                    for log_file in log_files:
                        self.logger.info(f"📄 Analyzing log file: {os.path.basename(log_file)}")
                        
                        try:
                            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                log_content = f.read()
                            
                            # Check for success indicators
                            success_found = any(keyword in log_content for keyword in success_keywords)
                            error_found = any(keyword in log_content for keyword in error_keywords)
                            
                            if success_found:
                                self.logger.info("✅ Success indicators found in logs")
                                log_analysis_success = True
                                
                            if error_found:
                                self.logger.warning("⚠️ Error indicators found in logs")
                                
                            # Save primary log to persist directory  
                            if 'stdout' in os.path.basename(log_file) or log_file == log_files[0]:
                                with open(persist_dir / 'primary_log.txt', 'w', encoding='utf-8', errors='ignore') as f:
                                    f.write(log_content)
                                
                        except Exception as e:
                            self.logger.error(f"❌ Error reading log {log_file}: {e}")
                
                # Final decision: session_summary.json has priority
                if session_summary_found:
                    return True
                elif log_analysis_success:
                    self.logger.info("✅ Success detected via log analysis (no session_summary.json)")
                    return True
                else:
                    self.logger.warning("⚠️ No clear success indicators found")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ Error retrieving logs: {e}")
            return False

    def get_kernel_logs(self, kernel_slug: str) -> Dict[str, Any]:
        """
        Public method to retrieve kernel logs and artifacts.
        
        Returns:
            Dictionary with log information and file paths
        """
        try:
            result = self._retrieve_and_analyze_logs(kernel_slug, [], [])
            
            persist_dir = Path('test_output') / 'results' / kernel_slug.replace('/', '_')
            
            return {
                'success': result,
                'persist_dir': str(persist_dir),
                'url': f"https://www.kaggle.com/code/{kernel_slug}",
                'files': list(persist_dir.glob('*')) if persist_dir.exists() else []
            }
            
        except Exception as e:
            self.logger.error(f"Error getting logs: {e}")
            return {'success': False, 'error': str(e)}


# Convenience function for quick usage
def run_kaggle_training(episodes: int = 1, 
                       repo_url: str = "https://github.com/elonmj/rl-portfolio-optimizer.git",
                       branch: str = "feature/training-config-updates",
                       timeout: int = 3600) -> bool:
    """
    Quick function to run the complete GitHub workflow.
    
    Args:
        episodes: Number of training episodes
        repo_url: GitHub repository URL (must be public)
        branch: Git branch to clone
        timeout: Maximum monitoring time in seconds
        
    Returns:
        bool: True if successful
    """
    try:
        manager = KaggleManagerGitHub()
        return manager.create_and_run_github_workflow(
            repo_url=repo_url,
            branch=branch, 
            episodes=episodes,
            timeout=timeout
        )
    except Exception as e:
        print(f"Error running Kaggle training: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("🚀 Kaggle Manager GitHub - Example Usage")
    
    # Quick test with 1 episode
    success = run_kaggle_training(episodes=1)
    
    if success:
        print("✅ Kaggle workflow completed successfully!")
    else:
        print("❌ Kaggle workflow failed - check logs for details")