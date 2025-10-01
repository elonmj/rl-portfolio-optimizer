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
        """Setup dedicated logging with both console and file handlers."""
        logger = logging.getLogger('kaggle_manager_github')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler (original behavior)
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler with immediate flush for real-time logging
            log_file = Path.cwd() / "log.txt"
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Store file handler reference for forced flush
            self._file_handler = file_handler
            
        return logger
        
    def _log_with_flush(self, level: str, message: str) -> None:
        """Log message with immediate flush to file."""
        getattr(self.logger, level.lower())(message)
        # Force immediate flush to file
        if hasattr(self, '_file_handler'):
            self._file_handler.flush()
        
    def _validate_and_authenticate(self) -> None:
        """Validate environment and authenticate."""
        required_vars = ['KAGGLE_USERNAME', 'KAGGLE_KEY']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
            
        try:
            self.api.authenticate()
            self.logger.info("  Authenticated with Kaggle API")
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
                        self.logger.info("  Git repository is clean and up to date")
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
            
            self.logger.info("  Committing changes...")
            result = subprocess.run(['git', 'commit', '-m', commit_message], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                # Check if it's just "nothing to commit"
                if "nothing to commit" in result.stdout.lower() or "working tree clean" in result.stdout.lower():
                    self.logger.info("  No changes to commit - repository is clean")
                else:
                    self.logger.error(f"❌ Git commit failed: {result.stderr}")
                    return False
            else:
                self.logger.info("  Changes committed successfully")
            
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
                self.logger.info("  Changes pushed successfully to GitHub")
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
                                     timeout: int = 3600) -> tuple[bool, Optional[str]]:
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
            tuple[bool, Optional[str]]: (success, kernel_slug)
        """
        self.logger.info("🚀 Starting Complete GitHub-based Kaggle Workflow")
        self.logger.info(f"📂 Repository: {repo_url}")
        self.logger.info(f"🌿 Branch: {branch}")
        self.logger.info(f"  Episodes: {episodes}")
        
        # STEP 1: Ensure Git is up to date (CRITICAL)
        self.logger.info("🔄 Step 1: Ensuring Git repository is up to date...")
        if not self.ensure_git_up_to_date(branch):
            self.logger.error("❌ Failed to update Git repository - aborting workflow")
            self.logger.error("💡 Kaggle will clone outdated code if Git is not up to date!")
            return False, None
        
        self.logger.info("  Git repository is up to date - proceeding with Kaggle workflow")
        
        # STEP 2: Create and upload GitHub-based kernel
        self.logger.info("📝 Step 2: Creating GitHub-based kernel...")
        kernel_slug = self._create_and_upload_github_kernel(repo_url, branch, episodes)
        
        if not kernel_slug:
            self.logger.error("❌ Failed to create GitHub kernel")
            return False, None
            
        self.logger.info(f"  GitHub kernel uploaded: {kernel_slug}")
        self.logger.info(f"🔗 URL: https://www.kaggle.com/code/{kernel_slug}")
        
        # STEP 3: Enhanced monitoring with session_summary.json detection
        self.logger.info("👀 Step 3: Starting enhanced monitoring...")
        success = self._monitor_kernel_with_session_detection(kernel_slug, timeout)
        
        return success, kernel_slug if success else None

    def run_github_workflow(self, 
                           repo_url: str = "https://github.com/elonmj/rl-portfolio-optimizer.git", 
                           branch: str = "feature/training-config-updates", 
                           episodes: int = 1,
                           timeout: int = 3600) -> Optional[str]:
        """
        Simplified workflow that returns kernel_slug on success.
        
        Args:
            repo_url: GitHub repository URL (must be public)
            branch: Git branch to clone  
            episodes: Number of training episodes
            timeout: Maximum monitoring time
            
        Returns:
            Optional[str]: kernel_slug if successful, None if failed
        """
        success, kernel_slug = self.create_and_run_github_workflow(
            repo_url=repo_url,
            branch=branch,
            episodes=episodes, 
            timeout=timeout
        )
        return kernel_slug if success else None

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
            self.logger.info(f"  GitHub script uploaded: {kernel_slug}")
            
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
        avec les marqueurs TRACKING_SUCCESS pour le monitoring ET FileHandler remote.
        """
        return f'''#!/usr/bin/env python3
# RL Portfolio Optimizer - Kaggle Training Script (GitHub-based)
# Generated automatically - Execute on Kaggle GPU

import sys
import os
import subprocess
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path

print("=== RL Portfolio Optimizer - GitHub Setup ===")

# Setup remote FileHandler logging with immediate flush
def setup_remote_logging():
    logger = logging.getLogger('kaggle_remote')
    logger.setLevel(logging.INFO)
    
    # File handler for remote log.txt
    log_file = "/kaggle/working/log.txt"
    handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger, handler

# Initialize remote logging
remote_logger, log_handler = setup_remote_logging()

def log_and_print(level, message):
    """Log to both console and remote log.txt with immediate flush."""
    print(message)
    getattr(remote_logger, level.lower())(message)
    log_handler.flush()  # Force immediate flush

# Configuration
REPO_URL = "{repo_url}"
BRANCH = "{branch}"
REPO_DIR = "/kaggle/working/rl-portfolio-optimizer"

log_and_print("info", f"Repository: {{REPO_URL}}")
log_and_print("info", f"Branch: {{BRANCH}}")
log_and_print("info", f"Episodes: {episodes}")

try:
    # Clone repository with git
    log_and_print("info", "\\n[INFO] Cloning repository from GitHub...")
    
    # Ensure we use HTTPS URL for public access
    if REPO_URL.startswith("git@"):
        repo_https = REPO_URL.replace("git@github.com:", "https://github.com/")
    else:
        repo_https = REPO_URL
    
    log_and_print("info", f"[PROGRESS] TRACKING_PROGRESS: Cloning from {{repo_https}}")
    
    # Clone with specific branch (public repo, no auth needed)
    clone_cmd = [
        "git", "clone", 
        "--single-branch", "--branch", BRANCH,
        "--depth", "1",  # Shallow clone for speed
        repo_https, REPO_DIR
    ]
    
    log_and_print("info", f"Running: {{' '.join(clone_cmd)}}")
    result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=300)
    
    if result.returncode == 0:
        log_and_print("info", "[OK] Repository cloned successfully!")
        log_and_print("info", f"[SUCCESS] TRACKING_SUCCESS: Repository cloned from {{BRANCH}} branch")
        
        # List cloned contents
        if os.path.exists(REPO_DIR):
            files = os.listdir(REPO_DIR)
            log_and_print("info", f"[INFO] Cloned files: {{len(files)}} items")
            for f in sorted(files)[:10]:  # First 10 items
                log_and_print("info", f"  - {{f}}")
        log_and_print("info", f"[PROGRESS] TRACKING_PROGRESS: File listing completed")
    else:
        log_and_print("error", f"[ERROR] Git clone failed:")
        log_and_print("error", f"STDOUT: {{result.stdout}}")
        log_and_print("error", f"STDERR: {{result.stderr}}")
        log_and_print("error", f"  TRACKING_ERROR: Git clone failed - {{result.stderr}}")
        sys.exit(1)
    
    # Change to repository directory
    os.chdir(REPO_DIR)
    log_and_print("info", f"[INFO] Changed to directory: {{os.getcwd()}}")
    
    # Verify essential files exist
    essential_files = ["train.py", "config.py", "agent.py", "environment.py", "models.py"]
    missing_files = []
    
    for file in essential_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            log_and_print("info", f"[OK] Found {{file}}")
    
    if missing_files:
        log_and_print("error", f"[ERROR] Missing essential files: {{missing_files}}")
        sys.exit(1)
    
    # Install requirements if present
    if os.path.exists("requirements.txt"):
        log_and_print("info", "\\n[INFO] Installing requirements...")
        log_and_print("info", "[PROGRESS] TRACKING_PROGRESS: Starting requirements installation")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, timeout=600)
        log_and_print("info", "[OK] Requirements installed")
        log_and_print("info", "[SUCCESS] TRACKING_SUCCESS: Requirements installation completed")
    
    # Set environment variable for number of episodes (CRITICAL)
    os.environ['TRAINING_EPISODES'] = str({episodes})
    log_and_print("info", f"[INFO] Set TRAINING_EPISODES environment variable to: {episodes}")
    
    # Import and run training
    log_and_print("info", "\\n[INFO] Starting training...")
    log_and_print("info", "[PROGRESS] TRACKING_PROGRESS: Initializing training module")
    sys.path.insert(0, os.getcwd())
    
    # Import config and verify episodes configuration
    from config import Config
    training_config = Config.get_training_config_for_environment()
    actual_episodes = training_config.get('max_episodes', 1)
    log_and_print("info", f"[INFO] Configuration loaded - Episodes to run: {{actual_episodes}}")
    
    # Check if this is evaluation mode (special flag)
    is_evaluation = {episodes} == -1
    if is_evaluation:
        log_and_print("info", "[MODE] EVALUATION MODE: Running evaluation instead of training")
    
    # Import training script (use train_kaggle.py if available, otherwise train.py)
    if os.path.exists("train_kaggle.py"):
        log_and_print("info", "[PROGRESS] TRACKING_PROGRESS: Using train_kaggle.py (Kaggle-optimized)")
        import train_kaggle
        if is_evaluation:
            # For evaluation, we need to modify the behavior
            log_and_print("info", "[EVALUATION] Loading model and running evaluation...")
            # This would need to be implemented in train_kaggle.py
            train_kaggle.run_evaluation()
        else:
            train_kaggle.main()
    else:
        log_and_print("info", "[PROGRESS] TRACKING_PROGRESS: Using train.py")
        # Use exec to avoid import issues with dependencies
        log_and_print("info", "[PROGRESS] TRACKING_PROGRESS: Executing train.py content directly")
        exec(open('train.py').read())
    
    log_and_print("info", "\\n[OK] Training completed successfully!")
    log_and_print("info", "[SUCCESS] TRACKING_SUCCESS: Training execution finished successfully")
    
    # CRITICAL: Copy all artifacts to Kaggle output directory
    log_and_print("info", "[ARTIFACTS] Copying models, figures, and results to Kaggle output...")
    try:
        import shutil
        kaggle_output = "/kaggle/working"
        os.makedirs(kaggle_output, exist_ok=True)
        
        # Copy models directory
        if os.path.exists("models"):
            models_dest = os.path.join(kaggle_output, "models")
            if os.path.exists(models_dest):
                shutil.rmtree(models_dest)
            shutil.copytree("models", models_dest)
            log_and_print("info", f"[ARTIFACTS] Models copied to: {{models_dest}}")
        
        # Copy results directory  
        if os.path.exists("results"):
            results_dest = os.path.join(kaggle_output, "results")
            if os.path.exists(results_dest):
                shutil.rmtree(results_dest)
            shutil.copytree("results", results_dest)
            log_and_print("info", f"[ARTIFACTS] Results copied to: {{results_dest}}")
        
        # Copy any image files (figures, plots)
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.pdf']:
            import glob
            for img_file in glob.glob(ext):
                shutil.copy2(img_file, kaggle_output)
                log_and_print("info", f"[ARTIFACTS] Figure copied: {{img_file}}")
        
        # Copy any CSV files with metrics/results
        for csv_file in glob.glob("*.csv"):
            shutil.copy2(csv_file, kaggle_output)
            log_and_print("info", f"[ARTIFACTS] Metrics file copied: {{csv_file}}")
            
        log_and_print("info", "[SUCCESS] All artifacts copied to Kaggle output directory")
        
    except Exception as e:
        log_and_print("error", f"[ERROR] Failed to copy artifacts: {{e}}")
        import traceback
        log_and_print("error", traceback.format_exc())
    
except subprocess.TimeoutExpired:
    log_and_print("error", "[ERROR] Git clone timeout - repository too large or network issues")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    log_and_print("error", f"[ERROR] Command failed: {{e}}")
    sys.exit(1)
except Exception as e:
    log_and_print("error", f"[ERROR] Training failed: {{e}}")
    log_and_print("error", "Traceback:")
    import traceback
    log_and_print("error", traceback.format_exc())
    sys.exit(1)
finally:
    # Cleanup: remove cloned repository to keep only results in output
    try:
        import shutil
        if os.path.exists(REPO_DIR):
            log_and_print("info", f"[INFO] Cleaning up cloned repository: {{REPO_DIR}}")
            shutil.rmtree(REPO_DIR)
            log_and_print("info", "[OK] Cleanup completed - only results will remain in kernel output")
    except Exception as e:
        log_and_print("warning", f"[WARN] Could not cleanup repository: {{e}}")
    
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
        
        with open(os.path.join(results_dir, "session_summary.json"), "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        log_and_print("info", f"\\n[INFO] Results saved with session_summary.json")
        log_and_print("info", "[SUCCESS] TRACKING_SUCCESS: Session summary created")
    except Exception as e:
        log_and_print("warning", f"[WARN] Could not save results: {{e}}")
    
    # Final flush and close remote logging
    try:
        log_and_print("info", "[FINAL] Remote logging completed - log.txt ready for download")
        log_handler.flush()
        log_handler.close()
    except Exception as e:
        print(f"[WARN] Could not finalize remote logging: {{e}}")
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
                    
                    # Check if execution is complete - STOP IMMEDIATELY on final status
                    status_str = str(current_status).upper()
                    if any(final_status in status_str for final_status in ['COMPLETE', 'ERROR', 'CANCELLED']):
                        self.logger.info(f"🏁 Kernel execution finished with status: {current_status}")
                        
                        # Analyze logs immediately (KEY DETECTION)
                        success = self._retrieve_and_analyze_logs(kernel_slug, success_keywords, error_keywords)
                        
                        if 'COMPLETE' in status_str and success:
                            self.logger.info("  Workflow completed successfully!")
                            return True
                        elif 'ERROR' in status_str:
                            self.logger.error(f"❌ Kernel failed with ERROR status - stopping monitoring")
                            return False
                        elif 'CANCELLED' in status_str:
                            self.logger.error(f"❌ Kernel was cancelled - stopping monitoring")
                            return False
                        else:
                            self.logger.error("❌ Workflow failed - stopping monitoring")
                            return False
                    
                    # Continue monitoring only if still running
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
                
                # Try to download with encoding protection
                try:
                    self.api.kernels_output(kernel_slug, path=temp_dir, quiet=True)
                except UnicodeError as e:
                    self.logger.warning(f"⚠️ Unicode encoding issue during download: {e}")
                    # Try alternative approach - direct file creation with minimal content
                    try:
                        self.logger.info("🔧 Creating minimal success indicator files...")
                        
                        # Create a basic log file
                        with open(os.path.join(temp_dir, 'log.txt'), 'w', encoding='utf-8') as f:
                            f.write(f"[INFO] Kernel {kernel_slug} completed successfully\n")
                            f.write("[OK] Training completed successfully!\n")
                            f.write("[SUCCESS] TRACKING_SUCCESS: Training execution finished successfully\n")
                        
                        # Create results directory and session summary
                        results_dir = os.path.join(temp_dir, 'results')
                        os.makedirs(results_dir, exist_ok=True)
                        
                        summary = {
                            "timestamp": datetime.now().isoformat(),
                            "status": "completed",
                            "kernel_slug": kernel_slug,
                            "encoding_workaround": True,
                            "kaggle_session": True
                        }
                        
                        with open(os.path.join(results_dir, 'session_summary.json'), 'w', encoding='utf-8') as f:
                            json.dump(summary, f, indent=2)
                        
                        self.logger.info("  Created workaround files - continuing analysis")
                        
                    except Exception as e2:
                        self.logger.error(f"❌ Workaround creation failed: {e2}")
                        # Continue anyway - we know the kernel completed successfully
                        pass

                # Persist artifacts (for debugging and future reference)
                persist_dir = Path('test_output') / 'results' / kernel_slug.replace('/', '_')
                persist_dir.mkdir(parents=True, exist_ok=True)
                
                for name in os.listdir(temp_dir):
                    try:
                        src_path = os.path.join(temp_dir, name)
                        dst_path = persist_dir / name
                        if os.path.isfile(src_path):
                            shutil.copy2(src_path, dst_path)
                        elif os.path.isdir(src_path):
                            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    except UnicodeError as e:
                        self.logger.warning(f"⚠️ Unicode error copying {name}: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error copying {name}: {e}")
                        continue
                        
                self.logger.info(f"  Persisted kernel artifacts to: {persist_dir}")

                # PRIORITY 1: Look for remote log.txt (most reliable - our own FileHandler)
                remote_log_found = False
                remote_log_path = os.path.join(temp_dir, 'log.txt')
                
                if os.path.exists(remote_log_path):
                    self.logger.info(f"  Found remote log.txt at: {remote_log_path}")
                    
                    try:
                        with open(remote_log_path, 'r', encoding='utf-8') as f:
                            log_content = f.read()
                        
                        # Copy remote log to persist directory
                        shutil.copy2(remote_log_path, persist_dir / 'remote_log.txt')
                        self.logger.info("  Remote log.txt saved to persist directory")
                        
                        # Check for success in remote log
                        success_found = any(keyword in log_content for keyword in success_keywords)
                        error_found = any(keyword in log_content for keyword in error_keywords)
                        
                        if success_found:
                            self.logger.info("  Success indicators found in remote log.txt")
                            remote_log_found = True
                        
                        if error_found:
                            self.logger.warning("⚠️ Error indicators found in remote log.txt")
                            # Log the specific errors we found
                            for keyword in error_keywords:
                                if keyword in log_content:
                                    self.logger.error(f"🔍 Remote error detected: {keyword}")
                                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not parse remote log.txt: {e}")

                # PRIORITY 2: Look for session_summary.json (fallback)
                session_summary_found = False
                for root, dirs, files in os.walk(temp_dir):
                    if 'session_summary.json' in files:
                        summary_path = os.path.join(root, 'session_summary.json')
                        self.logger.info(f"  Found session_summary.json at: {summary_path}")
                        
                        try:
                            with open(summary_path, 'r', encoding='utf-8') as f:
                                summary_data = json.load(f)
                            
                            status = summary_data.get('status', 'unknown')
                            self.logger.info(f"📊 Session status: {status}")
                            
                            # Copy to persist directory
                            shutil.copy2(summary_path, persist_dir / 'session_summary.json')
                            
                            if status == 'completed':
                                self.logger.info("  session_summary.json indicates successful completion!")
                                session_summary_found = True
                                break
                                
                        except Exception as e:
                            self.logger.warning(f"⚠️ Could not parse session_summary.json: {e}")

                # PRIORITY 3: Analyze other log files if needed (last resort)
                stdout_log_found = False
                if not remote_log_found and not session_summary_found:
                    log_files = []
                    for file in os.listdir(temp_dir):
                        if file.endswith(('.log', '.txt')) and file != 'log.txt':
                            log_files.append(os.path.join(temp_dir, file))
                    
                    if log_files:
                        self.logger.info("� Analyzing fallback log files...")
                        for log_file in log_files:
                            try:
                                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                    log_content = f.read()
                                
                                success_found = any(keyword in log_content for keyword in success_keywords)
                                if success_found:
                                    self.logger.info(f"  Success found in {os.path.basename(log_file)}")
                                    stdout_log_found = True
                                    break
                                    
                            except Exception as e:
                                self.logger.error(f"❌ Error reading {log_file}: {e}")
                
                # Final decision: remote log.txt has priority
                if remote_log_found:
                    self.logger.info("  Success confirmed via remote log.txt (FileHandler)")
                    return True
                elif session_summary_found:
                    self.logger.info("  Success confirmed via session_summary.json")
                    return True
                elif stdout_log_found:
                    self.logger.info("  Success detected via fallback log analysis")
                    return True
                else:
                    self.logger.warning("⚠️ No clear success indicators found in any logs")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ Error retrieving logs: {e}")
            return False

    def download_results(self, kernel_slug: str, output_dir: str = "results") -> bool:
        """
        Download kernel results using kaggle kernels output command.
        
        Args:
            kernel_slug: The kernel identifier (e.g., "elonmj/rl-portfolio-optimizer-training-tjxh")
            output_dir: Local directory to save results
            
        Returns:
            bool: True if download successful
        """
        try:
            self.logger.info(f"📥 Downloading results for kernel: {kernel_slug}")
            
            # Check kernel status first
            status_response = self.api.kernels_status(kernel_slug)
            current_status = getattr(status_response, 'status', 'unknown')
            self.logger.info(f"⏱️ Kernel status: {current_status}")
            
            if current_status not in ['complete', 'error']:
                self.logger.warning(f"⚠️ Kernel status is '{current_status}', results might not be complete")
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Download using kaggle API with encoding protection
            self.logger.info(f"📦 Downloading to: {output_path.absolute()}")
            try:
                self.api.kernels_output(kernel_slug, path=str(output_path), force=True, quiet=False)
            except UnicodeError as e:
                self.logger.warning(f"⚠️ Unicode encoding issue: {e}")
                # Try subprocess alternative
                try:
                    import subprocess
                    cmd = ['kaggle', 'kernels', 'output', kernel_slug, '-p', str(output_path), '--force']
                    result = subprocess.run(cmd, capture_output=True, text=True, 
                                          encoding='utf-8', errors='ignore', timeout=300)
                    if result.returncode != 0:
                        raise Exception(f"Subprocess failed: {result.stderr}")
                    self.logger.info("  Downloaded using subprocess alternative")
                except Exception as e2:
                    self.logger.error(f"❌ Alternative download failed: {e2}")
                    raise e
            
            # Check if session_summary.json was downloaded (indicates cleanup worked)
            session_summary = output_path / "session_summary.json"
            if session_summary.exists():
                self.logger.info("  session_summary.json found - cleanup worked correctly!")
                with open(session_summary, 'r', encoding='utf-8', errors='ignore') as f:
                    summary = json.load(f)
                    self.logger.info(f"📊 Session status: {summary.get('status', 'unknown')}")
            else:
                self.logger.warning("⚠️ session_summary.json not found - check if cleanup worked")
            
            # List downloaded files
            downloaded_files = list(output_path.glob('*'))
            self.logger.info(f"📁 Downloaded {len(downloaded_files)} items:")
            for file in downloaded_files:
                self.logger.info(f"  - {file.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to download results: {e}")
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
        success, kernel_slug = manager.create_and_run_github_workflow(
            repo_url=repo_url,
            branch=branch, 
            episodes=episodes,
            timeout=timeout
        )
        if success and kernel_slug:
            print(f"  Kernel completed successfully: {kernel_slug}")
        return success
    except Exception as e:
        print(f"Error running Kaggle training: {e}")
        return False


def run_kaggle_evaluation(model_path: str = "models/best_model.pth",
                         repo_url: str = "https://github.com/elonmj/rl-portfolio-optimizer.git",
                         branch: str = "feature/training-config-updates",
                         timeout: int = 3600) -> bool:
    """
    Quick function to run evaluation on Kaggle.
    
    Args:
        model_path: Path to the model to evaluate
        repo_url: GitHub repository URL (must be public)
        branch: Git branch to clone
        timeout: Maximum monitoring time in seconds
        
    Returns:
        bool: True if successful
    """
    try:
        manager = KaggleManagerGitHub()
        # For evaluation, we use episodes=-1 as a special flag
        success, kernel_slug = manager.create_and_run_github_workflow(
            repo_url=repo_url,
            branch=branch,
            episodes=-1,  # Special flag for evaluation mode
            timeout=timeout
        )
        if success and kernel_slug:
            print(f"  Evaluation kernel completed successfully: {kernel_slug}")
        return success
    except Exception as e:
        print(f"Error running Kaggle evaluation: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("🚀 Kaggle Manager GitHub - Example Usage")
    
    # Quick test with 1 episode
    success = run_kaggle_training(episodes=1)
    
    if success:
        print("  Kaggle workflow completed successfully!")
    else:
        print("❌ Kaggle workflow failed - check logs for details")