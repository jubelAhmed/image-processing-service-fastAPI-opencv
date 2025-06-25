"""
Rich logging configuration and utility functions.
"""

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.traceback import install
import logging
import sys
import time

# Install rich traceback handler
install()

# Create rich console
console = Console()

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)

logger = logging.getLogger("facial_api")

def log_startup_banner(app_name, version):
    """Display a startup banner in the console."""
    console.print(Panel.fit(f"[bold blue]{app_name}[/bold blue] [green]v{version}[/green]", 
                           subtitle="Facial Contour Masking API", 
                           border_style="green"))

def log_request(request, data=None):
    """Log an incoming API request."""
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    endpoint = request.url.path
    logger.info(f"Request: [bold]{method.upper()}[/bold] {endpoint} from [italic]{client_ip}[/italic]")
    if data:
        # Log only essential data, not the entire request body
        log_data = {}
        if hasattr(data, "job_id") and data.job_id:
            log_data["job_id"] = data.job_id
        if hasattr(data, "options") and data.options:
            log_data["options"] = data.options
        logger.debug(f"Request data: {log_data}")

def log_response(request, response_data):
    """Log an API response."""
    method = request.method
    endpoint = request.url.path
    status_code = 200  # Default for successful response
    color = "green" if status_code < 400 else "red"
    logger.info(f"Response: [bold]{method.upper()}[/bold] {endpoint} → [bold {color}]{status_code}[/bold {color}]")

def log_job_status(job_id, status, error=None):
    """Log a job status update."""
    color_map = {
        "queued": "yellow",
        "pending": "yellow",
        "processing": "blue",
        "completed": "green",
        "failed": "red"
    }
    color = color_map.get(status, "white")
    error_str = f" - Error: {error}" if error else ""
    logger.info(f"Job [bold]{job_id}[/bold]: [bold {color}]{status}[/bold {color}]{error_str}")

def log_processing_step(step_name, success=True):
    """Log a processing step."""
    status = "[bold green]✓[/bold green]" if success else "[bold red]✗[/bold red]"
    logger.info(f"Processing step: [bold]{step_name}[/bold] {status}")

def log_error(error_message, exception=None):
    """Log an error with optional exception details."""
    logger.error(f"[bold red]ERROR:[/bold red] {error_message}")
    if exception:
        logger.exception(exception)


def log_job_table(jobs):
    """Display a table of jobs."""
    table = Table(title="Job Status")
    table.add_column("Job ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Created At", style="yellow")
    table.add_column("Updated At", style="yellow")
    
    for job_id, job in jobs.items():
        status_style = {
            "pending": "yellow",
            "processing": "blue",
            "completed": "green",
            "failed": "red"
        }.get(job.status, "white")
        
        table.add_row(
            job_id,
            f"[{status_style}]{job.status}[/{status_style}]",
            job.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            job.updated_at.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    console.print(table)
