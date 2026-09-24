"""Required-agent workflow for novel planning and drafting."""

from .models import ProjectBrief
from .workflow import NovelWorkflow

__all__ = ["NovelWorkflow", "ProjectBrief"]
__version__ = "0.1.0"

