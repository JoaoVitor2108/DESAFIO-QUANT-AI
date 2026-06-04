import sys
from pathlib import Path

# Garante que o root do projeto está no sys.path para imports funcionarem
sys.path.insert(0, str(Path(__file__).parent.parent))
