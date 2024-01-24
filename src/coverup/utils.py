from pathlib import Path


class TemporaryOverwrite:
    """Context handler that overwrites a file, and restores it upon exit."""
    def __init__(self, file: Path, new_content: str):
        self.file = file
        self.new_content = new_content
        self.backup = file.parent / (file.name + ".bak") if file.exists() else None

    def __enter__(self):
        if self.file.exists():
            self.file.replace(self.backup)

        self.file.write_text(self.new_content)
        self.file.touch()

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.unlink()
        if self.backup:
            self.backup.replace(self.file)
