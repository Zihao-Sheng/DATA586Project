from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "maintenance" / "ensure_packages.py"


class RequirementsWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("DATA586 Requirements Checker")
        self.root.geometry("760x460")

        self.status_var = tk.StringVar(value="Ready")

        self.text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state=tk.DISABLED)
        self.text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 8))

        controls = tk.Frame(self.root)
        controls.pack(fill=tk.X, padx=12, pady=(0, 12))

        self.run_button = tk.Button(controls, text="Check and Install", command=self.start)
        self.run_button.pack(side=tk.LEFT)

        close_button = tk.Button(controls, text="Close", command=self.root.destroy)
        close_button.pack(side=tk.RIGHT)

        status_label = tk.Label(controls, textvariable=self.status_var, anchor="w")
        status_label.pack(side=tk.LEFT, padx=(12, 0))

        self.process: subprocess.Popen[str] | None = None
        self.pending_cli_args: list[str] = []
        self.start()

    @staticmethod
    def detect_installed_torch_variant() -> str | None:
        try:
            version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            return None
        if "+cu128" in version:
            return "cu128"
        if "+cpu" in version:
            return "cpu"
        return None

    @staticmethod
    def has_nvidia_gpu() -> bool:
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return False
        return completed.returncode == 0 and bool(completed.stdout.strip())

    def choose_torch_install_args(self) -> list[str]:
        if not self.has_nvidia_gpu():
            return []

        installed_variant = self.detect_installed_torch_variant()
        if installed_variant == "cu128":
            return []

        if installed_variant == "cpu":
            install_cuda = messagebox.askyesno(
                "PyTorch CUDA Install",
                "NVIDIA GPU detected and CPU-only PyTorch is installed.\n\nInstall the CUDA-enabled cu128 build instead?",
                default=messagebox.YES,
            )
            return ["--torch-variant", "cu128", "--yes"] if install_cuda else []

        install_cuda = messagebox.askyesno(
            "PyTorch CUDA Install",
            "NVIDIA GPU detected.\n\nInstall the CUDA-enabled cu128 build of PyTorch?",
            default=messagebox.YES,
        )
        if install_cuda:
            return ["--torch-variant", "cu128", "--yes"]
        return ["--torch-variant", "cpu", "--yes"]

    def append(self, text: str) -> None:
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, text)
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return

        self.pending_cli_args = self.choose_torch_install_args()

        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.configure(state=tk.DISABLED)
        self.status_var.set("Running...")
        self.run_button.configure(state=tk.DISABLED)

        thread = threading.Thread(target=self._run_process, daemon=True)
        thread.start()

    def _run_process(self) -> None:
        command = [sys.executable, str(SCRIPT_PATH), *self.pending_cli_args]
        self.process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.root.after(0, self.append, line)

        return_code = self.process.wait()
        self.root.after(0, self._finish, return_code)

    def _finish(self, return_code: int) -> None:
        self.run_button.configure(state=tk.NORMAL)
        if return_code == 0:
            self.status_var.set("Finished")
            messagebox.showinfo("Requirements", "Package check finished.")
        else:
            self.status_var.set(f"Failed ({return_code})")
            messagebox.showerror("Requirements", f"Package check failed with exit code {return_code}.")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    RequirementsWindow().run()


if __name__ == "__main__":
    main()
