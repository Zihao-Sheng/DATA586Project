from __future__ import annotations

import codecs
import subprocess
import sys
from pathlib import Path


def run_and_stream(command: list[str], cwd: Path) -> None:
    try:
        from IPython.display import display
    except Exception:
        display = None

    status_handle = display("", display_id=True) if display is not None else None
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert process.stdout is not None

    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    buffer = ""
    live_line = ""

    def emit_live_line(text: str) -> None:
        if status_handle is not None:
            status_handle.update(text)
            return
        sys.stdout.write("\r" + text)
        sys.stdout.flush()

    def emit_line(text: str) -> None:
        nonlocal live_line
        if status_handle is not None and live_line:
            status_handle.update("")
        print(text)
        live_line = ""

    while True:
        chunk = process.stdout.read(1024)
        if not chunk:
            tail = decoder.decode(b"", final=True)
            if tail:
                buffer += tail
            break

        text = decoder.decode(chunk)
        if text:
            buffer += text

        while buffer:
            newline_index = buffer.find("\n")
            carriage_index = buffer.find("\r")
            cut_positions = [index for index in (newline_index, carriage_index) if index != -1]
            if not cut_positions:
                live_line = buffer
                emit_live_line(live_line)
                buffer = ""
                break

            cut_index = min(cut_positions)
            delimiter = buffer[cut_index]
            segment = buffer[:cut_index]
            buffer = buffer[cut_index + 1 :]

            if delimiter == "\r":
                live_line = segment
                emit_live_line(live_line)
                continue

            if live_line and not segment:
                segment = live_line
            emit_line(segment)

    if buffer:
        emit_line(buffer)
    if live_line and not buffer:
        emit_line(live_line)
    if status_handle is not None:
        status_handle.update("")

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
