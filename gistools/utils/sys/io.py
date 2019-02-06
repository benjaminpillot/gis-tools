# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""

# __all__ = []
# __version__ = '0.1'
import ctypes
import datetime
import io
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from tempfile import gettempdir

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2018, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


libc = ctypes.CDLL(None)
c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')


class LogFile:

    entry = 0

    def __init__(self, name="log_file"):
        self.path = os.path.join(gettempdir(), name + ".log")
        self.stream = io.BytesIO()

    def write(self, entry):
        with open(self.path, "a") as log_file:
            log_file.write("[%d] %s: %s\n" % (self.entry, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), entry))
        self.entry += 1

    @contextmanager
    def stdout_redirector(self):
        # The original fd stdout points to. Usually 1 on POSIX systems.
        original_stdout_fd = sys.stdout.fileno()

        def _redirect_stdout(to_fd):
            """Redirect stdout to the given file descriptor."""
            # Flush the C-level buffer stdout
            libc.fflush(c_stdout)
            # Flush and close sys.stdout - also closes the file descriptor (fd)
            sys.stdout.close()
            # Make original_stdout_fd point to the same file as to_fd
            os.dup2(to_fd, original_stdout_fd)
            # Create a new sys.stdout that points to the redirected fd
            sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

        # Save a copy of the original stdout fd in saved_stdout_fd
        saved_stdout_fd = os.dup(original_stdout_fd)
        tfile = tempfile.TemporaryFile(mode='w+b')
        try:
            # Create a temporary file and redirect stdout to it
            _redirect_stdout(tfile.fileno())
            # Yield to caller, then redirect stdout back to the saved fd
            yield
            _redirect_stdout(saved_stdout_fd)
            # Copy contents of temporary file to the given stream
            tfile.flush()
            tfile.seek(0, io.SEEK_SET)
            self.stream.write(tfile.read())
        finally:
            tfile.close()
            os.close(saved_stdout_fd)

        self.write(self.stream.getvalue().decode('utf-8'))

    @contextmanager
    def stderr_redirector(self):
        # The original fd stderr points to. Usually 1 on POSIX systems.
        original_stderr_fd = sys.stderr.fileno()

        def _redirect_stderr(to_fd):
            """Redirect stdout to the given file descriptor."""
            # Flush the C-level buffer stderr
            libc.fflush(c_stderr)
            # Flush and close sys.stderr - also closes the file descriptor (fd)
            sys.stderr.close()
            # Make original_stderr_fd point to the same file as to_fd
            os.dup2(to_fd, original_stderr_fd)
            # Create a new sys.stderr that points to the redirected fd
            sys.stderr = io.TextIOWrapper(os.fdopen(original_stderr_fd, 'wb'))

        # Save a copy of the original stderr fd in saved_stderr_fd
        saved_stderr_fd = os.dup(original_stderr_fd)
        tfile = tempfile.TemporaryFile(mode='w+b')
        try:
            # Create a temporary file and redirect stderr to it
            _redirect_stderr(tfile.fileno())
            # Yield to caller, then redirect stderr back to the saved fd
            yield
            _redirect_stderr(saved_stderr_fd)
            # Copy contents of temporary file to the given stream
            tfile.flush()
            tfile.seek(0, io.SEEK_SET)
            self.stream.write(tfile.read())
        finally:
            tfile.close()
            os.close(saved_stderr_fd)

        self.write(self.stream.getvalue().decode('utf-8'))


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, name="out", log_level=logging.INFO):
        self.path = os.path.join(tempfile.gettempdir(), name + ".log")
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
        logging.basicConfig(level=self.log_level,
                            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
                            filename=self.path,
                            filemode='a')

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
