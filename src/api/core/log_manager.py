"""Logging system for the application."""

import logging
import socket
import socketserver
import struct
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Final, ClassVar

from logging.handlers import RotatingFileHandler, SocketHandler
from rich.logging import RichHandler

from .exceptions import NetworkError, ConnectionError
from ..utils import get_repo_root

# Define a directory for log files and ensure it exists
LOGS_DIR: Path = Path(get_repo_root()) / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Constants for configuration
DEFAULT_PORT: Final[int] = 9020
BUFFER_SIZE: Final[int] = 100
MAX_LOG_SIZE: Final[int] = 10**6  # 1MB
BACKUP_COUNT: Final[int] = 5
SOCKET_TIMEOUT: Final[float] = 1.0


class DeviceType(Enum):
    """Enumeration for device types in the system."""

    SERVER = auto()
    PARTICIPANT = auto()


@dataclass
class LogConfig:
    """Configuration settings for the logging system."""

    level: int
    default_file: Path
    model_file: Optional[Path] = None
    enable_console: bool = True
    enable_file: bool = True
    enable_rich_tracebacks: bool = True
    console_format: str = "%(message)s"
    file_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class LoggingTheme:
    """Theme settings for colorized logging output."""

    # Device type colors
    server_color: str = "cyan"
    participant_color: str = "green"
    unknown_color: str = "white"

    # Alert and status colors
    error_style: str = "bold red"
    warning_style: str = "yellow"
    success_style: str = "bold green"
    info_style: str = "blue"

    # Keywords to highlight in log messages
    alert_keywords: List[str] = field(
        default_factory=lambda: ["timed out", "error", "failed", "exception", "warning"]
    )
    success_keywords: List[str] = field(
        default_factory=lambda: ["success", "completed", "connected"]
    )

    def get_device_color(self, device_type: Optional[DeviceType]) -> str:
        """Get the appropriate color for a device type."""
        if not device_type:
            return self.unknown_color

        return {
            DeviceType.SERVER: self.server_color,
            DeviceType.PARTICIPANT: self.participant_color,
        }.get(device_type, self.unknown_color)


class LoggingContext:
    """Thread-safe context manager for storing logging context."""

    _device: ClassVar[Optional[DeviceType]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _theme: ClassVar[LoggingTheme] = LoggingTheme()

    @classmethod
    def set_device(cls, device: DeviceType) -> None:
        """Set the current device type in a thread-safe manner."""
        with cls._lock:
            cls._device = device

    @classmethod
    def get_device(cls) -> Optional[DeviceType]:
        """Retrieve the current device type in a thread-safe manner."""
        with cls._lock:
            return cls._device

    @classmethod
    def set_theme(cls, theme: LoggingTheme) -> None:
        """Set the logging theme."""
        with cls._lock:
            cls._theme = theme

    @classmethod
    def get_theme(cls) -> LoggingTheme:
        """Retrieve the current logging theme."""
        with cls._lock:
            return cls._theme


class ColorByDeviceFormatter(logging.Formatter):
    """Custom formatter that adds device-specific color coding to log messages."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors and highlights based on device type and keywords."""
        # Format the basic message
        message = super().format(record)

        # Extract just the message part if it contains metadata
        if " - " in message:
            message = message.split(" - ", 1)[-1]

        # Get device information and theme
        device = LoggingContext.get_device()
        theme = LoggingContext.get_theme()
        device_str = device.name if device else "UNKNOWN"
        color = theme.get_device_color(device)

        # Highlight alert keywords
        for keyword in theme.alert_keywords:
            if keyword.lower() in message.lower():
                message = message.replace(
                    keyword, f"[{theme.error_style}]{keyword}[/{theme.error_style}]"
                )

        # Highlight success keywords
        for keyword in theme.success_keywords:
            if keyword.lower() in message.lower():
                message = message.replace(
                    keyword, f"[{theme.success_style}]{keyword}[/{theme.success_style}]"
                )

        # Format with device type prefix
        return f"[{color}]{device_str}[/{color}]: {message}"


class BufferedSocketHandler(SocketHandler):
    """A buffered socket handler for transmitting log messages over the network."""

    def __init__(
        self,
        host: str,
        port: int,
        buffer_size: int = BUFFER_SIZE,
        timeout: float = SOCKET_TIMEOUT,
    ):
        """Initialize the buffered socket handler."""
        super().__init__(host, port)
        self.buffer: List[str] = []
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.lock = threading.Lock()
        self.sock: Optional[socket.socket] = None
        self.connection_error = False

    def createSocket(self) -> None:
        """Create a non-blocking socket connection for log transmission."""
        if self.connection_error:
            return

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.host, self.port))
            self.sock.setblocking(False)
        except Exception as e:
            self.sock = None
            self.connection_error = True
            raise ConnectionError(f"Socket creation failed: {e}") from e

    def emit(self, record: logging.LogRecord) -> None:
        """Buffer log record and flush when buffer reaches threshold."""
        if self.connection_error:
            return

        try:
            if not self.sock:
                self.createSocket()

            with self.lock:
                self.buffer.append(self.format(record))
                if len(self.buffer) >= self.buffer_size:
                    self.flush()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        """Flush buffered logs by sending each message over the socket."""
        if not self.buffer or self.connection_error:
            return

        with self.lock:
            try:
                for msg in self.buffer:
                    self._send_log(msg)
            finally:
                self.buffer.clear()

    def _send_log(self, msg: str) -> None:
        """Send a single log message with length-prefixed framing for proper message boundaries."""
        if not self.sock or self.connection_error:
            return

        try:
            msg_bytes = msg.encode("utf-8")
            # Prepend the message length as a 4-byte big-endian integer
            self.sock.sendall(struct.pack(">L", len(msg_bytes)) + msg_bytes)
        except BlockingIOError:
            # If non-blocking send fails, re-buffer the message
            self.buffer.append(msg)
        except Exception:
            self.connection_error = True
            raise

    def close(self) -> None:
        """Flush any remaining messages and close the socket."""
        with self.lock:
            self.flush()
            if self.sock:
                self.sock.close()
            super().close()


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """Handles incoming log records from the network."""

    def handle(self) -> None:
        """Process incoming log messages using length-prefixed framing protocol."""
        logger = logging.getLogger("split_computing_logger")

        while True:
            try:
                # Read the first 4 bytes to determine message length
                chunk = self.connection.recv(4)
                if len(chunk) < 4:
                    break

                # Unpack the length prefix
                slen = struct.unpack(">L", chunk)[0]

                # Read the actual log message based on the length
                chunk = self.connection.recv(slen)
                while len(chunk) < slen:
                    chunk += self.connection.recv(slen - len(chunk))

                # Log the received message
                logger.info(chunk.decode("utf-8"))
            except Exception:
                break


class DaemonThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """A threaded TCP server for centralized log handling."""

    allow_reuse_address = True
    daemon_threads = True


def setup_logger(
    is_server: bool = False,
    device: Optional[DeviceType] = None,
    config: Optional[Dict[str, Any]] = None,
    theme: Optional[LoggingTheme] = None,
) -> logging.Logger:
    """Configure and return the application logger."""
    logger = logging.getLogger("split_computing_logger")

    # Return the existing logger if it's already configured
    if logger.hasHandlers():
        return logger

    # Parse configuration and set up the logger
    log_config = _parse_log_config(config)

    # Set theme if provided
    if theme:
        LoggingContext.set_theme(theme)

    # Configure the logger with handlers and formatters
    _configure_logger(logger, log_config, is_server, device)

    return logger


def _parse_log_config(config: Optional[Dict[str, Any]]) -> LogConfig:
    """Extract logging configuration from a dictionary with sensible defaults."""
    level = logging.INFO
    default_file = LOGS_DIR / "app.log"
    model_file = None
    enable_console = True
    enable_file = True
    enable_rich_tracebacks = True

    if config:
        if "logging" in config:
            logging_config = config["logging"]
            # Parse log level
            level_name = logging_config.get("log_level", "INFO").upper()
            level = getattr(logging, level_name)

            # Parse file paths
            default_file = Path(logging_config.get("log_file", default_file))

            # Parse feature flags
            enable_console = logging_config.get("enable_console", enable_console)
            enable_file = logging_config.get("enable_file", enable_file)
            enable_rich_tracebacks = logging_config.get(
                "enable_rich_tracebacks", enable_rich_tracebacks
            )

        if "model" in config and config["model"].get("log_file"):
            model_file = Path(config["model"]["log_file"])

    return LogConfig(
        level=level,
        default_file=default_file,
        model_file=model_file,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_rich_tracebacks=enable_rich_tracebacks,
    )


def _configure_logger(
    logger: logging.Logger,
    config: LogConfig,
    is_server: bool,
    device: Optional[DeviceType],
) -> None:
    """Configure the logger with appropriate handlers and formatters."""
    # Set the base log level
    logger.setLevel(config.level)

    # Create formatters
    file_formatter = logging.Formatter(
        config.file_format,
        datefmt=config.date_format,
    )

    # Initialize handlers
    handlers = []

    # Add file handlers if enabled
    if config.enable_file:
        # Main application log
        handlers.append(
            RotatingFileHandler(
                config.default_file, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
            )
        )

        # Model-specific log if configured
        if config.model_file:
            handlers.append(
                RotatingFileHandler(
                    config.model_file, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
                )
            )

    # Add console handler if enabled
    if config.enable_console:
        rich_handler = RichHandler(
            rich_tracebacks=config.enable_rich_tracebacks,
            show_time=True,
            show_level=True,
            markup=True,
            log_time_format="[%Y-%m-%d %H:%M:%S]",
        )
        handlers.append(rich_handler)

    # Configure and add handlers to logger
    for handler in handlers:
        handler.setLevel(config.level)

        # Apply appropriate formatter based on handler type
        if isinstance(handler, RotatingFileHandler):
            handler.setFormatter(file_formatter)
        else:
            handler.setFormatter(ColorByDeviceFormatter(config.console_format))

        logger.addHandler(handler)

    # Set the device type in the logging context
    if device:
        LoggingContext.set_device(device)
    else:
        LoggingContext.set_device(
            DeviceType.SERVER if is_server else DeviceType.PARTICIPANT
        )


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            # Try to bind to the port
            s.bind(("", port))
            # If we get here, the port is free
            return False
        except OSError:
            # If we get an error, the port is in use
            return True


def find_available_port(start_port: int = DEFAULT_PORT, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    current_port = start_port
    for _ in range(max_attempts):
        if not is_port_in_use(current_port):
            return current_port
        current_port += 1
    # If we can't find a free port, return the original port and let the caller handle the error
    return start_port


# Track the active logging server
_active_logging_server = None

def get_logging_server() -> Optional[DaemonThreadingTCPServer]:
    """Get the active logging server instance if one exists."""
    return _active_logging_server

def start_logging_server(
    port: int = DEFAULT_PORT,
    device: Optional[DeviceType] = None,
    config: Optional[Dict[str, Any]] = None,
    find_free_port: bool = True,
    disable_logging_server: bool = False,
) -> Optional[DaemonThreadingTCPServer]:
    """Start a TCP server for centralized logging.

    Args:
        port: Port to use for the logging server
        device: Device type for logger context
        config: Configuration dictionary
        find_free_port: If True, will try to find an available port if the specified one is in use
        disable_logging_server: If True, will not start a logging server at all (useful for worker processes)

    Returns:
        The server object or None if disabled or if already running
    """
    global _active_logging_server
    
    # Set up logger first
    logger = setup_logger(is_server=True, device=device, config=config)

    # If logging server is disabled, return early
    if disable_logging_server:
        logger.debug("Logging server disabled by configuration")
        return None

    # Check if we're in a worker process (multiprocessing)
    if threading.current_thread().name != "MainThread":
        logger.debug("Skipping logging server in worker thread")
        return None

    # Check if the port is already in use
    if is_port_in_use(port):
        if find_free_port:
            # Try to find an available port
            original_port = port
            port = find_available_port(port)
            if port != original_port:
                logger.info(f"Port {original_port} in use, using port {port} instead")
            else:
                logger.warning(f"Could not find an available port, will try to use {port}")
        else:
            # Port is in use and we're not looking for a new one
            logger.info(f"Logging server already running on port {port}, not starting a new one")
            return None

    try:
        # Create and start the server
        server = DaemonThreadingTCPServer(("", port), LogRecordStreamHandler)

        # Start server in a daemon thread
        server_thread = threading.Thread(
            target=server.serve_forever, name="LoggingServerThread", daemon=True
        )
        server_thread.start()

        logger.info(f"Logging server started on port {port}")
        _active_logging_server = server
        return server

    except OSError as e:
        error_msg = f"Failed to start logging server: {e}"
        logger.error(error_msg)
        raise NetworkError(error_msg) from e


def shutdown_logging_server(server: DaemonThreadingTCPServer) -> None:
    """Shutdown the logging server gracefully."""
    global _active_logging_server
    
    if server:
        logger = logging.getLogger("split_computing_logger")
        logger.info("Shutting down logging server")
        server.shutdown()
        server.server_close()
        
        # Clear the active logging server reference
        if _active_logging_server == server:
            _active_logging_server = None


def get_logger() -> logging.Logger:
    """Get the pre-configured application logger."""
    return logging.getLogger("split_computing_logger")
