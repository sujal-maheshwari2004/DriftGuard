from driftguard.config import DriftGuardSettings
from driftguard.logging_config import configure_logging, get_logger
from driftguard.mcp import create_mcp_server
from driftguard.utils.normalization import ensure_available


logger = get_logger(__name__)


# =====================================================
# START SERVER
# =====================================================

def main(settings: DriftGuardSettings | None = None):
    settings = settings or DriftGuardSettings.from_env()
    configure_logging(settings.log_level)
    logger.info(
        "Starting DriftGuard MCP server backend=%s graph=%s",
        settings.storage_backend,
        settings.sqlite_filepath
        if settings.storage_backend == "sqlite"
        else settings.graph_filepath,
    )

    # Fail here rather than on the first register_mistake call.
    ensure_available()

    mcp = create_mcp_server(settings=settings)
    mcp.run()


if __name__ == "__main__":
    main()
