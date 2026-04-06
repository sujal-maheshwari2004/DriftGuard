from driftguard.config import DEFAULT_SETTINGS, DriftGuardSettings
from driftguard.logging_config import configure_logging, get_logger
from driftguard.mcp import create_mcp_server


logger = get_logger(__name__)


# =====================================================
# START SERVER
# =====================================================

def main(settings: DriftGuardSettings | None = None):
    settings = settings or DEFAULT_SETTINGS
    configure_logging(settings.log_level)
    logger.info("Starting DriftGuard MCP server")
    mcp = create_mcp_server(settings=settings)
    mcp.run()


if __name__ == "__main__":
    main()
