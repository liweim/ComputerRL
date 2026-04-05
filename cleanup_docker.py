#!/usr/bin/env python3
import argparse
import docker
import logging


def cleanup_osworld_containers(remove_running: bool = False) -> None:
    """Cleanup osworld-related containers, skipping running ones by default."""
    logger = logging.getLogger("cleanup_docker")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)

    client = docker.from_env()
    containers = client.containers.list(all=True, filters={"ancestor": "happysixd/osworld-docker"})

    if not containers:
        logger.info("No happysixd/osworld-docker containers found.")
        return

    removed_count = 0
    skipped_count = 0
    for container in containers:
        try:
            if container.status == "running":
                if remove_running:
                    container.stop(timeout=5)
                    container.remove(force=True)
                    logger.info(f"Stopped and removed running container: {container.name}")
                    removed_count += 1
                else:
                    skipped_count += 1
            else:
                container.remove(force=True)
                logger.info(f"Removed exited container: {container.name}")
                removed_count += 1
        except Exception as e:
            logger.warning(f"Failed to remove container {container.name}: {e}")

    logger.info(f"Cleanup completed. Removed: {removed_count}, Skipped (running): {skipped_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup OSWorld Docker containers.")
    parser.add_argument(
        "--force-running",
        action="store_true",
        help="Also stop and remove currently running happysixd/osworld-docker containers.",
    )
    args = parser.parse_args()
    cleanup_osworld_containers(remove_running=args.force_running)
