import subprocess, sys

from rag_system.settings import settings


def main() -> None:
    result = subprocess.run(
        [
            "docker", "compose", "exec", "mongo", "mongosh",
            settings.DATABASE_HOST
        ],
    )
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
