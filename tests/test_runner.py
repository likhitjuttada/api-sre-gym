import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import importlib

from gym.registry import BUG_REGISTRY
from gym.runner import GymRunner


def main() -> None:
    results: list[tuple[str, bool, str]] = []
    for bug, module_path in BUG_REGISTRY.items():
        runner = GymRunner(startup_timeout=8.0)
        try:
            base = runner.start(module_path)
            mod = importlib.import_module(module_path)
            gt = getattr(mod, 'GROUND_TRUTH', None)
            assert gt is not None, f'{module_path} missing GROUND_TRUTH'
            assert gt['bug_type'] == bug, f'{module_path} bug_type mismatch'
            h = httpx.get(f'{base}/healthz', timeout=2.0)
            assert h.status_code == 200
            o = httpx.get(f'{base}/openapi.json', timeout=2.0)
            assert o.status_code == 200
            results.append((bug, True, base))
        except Exception as exc:
            results.append((bug, False, str(exc)))
        finally:
            runner.shutdown()

    ok = sum(1 for _, passed, _ in results if passed)
    for bug, passed, info in results:
        print(('PASS' if passed else 'FAIL'), bug, info)
    print(f'\n{ok}/{len(results)} apps healthy')
    if ok != len(results):
        sys.exit(1)


if __name__ == '__main__':
    main()
