"""Dispatch editing commands: python -m gsedit COMMAND [arguments]."""
import runpy
import sys

from gsedit.runtime import commands


def main():
    registry = commands()
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "--list"}:
        print("Usage: python -m gsedit COMMAND [arguments]\n")
        for group in sorted({module.split('.')[1] for module in registry.values()}):
            print(group + ":")
            for name, module in sorted(registry.items()):
                if module.split('.')[1] == group:
                    print("  " + name)
        return
    name = sys.argv.pop(1).removesuffix('.py')
    if name not in registry:
        raise SystemExit(f"Unknown command: {name}. Use python -m gsedit --list")
    sys.argv[0] = registry[name]
    runpy.run_module(registry[name], run_name="__main__")


if __name__ == "__main__":
    main()
