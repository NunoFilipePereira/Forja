"""Ponto de entrada para as versões compiladas (PyInstaller)."""
import multiprocessing

from forja.__main__ import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
