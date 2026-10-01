"""Shared setup for the whole suite, and the reason rfixtures imports from every subfolder.

pytest's default prepend import mode puts this file's folder, tests/, at the front of sys.path
when it loads it, before it imports any test module. So test modules in tests/nuggets,
tests/wcluster and the other subfolders import rfixtures and the R fixtures it loads, whatever
order pytest collects them in and whichever single file is run.
"""
