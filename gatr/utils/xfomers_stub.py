# xformers_stub.py
"""
Stub module to replace xformers for Mac compatibility
"""
class __EmptyModule:
    def __getattr__(self, name):
        return None

import sys
sys.modules['xformers'] = __EmptyModule()
sys.modules['xformers.ops'] = __EmptyModule()