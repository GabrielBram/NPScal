from scalapack4py import ScaLAPACK4py
from ctypes import CDLL, RTLD_GLOBAL
from scalapack4py.blacsdesc import blacs_desc
from abc import ABC, abstractmethod

class GeneralRegister():

    def __init__(self):
        self._REGISTER = {}

    def set_register(self, key, val):
        self._REGISTER[key] = val

    def get_register(self, key):
        return self._REGISTER[key]

    def check_register(self, key):
        return key in self._REGISTER.keys()

    def clear_register(self):
        for key in list(self._REGISTER.keys()):
            self.unset_register(key)
    
    @abstractmethod
    def unset_register(self, key):
        pass

class DescrRegister(GeneralRegister):

    def __init__(self):
        super().__init__()

    def unset_register(self, key):
        del self._REGISTER[key]

class ContextRegister(GeneralRegister):

    def __init__(self):
        super().__init__()

    def unset_register(self, key):
        self._REGISTER[key].close_context()
        del self._REGISTER[key]
        
class BLACSContextManager():
    """
    Given that ab application codebase may be working with
    several BLACS contexts, it will be useful to track them
    using a registry.
    """
    def __init__(self, context_tag, nproc, mproc, lib):

        if CTXT_Register.check_register(context_tag):
            raise Exception(f"context_tag [{context_tag}] already exists. Please specify a new context tag")

        if isinstance(lib, str):
            self.lib = ScaLAPACK4py(CDLL(lib, mode=RTLD_GLOBAL))
        else:
            self.lib = lib

        self.MP, self.NP = nproc, mproc
        self.ctxt = self.lib.make_blacs_context(self.lib.get_default_system_context(), nproc, mproc)
        self.tag = context_tag
        
        # Finally, add the BLACS Context to the Register
        CTXT_Register.set_register(context_tag, self)

    def close_context(self):
        self.lib.close_blacs_context(self.ctxt)

    def __repr__(self):
        return str(self.ctxt)

class BLACSDESCRManager(blacs_desc):
    """
    Given that ab application codebase may be working with
    several BLACS contexts, it will be useful to track them
    using a registry.
    """
    def __init__(self, context_tag, descr_tag, lib, m=0, n=0, mb=1, nb=1, rsrc=0, csrc=0, lld=None, buf=None):

        if not(CTXT_Register.check_register(context_tag)):
            raise Exception(f"context_tag [{context_tag}] does not exist. Please specify an existing context.")

        if DESCR_Register.check_register(descr_tag):
            raise Exception(f"descr_tag [{descr_tag}] already exists. Please use another descr_tag.")

        if isinstance(lib, str):
            self.lib = ScaLAPACK4py(CDLL(lib, mode=RTLD_GLOBAL))
        else:
            self.lib = lib

        self.tag = descr_tag
        ctxt = CTXT_Register.get_register(context_tag).ctxt

        super().__init__(self.lib, ctxt, m, n, mb, nb, rsrc, csrc, lld, buf)

        # Finally, add the BLACS Context to the Register
        DESCR_Register.set_register(descr_tag, self)

CTXT_Register = ContextRegister()
DESCR_Register = DescrRegister()
