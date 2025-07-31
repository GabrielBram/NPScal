from scalapack4py.blacsdesc import blacs_desc

class GeneralRegister():

    def __init__(self):

        self._REGISTER = {}

    def set_register(self, key, val):
        self._REGISTER[key] = val

    def get_register(self, key):
        return self._REGISTER[key]

    def unset_register(self, key):
        self._REGISTER[key].pop()

    def check_register(self, key):
        return key in self._REGISTER.keys()

    def clean_dead_contexts(self):
        for val in self._REGISTER.values():
            print(val)
            
class BLACSContextManager():
    """
    Given that ab application codebase may be working with
    several BLACS contexts, it will be useful to track them
    using a registry.
    """
    def __init__(self, context_tag, nproc, mproc, lib):

        if CTXT_Register.check_register(context_tag):
            raise Exception(f"context_tag [{context_tag}] already exists. Please specify a new context tag")

        self.lib = lib
        self.MP, self.NP = nproc, mproc
        self.ctxt = self.lib.make_blacs_context(self.lib.get_default_system_context(), nproc, mproc)
        self.tag = context_tag
        
        # Finally, add the BLACS Context to the Register
        CTXT_Register.set_register(context_tag, self)

    def del_context(self):
        return None

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

        self.tag = descr_tag
        ctxt = CTXT_Register.get_register(context_tag).ctxt

        super().__init__(lib, ctxt, m, n, mb, nb, rsrc, csrc, lld, buf)

        # Finally, add the BLACS Context to the Register
        DESCR_Register.set_register(descr_tag, self)

    #TODO: KILL REGISTERS AND CONTEXT        
    def del_descr(self):
        return None

CTXT_Register = GeneralRegister()
DESCR_Register = GeneralRegister()
