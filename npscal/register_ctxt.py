class GridRegister():

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

class BLACSContextWrapper():
    """
    Given that ab application codebase may be working with
    several BLACS contexts, it will be useful to track them
    using a registry.
    """
    def __init__(self, context_tag, mproc, nproc, lib):

        self.sl = lib

        if Register.check_register(context_tag):
            print("CONTEXT EXISTS - SETTING CONTEXT WRAPPER TO SPECIFIED REGISTER")
            self.ctxt = Register.get_register(context_tag)
        else:
            self.ctxt = self.sl.make_blacs_context(self.sl.get_default_system_context(), mproc, nproc)
            Register.set_register(self.ctxt, self)

Register = GridRegister()
