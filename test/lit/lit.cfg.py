import os
import lit.formats

config.name = "LLVMDSDL"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir", ".txt"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.dirname(__file__)

config.substitutions.append(("%dsdlc", config.dsdlc))
config.substitutions.append(("%dsdl-opt", config.dsdl_opt))
