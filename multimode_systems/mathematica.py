import sympy as sp

from wolframclient.language import wl, wlexpr
from wolframclient.evaluation import WolframLanguageSession

def create_wolfram_kernel():
    return WolframLanguageSession('/Applications/Mathematica.app/Contents/MacOS/WolframKernel')

def convert_sympy_equ_system_to_mathematica(equ_system):
    math_string = '{'
    for equ in equ_system:
        math_string += sp.mathematica_code(equ) + '==0,'
    math_string = math_string[:-1]  #cut the last comma
    math_string += '}'
    return wlexpr(math_string)

def convert_sympy_variables_to_mathematica(variables):
    math_string = '{'
    for var in variables:
        math_string += sp.mathematica_code(var) + ','
    math_string = math_string[:-1]  #cut the last comma
    math_string += '}'
    return wlexpr(math_string)

def save_conditions_and_variables(conditions, variables, filename, conditions_name='conditions', variables_name='variables'):
    session = create_wolfram_kernel()
    session.evaluate(wlexpr('%s=%s'%(conditions_name, convert_sympy_equ_system_to_mathematica(conditions))))
    session.evaluate(wlexpr('%s=%s'%(variables_name, convert_sympy_variables_to_mathematica(variables))))
    session.evaluate(wlexpr('DumpSave["%s", {%s, %s}]'%(filename, conditions_name, variables_name)))
    session.terminate()