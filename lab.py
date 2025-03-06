"""
6.101 Lab:
LISP Interpreter Part 2
"""

#!/usr/bin/env python3
import sys

sys.setrecursionlimit(20_000)

# KEEP THE ABOVE LINES INTACT, BUT REPLACE THIS COMMENT WITH YOUR lab.py FROM
# THE PREVIOUS LAB, WHICH SHOULD BE THE STARTING POINT FOR THIS LAB.

# NO ADDITIONAL IMPORTS!

#############################
# Scheme-related Exceptions #
#############################


class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression
    """
    lines = source.split("\n")  # separate lines
    raw_lines = [line.split(";")[0] for line in lines]  # remove comments
    tokens = []
    for line in raw_lines:
        # find all the tokens in each line
        cur = ""
        for char in line + " ":
            if char in " ()":
                tokens.append(cur)
                tokens.append(char.strip())
                cur = ""
            else:
                cur += char
    return [tok for tok in tokens if tok]


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """

    def parse_helper(idx):
        if tokens[idx] == "(":
            final_expr = []
            idx += 1
            while idx < len(tokens) and tokens[idx] != ")":
                expr, idx = parse_helper(idx)
                final_expr.append(expr)
            return final_expr, idx + 1
        elif tokens[idx] == ")":
            raise SchemeSyntaxError("unmatched )")
        else:
            return number_or_symbol(tokens[idx]), idx + 1

    out_expr, out_idx = parse_helper(0)
    if out_idx != len(tokens):
        raise SchemeSyntaxError("incomplete expr")
    return out_expr


######################
# Built-in Functions #
######################


def mul(args):
    """
    Given a list of evaluated args, return the
    result of multiplying all the numbers together
    """
    result = 1
    for num in args:
        result *= num
    return result


def div(args):
    """
    Given a list of evaluated args, return the
    result of dividing all the numbers from left to right
    """
    result = args[0]
    for num in args[1:]:
        result /= num
    return result


# good way to avoid repetition in the comparisons!
_compare_funcs = {
    ">": lambda x, y: x > y,
    ">=": lambda x, y: x >= y,
    "equal?": lambda x, y: x == y,
    "<=": lambda x, y: x <= y,
    "<": lambda x, y: x < y,
}


def compare_args(args, compare_str):
    """
    Given a list of evaluated args, and a comparison
    string, return whether all consecutive pairs of
    args satisfy the comparison

    Ex: compare_args([1,2,3], "<") is equivalent to
    1 < 2 and 2 < 3
    """
    compare_func = _compare_funcs[compare_str]
    for i in range(1, len(args)):
        if not compare_func(args[i - 1], args[i]):
            return False
    return True


def not_func(args):
    """
    Given a single evaluated arg (boolean) return the
    opposite boolean.
    not_func([True]) -> False
    not_func([False]) -> True
    """
    if len(args) != 1:
        raise SchemeEvaluationError(f"not expects 1 arg but got {args}")
    return not args[0]


def get_car(args):
    """
    Given args [PairObject] return the car of the Pair.
    Raises error if number of args does not match.
    """
    if len(args) != 1:
        raise SchemeEvaluationError(f"car expects 1 arg but got {args}")
    return args[0].car


def get_cdr(args):
    """
    Given args [PairObject] return the cdr of the Pair.
    Raises error if number of args does not match.
    """
    if len(args) != 1:
        raise SchemeEvaluationError(f"cdr expects 1 arg but got {args}")
    return args[0].cdr


def make_list(args):
    """
    Given a list of 0 or more args, make and return a new PairObject
    representing the linked list containing all the args.
    """
    if not args:
        return empty_list
    else:
        return Pair(args[0], make_list(args[1:]))


def is_list(args):
    """
    Given args [evaluted_arg], determine if arg is a
    linked list. Returns a boolean.
    """
    if len(args) != 1:
        raise SchemeEvaluationError(f"list? bad {args=}")
    llist = args[0]

    def helper(llist):
        if llist is empty_list:
            return True
        return isinstance(llist, Pair) and helper(llist.cdr)

    return helper(llist)


def append_lists(args):
    """
    Given a list of linked lists, return a new linked list
    that is the result of appending all the lists together.
    """
    if not args:
        return empty_list
    elif args[0] == empty_list:
        return append_lists(args[1:])
    return Pair(args[0].car, append_lists([args[0].cdr] + args[1:]))


def length_list(args):
    """
    Given [llist] calculate the length of the linked list.
    """
    if len(args) != 1:
        raise SchemeEvaluationError(f"length bad args {args=}")
    llist = args[0]

    def helper(llist):
        if llist is empty_list:
            return 0
        return 1 + helper(llist.cdr)

    return helper(llist)


def index_list(args):
    """
    Given [llist, index] return the object at that
    index in the linked list.
    """
    if len(args) != 2:
        raise SchemeEvaluationError(f"list-ref bad {args=}")
    llist = args[0]
    index = args[1]

    def helper(llist, index):
        if index == 0:
            return llist.car
        return helper(llist.cdr, index - 1)

    return helper(llist, index)


def display(args):
    print(*args)
    return ""


scheme_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": mul,
    "/": div,
    # comparisons
    "#t": True,
    "#f": False,
    "equal?": lambda args: compare_args(args, "equal?"),
    ">": lambda args: compare_args(args, ">"),
    ">=": lambda args: compare_args(args, ">="),
    "<": lambda args: compare_args(args, "<"),
    "<=": lambda args: compare_args(args, "<="),
    "not": not_func,
    # Linked list methods
    "cons": lambda args: Pair(*args),
    "car": get_car,
    "cdr": get_cdr,
    "list": make_list,
    "list?": is_list,
    "append": append_lists,
    "length": length_list,
    "list-ref": index_list,
    "begin": lambda args: args[-1],
    # extras:
    "display": display,
}

empty_list = None  # () rep


##############
# Evaluation #
##############


def evaluate_file(fname, frame=None):
    with open(fname, "r") as file:
        expr = file.read()

    return evaluate(parse(tokenize(expr)), frame)


def evaluate(tree, frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    if frame is None:
        frame = make_initial_frame()

    if isinstance(tree, str):
        return frame.get(tree)
    elif isinstance(tree, (int, float)):
        return tree
    elif isinstance(tree, list):
        if not tree:
            return empty_list  # () rep is None
        elif tree[0] == "define":
            if isinstance(tree[1], list):  # short func definition
                # [define, [square, x], [*, x, x]]
                name = tree[1][0]
                params = tree[1][1:]
                body = tree[2]
                val = Lambda(params, body, frame)
                frame.variables[name] = val
                return val
            else:  # regular variable definition
                # [define, x, [+, 3, 2]]
                val = evaluate(tree[2], frame)
                frame.variables[tree[1]] = val
                return val
        elif tree[0] == "lambda":  # create function object
            # [lambda, [x, y], [+, x, y]]
            params = tree[1]
            body = tree[2]
            return Lambda(params, body, frame)
        elif tree[0] == "if":
            # (if PRED TRUE_EXP FALSE_EXP)
            if evaluate(tree[1], frame):
                return evaluate(tree[2], frame)
            else:
                return evaluate(tree[3], frame)
        elif tree[0] == "and":
            # (and bool1 bool2 bool3...)
            return all(evaluate(x, frame) for x in tree[1:])
        elif tree[0] == "or":
            # (or bool1 bool2 bool3...)
            return any(evaluate(x, frame) for x in tree[1:])
        elif tree[0] == "del":
            # [del var]
            return frame.remove(tree[1])
        elif tree[0] == "let":
            # (let ((VAR1 VAL1) (VAR2 VAL2) (VAR3 VAL3) ...) BODY)
            new_frame = Frame()
            new_frame.parent = frame
            for var_val in tree[1]:
                var, val = var_val
                new_frame.variables[var] = evaluate(val, frame)
            return evaluate(tree[2], new_frame)
        elif tree[0] == "set!":
            # [set! var expr]
            var = tree[1]
            val = evaluate(tree[2], frame)
            frame.setbang(var, val)
            return val

        func = evaluate(tree[0], frame)
        args = []
        for subtree in tree[1:]:
            args.append(evaluate(subtree, frame))

        try:
            if isinstance(func, Lambda):
                return func.call(args)
            elif callable(func):
                return func(args)
        except SchemeError:
            raise
        except Exception as e:
            raise SchemeEvaluationError(f"{tree} cannot be evaluated because {e}")

    raise SchemeEvaluationError(f"Invalid tree {tree=}")


def make_initial_frame():
    frame = Frame()
    frame.parent = builtins_frame
    return frame


class Frame:
    def __init__(self) -> None:
        self.parent = None
        self.variables = {}

    def __repr__(self) -> str:
        return f"vars:{self.variables}, parent:{self.parent}"

    def get(self, name):
        if name in self.variables:
            return self.variables[name]
        elif self.parent is None:
            raise SchemeNameError(f"{name} does not exist in frame")
        return self.parent.get(name)

    def remove(self, name):
        if name in self.variables:
            return self.variables.pop(name)

        raise SchemeNameError(f"{name} does not exist in current frame")

    def setbang(self, name, val):
        if name in self.variables:
            self.variables[name] = val
        elif self.parent is not None:
            self.parent.setbang(name, val)
        else:
            raise SchemeNameError(f"{name} does not exist in current frame")


builtins_frame = Frame()
builtins_frame.variables = scheme_builtins.copy()


class Lambda:
    def __init__(self, params, body, parent) -> None:
        if not isinstance(params, list):
            raise SchemeEvaluationError("Bad param names!")
        self.params = params
        self.body = body
        self.parent_frame = parent

    def __repr__(self) -> str:
        return f"Lambda({self.params}, {self.body}, frame)"

    def call(self, args):
        if len(args) != len(self.params):
            raise SchemeEvaluationError

        frame = Frame()
        frame.parent = self.parent_frame
        for i in range(len(args)):
            var = self.params[i]
            frame.variables[var] = args[i]

        return evaluate(self.body, frame)


class Pair:
    def __init__(self, car, cdr) -> None:
        self.car = car
        self.cdr = cdr

    def __repr__(self):
        return f"Pair({self.car!r}, {self.cdr!r})"
        #!r is shorthand for repr(self.car)


if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    import sys

    fnames = sys.argv[1:]
    global_frame = make_initial_frame()
    for fname in fnames:
        print(f"loading {fname}...")
        print(evaluate_file(fname, global_frame))

    import os

    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    import schemerepl

    schemerepl.SchemeREPL(
        sys.modules[__name__], use_frames=True, verbose=False, global_frame=global_frame
    ).cmdloop()

    # print(parse(tokenize("(list? 7)")))