def compare(arg1, arg2, opt):
    if opt == 'smaller':
        return arg1 if arg1 < arg2 else arg2
    elif opt == 'bigger':
        return arg1 if arg1 > arg2 else arg2