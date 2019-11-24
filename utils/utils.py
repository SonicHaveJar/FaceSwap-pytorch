def compare(arg1, arg2, opt):
    if opt == 'smaller':
        return arg1 if arg1 < arg2 else arg2
    elif opt == 'bigger':
        return arg1 if arg1 > arg2 else arg2

#Forward propagation with the autoencoders combined
def forward(x, A, B, direction='A2B'):
    if direction == 'A2B':
        out = A.encoder(x)
        out = B.decoder(out)
        return out
    else:        
        out = B.encoder(x)
        out = A.decoder(out)
        return out