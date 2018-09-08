

# Convert from base ten to any base less than ten with arbitrary precision and fractionals
# Handles fractional values
def dec_to_base(dec, radix, base=8, places=100):

    whole = int(dec)
    whole_accum = []
    while whole:
        whole_accum.append(str(whole % base))
        whole = int(whole / base)

    n = 0
    # Extract point, then un-normalize to integer with saved radix offset
    rem = int(dec % 1 * 10**radix)
    rem_accum = []
    while rem and n < places:
        n += 1
        product = rem * base

        # Extract most significant digit. Following digits are remainders
        lead = product // 10**radix
        rem_accum.append(str(lead))

        # Compute remainder by subtracting most significant digit
        rem = product - lead * 10**radix

    return "".join(reversed(whole_accum)) + '.' + "".join(rem_accum)


print(dec_to_base(.782, base=8, radix=3))
# .6203044672274324773716662132071260101422335136152375747331055034530040611156457065176763554426416254
print(dec_to_base(.782, base=2, radix=3))
# .1100100000110001001001101110100101111000110101001111110111110011101101100100010110100001110010101100


print(dec_to_base(.694, base=8, radix=3))
# .5432477371666213207126010142233513615237574733105503453004061115645706517676355442641625402030446722
print(dec_to_base(.694, base=2, radix=3))
# .1011000110101001111110111110011101101100100010110100001110010101100000010000011000100100110111010010
