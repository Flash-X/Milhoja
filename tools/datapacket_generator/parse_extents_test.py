import milhoja_utility as md

extents = md.parse_extents("cc(NGUARD)", 0, 0, size="FArray4D")
assert extents[0] == "(nxb_ + 2 * nGuard_ * MILHOJA_K1D) * (nyb_ + 2 * nGuard_ * MILHOJA_K2D) * (nzb_ + 2 * nGuard_ * MILHOJA_K3D) * ( ( (0) + (0) + 1 ) ) * sizeof(FArray4D)"
assert extents[1] == '( (0) + (0) + 1 )'
assert extents[2] == 'cc'

extents = md.parse_extents("cc(NGUARD)", 0, 5, size="FArray4D")
assert extents[0] == "(nxb_ + 2 * nGuard_ * MILHOJA_K1D) * (nyb_ + 2 * nGuard_ * MILHOJA_K2D) * (nzb_ + 2 * nGuard_ * MILHOJA_K3D) * ( ( (5) + (0) + 1 ) ) * sizeof(FArray4D)"
assert extents[1] == "( (5) + (0) + 1 )"
assert extents[2] == 'cc'

extents = md.parse_extents("cc(NGUARD)", 0, 0, size="FArray4D")
assert extents[0] == "(nxb_ + 2 * nGuard_ * MILHOJA_K1D) * (nyb_ + 2 * nGuard_ * MILHOJA_K2D) * (nzb_ + 2 * nGuard_ * MILHOJA_K3D) * ( ( (0) + (0) + 1 ) ) * sizeof(FArray4D)"
assert extents[1] == "( (0) + (0) + 1 )"
assert extents[2] == 'cc'

extents = md.parse_extents("fcx(0)", 0, 4, "Real")
assert extents[0] == "((nxb_+1) + 2 * 0) * ((nyb_) + 2 * 0) * ((nzb_) + 2 * 0) * ( ( (4) + (0) + 1 ) ) * sizeof(Real)"
assert extents[1] == '( (4) + (0) + 1 )'
assert extents[2] == 'fcx'

extents = md.parse_extents("fcy(0)", 0, 4, "Real")
assert extents[0] == "((nxb_) + 2 * 0) * ((nyb_+1) + 2 * 0) * ((nzb_) + 2 * 0) * ( ( (4) + (0) + 1 ) ) * sizeof(Real)"
assert extents[1] == "( (4) + (0) + 1 )"
assert extents[2] == 'fcy'

extents = md.parse_extents("fcz(0)", 0, 4, "Real")
assert extents[0] == "((nxb_) + 2 * 0) * ((nyb_) + 2 * 0) * ((nzb_+1) + 2 * 0) * ( ( (4) + (0) + 1 ) ) * sizeof(Real)"
assert extents[1] == "( (4) + (0) + 1 )"
assert extents[2] == 'fcz'

extents = md.parse_extents([16, 16, 1, 4], 0, 0)
assert extents[0] == '(16 * 16 * 1 * 4)'
assert extents[1] == 4
assert extents[2] == None

extents = md.parse_extents([16, 16, 1, 4], 0, 0, size='Real')
print(extents)
assert extents[0] == '(16 * 16 * 1 * 4) * sizeof(Real)'
assert extents[1] == 4
assert extents[2] == None

print("Tests completed, no problems found.")