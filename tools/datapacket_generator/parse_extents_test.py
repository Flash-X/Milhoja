import milhoja_data as md

extents = md.parse_extents("cc(NGUARD, 1)", "FArray4D")
assert extents[0] == "(nxb + 2 * NGUARD * MILHOJA_K1D) * (nyb + 2 * NGUARD * MILHOJA_K2D) * (nzb + 2 * NGUARD * MILHOJA_K3D) * (1) * sizeof(FArray4D)"
assert extents[1] == 1
assert extents[2] == 'cc'

extents = md.parse_extents("cc(NGUARD, NUNKVAR-1)", "FArray4D")
assert extents[0] == "(nxb + 2 * NGUARD * MILHOJA_K1D) * (nyb + 2 * NGUARD * MILHOJA_K2D) * (nzb + 2 * NGUARD * MILHOJA_K3D) * (NUNKVAR-1) * sizeof(FArray4D)"
assert extents[1] == "NUNKVAR-1"
assert extents[2] == 'cc'

extents = md.parse_extents("cc(nguard, nunkvar-2)", "FArray4D")
assert extents[0] == "(nxb + 2 * NGUARD * MILHOJA_K1D) * (nyb + 2 * NGUARD * MILHOJA_K2D) * (nzb + 2 * NGUARD * MILHOJA_K3D) * (NUNKVAR-2) * sizeof(FArray4D)"
assert extents[1] == "NUNKVAR-2"
assert extents[2] == 'cc'

extents = md.parse_extents("fcx(0, NFLUXES)", "Real")
assert extents[0] == "((nxb+1) + 2 * 0) * ((nyb) + 2 * 0) * ((nzb) + 2 * 0) * NFLUXES * sizeof(Real)"
assert extents[1] == "NFLUXES"
assert extents[2] == 'fcx'

extents = md.parse_extents("fcy(0, NFLUXES)", "Real")
assert extents[0] == "((nxb) + 2 * 0) * ((nyb+1) + 2 * 0) * ((nzb) + 2 * 0) * NFLUXES * sizeof(Real)"
assert extents[1] == "NFLUXES"
assert extents[2] == 'fcy'

extents = md.parse_extents("fcz(0, NFLUXES)", "Real")
assert extents[0] == "((nxb) + 2 * 0) * ((nyb) + 2 * 0) * ((nzb+1) + 2 * 0) * NFLUXES * sizeof(Real)"
assert extents[1] == "NFLUXES"
assert extents[2] == 'fcz'

extents = md.parse_extents([16, 16, 1, 4])
assert extents[0] == '(16 * 16 * 1 * 4)'
assert extents[1] == 4
assert extents[2] == None

print("Tests completed, no problems found.")