local scalars : e(scalars)
local matrices : e(matrices)
local macros : e(macros)

mata:
fh = fopen(st_local("snapshot"), "w")
names = tokens(st_local("scalars"))
for (i = 1; i <= cols(names); i++) {
        fput(fh, "scalar/" + names[i] + char(9) +
                strofreal(st_numscalar("e(" + names[i] + ")"), "%24.17g"))
}
names = tokens(st_local("matrices"))
for (i = 1; i <= cols(names); i++) {
        a = st_matrix("e(" + names[i] + ")")
        rs = st_matrixrowstripe("e(" + names[i] + ")")
        cs = st_matrixcolstripe("e(" + names[i] + ")")
        for (j = 1; j <= rows(a); j++) {
                for (k = 1; k <= cols(a); k++) {
                        key = "matrix/" + names[i] + "/" +
                                strofreal(j) + "/" + strofreal(k)
                        fput(fh, key + char(9) +
                                strofreal(a[j, k], "%24.17g"))
                }
                fput(fh, "row/" + names[i] + "/" + strofreal(j) +
                        char(9) + invtokens(rs[j, ], "|"))
        }
        for (k = 1; k <= cols(a); k++) {
                fput(fh, "col/" + names[i] + "/" + strofreal(k) +
                        char(9) + invtokens(cs[k, ], "|"))
        }
}
names = tokens(st_local("macros"))
for (i = 1; i <= cols(names); i++) {
        fput(fh, "macro/" + names[i] + char(9) +
                st_global("e(" + names[i] + ")"))
}
a = st_matrix("mat_res_XX")
for (j = 1; j <= rows(a); j++) {
        for (k = 1; k <= cols(a); k++) {
                fput(fh, "table/" + strofreal(j) + "/" + strofreal(k) +
                        char(9) + strofreal(a[j, k], "%24.17g"))
        }
}
fclose(fh)
end

file open types using "`snapshot'", write append
file write types "data/N" _tab %12.0f (_N) _n
foreach variable of varlist _all {
        local storage : type `variable'
        file write types "data/type/`variable'" _tab "`storage'" _n
}
file close types
