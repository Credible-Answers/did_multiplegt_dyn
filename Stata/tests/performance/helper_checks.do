version 16.0
clear all
set more off
set seed 172026
args candidate
quietly run "`candidate'"

/*
Stress the covariance replacement against the actual generate/summarize
sequence, including float rounding, missing values, zero rows, and large
values. Run under both supported default storage types.
*/
foreach storage in float double {
        clear
        set type `storage'
        set obs 10000
        generate byte first_obs_by_gp_XX = mod(_n, 5) == 1
        generate a = rnormal() * 100
        generate b = rnormal() * 100
        replace a = . if mod(_n, 41) == 0
        replace b = .a if mod(_n, 43) == 0
        replace a = 1e20 if mod(_n, 113) == 0
        replace b = -1e20 if mod(_n, 113) == 0
        scalar G_XX = 2000
        scalar se1 = 0.13
        scalar se2 = 0.27
        scalar d1 = 1.7
        scalar d2 = -2.3

        foreach normalized in 0 1 {
                if `normalized' {
                        generate pair = a/d1 + b/d2
                        local denominators d1 d2
                }
                else {
                        generate pair = a + b
                        local denominators
                }

                generate squared = pair^2 * first_obs_by_gp_XX
                quietly summarize squared
                scalar reference = (r(sum)/G_XX^2 - se1^2 - se2^2)/2

                mata: _didmgt_dyn_covariance("a b", "se1 se2", ///
                        st_local("denominators"), "actual", ///
                        st_local("storage"))
                assert reldif(reference, actual[1, 2]) <= 1e-14
                assert actual[1, 1] == se1^2
                assert actual[2, 2] == se2^2
                drop pair squared
        }
}

clear
set obs 10000
generate int time_XX = mod(_n, 50) + 1
generate double total = time_XX * sqrt(2)
replace total = . if time_XX == 20
scalar t_min_XX = 3
scalar T_max_XX = 48
scalar reference = 0

forvalues t = 3/48 {
        quietly summarize total if time_XX == `t'

        if !missing(r(mean)) {
                scalar reference = reference + r(mean)
        }
}

mata: _didmgt_dyn_time_total("total", "actual_total")
assert reldif(reference, actual_total) <= 1e-14
display "HELPER_CHECKS_PASSED"
exit, clear
