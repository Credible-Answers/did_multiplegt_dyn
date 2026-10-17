version 16.0
args outdir
clear all
set more off
set seed 1701027
set sortseed 1701027

*! 1.0.0 Test fixture 17oct2026
program define dmdf_fixture
        version 16.0
        args groups periods filename outdir
        clear
        set obs `groups'
        generate long id = _n
        generate int cohort = 3 + mod(id, max(`periods' - 4, 1))
        replace cohort = 999 if mod(id, 5) == 0
        generate byte baseline = mod(id, 3)
        generate byte direction = 2 * (mod(id, 2) == 0) - 1
        generate int cluster = ceil(id/10)
        generate byte stratum = mod(cluster, 3)
        generate byte subgroup = mod(cluster, 2)
        generate double alpha = rnormal()
        generate double heterogeneity = runiform()
        expand `periods'
        bysort id: generate int time = _n
        generate double x1 = rnormal() + time/20
        generate double x2 = rnormal() + alpha/3
        generate double weight = 0.5 + 2 * runiform()
        generate double treatment = baseline + direction * (time >= cohort)
        replace treatment = treatment + direction * (time >= cohort + 5)
        generate double y = alpha + 0.05 * time + 0.3 * x1 - 0.2 * x2 ///
                + 0.015 * treatment + 0.002 * max(time - cohort, 0) ///
                + rnormal()
        save "`outdir'/`filename'.dta", replace
end

dmdf_fixture 600 16 small "`outdir'"
