version 16.0
args outdir dates
clear all
set more off
set sortseed 171026
set max_memory 16g

*! 1.0.0 Local benchmark fixture 17oct2026
program define dmdf_staggered_fixture
        version 16.0
        args units periods cohorts first last filename outdir
        tempvar cohort_tag

        clear
        set seed `=1700000 + `units' + `periods''
        set obs `units'
        generate long id = _n
        generate byte never = mod(id, 5) == 0
        generate long treated_rank = id - floor(id/5)
        generate int cohort_index = mod(treated_rank - 1, `cohorts')
        generate int cohort = `first' + ///
                floor(cohort_index * (`last' - `first')/(`cohorts' - 1))
        replace cohort = . if never
        generate int cluster = ceil(id/10)
        generate double alpha = rnormal()
        generate double heterogeneity = runiform()

        * Verify the intended adoption structure before expanding the panel.
        egen byte `cohort_tag' = tag(cohort) if !missing(cohort)
        quietly count if `cohort_tag' == 1
        assert r(N) == `cohorts'
        quietly count if never
        assert r(N) == `units'/5
        assert inrange(cohort, `first', `last') if !never
        drop `cohort_tag' cohort_index treated_rank

        expand `periods'
        bysort id: generate int time = _n
        generate byte treatment = time >= cohort & !missing(cohort)
        generate double x1 = rnormal() + time/20
        generate double innovation = rnormal()
        bysort id (time): generate double noise = innovation
        by id (time): replace noise = ///
                0.6 * noise[_n - 1] + innovation if _n > 1
        generate double effect = treatment * ///
                (0.2 + 0.15 * heterogeneity + ///
                0.005 * min(max(time - cohort, 0), 12))
        generate double y = alpha + 0.015 * time + 0.3 * x1 + ///
                effect + noise
        drop innovation noise effect

        isid id time
        assert _N == `units' * `periods'
        by id (time): assert treatment >= treatment[_n - 1] if _n > 1
        by id (time): assert treatment[1] == 0
        assert !missing(y, treatment, cluster)
        save "`outdir'/`filename'.dta", replace

        display as text "Verified: `filename'; `units' units; " ///
                "`periods' periods; `cohorts' adoption dates."
end

/*
The optional stress fixtures retain one adopter per date and 20% never-
treated units. The calendar length grows with the adoption-date count.
*/
local units = `dates'/4 * 5
local periods = `dates' + 40
local last = `dates' + 20
dmdf_staggered_fixture `units' `periods' `dates' 21 `last' ///
        stress_`dates' "`outdir'"
