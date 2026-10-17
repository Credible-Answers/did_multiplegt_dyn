version 16.0
args outdir
clear all
set more off
set seed 17102026
set sortseed 17102026

* Staggered cycles include both initial increases and initial decreases.
set obs 400
generate long id = _n
generate byte initial = mod(id, 2)
generate byte direction = 1 - 2 * initial
generate byte never = mod(ceil(id/2), 5) == 0
generate int first = 6 + mod(ceil(id/2), 8)
generate int cluster = ceil(id/10)
generate double alpha = rnormal()
generate double heterogeneity = runiform()
expand 40
bysort id: generate int time = _n
generate int event_time = time - first
generate byte phase = mod(floor(max(event_time, 0)/7), 4)
generate byte treatment = initial
replace treatment = initial + direction if event_time >= 0 & ///
        mod(phase, 2) == 0 & !never
generate double x1 = rnormal() + time/20
generate double weight = 0.5 + runiform()
generate double innovation = rnormal()
bysort id (time): generate double noise = innovation
by id (time): replace noise = ///
        0.5 * noise[_n - 1] + innovation if _n > 1
xtset id time
generate double y = alpha + 0.02 * time + 0.3 * x1 + ///
        (0.25 + 0.1 * heterogeneity) * treatment + ///
        0.15 * L.treatment + 0.05 * L2.treatment + noise
replace y = alpha + 0.02 * time + 0.3 * x1 + ///
        (0.25 + 0.1 * heterogeneity) * treatment + noise if time <= 2
drop innovation noise
assert !missing(y, treatment)
isid id time
save "`outdir'/reversal_binary.dta", replace

* Every unit eventually switches; reset can reuse stable treatment spells.
replace never = 0
replace treatment = initial + direction * ///
        (event_time >= 0 & mod(phase, 2) == 0)
replace y = alpha + 0.02 * time + 0.3 * x1 + ///
        (0.25 + 0.1 * heterogeneity) * treatment + ///
        0.15 * L.treatment + 0.05 * L2.treatment + rnormal()
replace y = alpha + 0.02 * time + 0.3 * x1 + ///
        (0.25 + 0.1 * heterogeneity) * treatment + rnormal() if time <= 2
assert !missing(y, treatment)
save "`outdir'/reversal_all.dta", replace

/*
Dose paths cross both sides of baseline: 2 -> 3 -> 1 -> 2 -> 3 ...
and the reverse sequence. This exercises dont_drop_larger_lower.
*/
generate byte delta = cond(phase == 0, 1, ///
        cond(phase == 1, -1, cond(phase == 2, 0, 1)))
replace treatment = 2 + direction * delta if event_time >= 0
replace treatment = 2 if event_time < 0
replace never = mod(ceil(id/2), 5) == 0
replace treatment = 2 if never
replace y = alpha + 0.02 * time + 0.3 * x1 + ///
        (0.25 + 0.1 * heterogeneity) * treatment + ///
        0.15 * L.treatment + 0.05 * L2.treatment + rnormal()
replace y = alpha + 0.02 * time + 0.3 * x1 + ///
        (0.25 + 0.1 * heterogeneity) * treatment + rnormal() if time <= 2
assert !missing(y, treatment)
save "`outdir'/reversal_dose.dta", replace
display "NONABSORBING_FIXTURES_PASSED"
