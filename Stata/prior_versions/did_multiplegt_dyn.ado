*﻿* did_multiplegt_dyn : did_multiplegt, robust to dynamic effects and with asymptotic variances
** Henceforth, "the paper" refers to "Difference-in-Differences Estimators of Intertemporal
** Treatment Effects", by Chaisemartin and D'Haultfoeuille.
** Henceforth, "the companion paper" refers to "Estimators and Variance Estimators Computed by the
** did_multiplegt_dyn Command", by Chaisemartin et al.
** The adofile contains two programs. 
** Each program contains numbered sections, tagged with "//////////".
** Sections may contain unnumbered subsections, tagged with "/////".
** Subsections may contain unnumbered subsubsections, tagged with "//".
** Subsubsections may be further divided into paragraphs, tagged with "*".
** Comments are also tagged with "*".
** This version : December 31st, 2025

** This version includes Diego's changes:
**** Fixes to variance estimation with controls() (tks)
**** Fixes to same_switchers() and same_switchers_pl() with unbalanced panel
**** Fixes to controls() with unbalanced panel
**** no_updates turned into _no_updates
**** Fix to cluster variances
**** Fixed norm weights with unbalanced panels
**** Fixed effect_equal()
**** (proposed) comprehensive fix to demeaning with clusters + simplified syntax
**** (proposed) fix to demeaning when E_hat_denom == 1
**** more_granular_demeaning
**** Added Clement's second fix to placebo
**** Replaced _r_ub and _r_lb with explicit CI computation
**** Added sup test from sotable

** Felix:
**** Delete if d_sq_int_XX==`l' condition when summing placebo variances 
**** Adjust - when storing the switcher out variances
**** Add cap drop group2_XX to Tom's change regarding the re-counting of groups
**** Add different_paths_XX in the group statement "gegen rank_paths_XX=group(-count_path_XX different_paths_XX) if count_path_XX!=." when generating variable to distinguish paths for the by_path() option

** Clément:
**** Demeaning for singular cohorts
**** No demeaning if singular cohorts have joint DoF = 1

**** Line 3204, add if "`trends_lin'"=="" condition when adding the avg_effect estimate to the matrix used for esttab (solve delta_XX not found error) -> needed to add this condition at multiple places
**** Add ereturn for p-value test joint nullity effects

*** DAC:
**** Added HC2 BM DOF SES to predict_het

********************************************************************************
*                                 PROGRAM 1                                    *
********************************************************************************

capture program drop did_multiplegt_dyn

program did_multiplegt_dyn, eclass
	version 12.0
	syntax varlist(min=4 max=4 numeric) [if] [in] [, effects(integer 1) placebo(integer 0) switchers(string) only_never_switchers controls(varlist numeric) trends_nonparam(varlist numeric) weight(varlist numeric max=1) dont_drop_larger_lower NORMALIZED cluster(varlist numeric max=1) graphoptions(string) save_results(string) graph_off same_switchers same_switchers_pl effects_equal(string)  drop_if_d_miss_before_first_switch trends_lin ci_level(integer 95) by(varlist numeric max=1) predict_het(string) predict_het_hc2bm design(string) date_first_switch(string)  NORMALIZED_weights CONTinuous(integer 0) save_sample less_conservative_se more_granular_demeaning by_path(string) bootstrap(string) _no_updates]
	
////////// 0. Auto-updates
if "`_no_updates'" == "" {
	if uniform() < 0.01 {
		noi ssc install did_multiplegt_dyn, replace
	}
}	

////////// 1. Checking that necessary tools to run command installed, and that syntax correctly specified.	
		
///// Check if gtools is installed, if not present link to install
qui cap which gtools
if _rc{
	di ""
	di as error "You have not installed the gtools package which is used within the did_multiplegt_dyn command."
	di `"{stata "ssc install gtools": Click here to install gtools}"'
	di as input _continue ""
	
	exit
}	

// Save controls in a specific local if trends_lin by() continuous() and continuous are specified at the same time
if "`trends_lin'"!=""&"`by'"!=""&`continuous'!=0{
	local controls_orig_XX = "`controls'"
}

// Error message if by and by_path specified at the same time
if "`by'"!=""&"`by_path'"!=""{	
	di as error ""
	di as error "The options by() and by_path() can not be specified at the same time!"
	di as input _continue ""
	exit
}


if `continuous'<0{
	di ""
	di as error "The input of the continuous() option has to be a positive integer!"
	di as input _continue ""
	
	exit
}

if ("`by_path'"!=""&"`graphoptions'"!=""){
	di ""
	di as error "by_path() builds on pre-specified graph options so it does not allow for any further inputs in graphoptions()!"
	di as input _continue ""
	
	exit
}	

if ("`by_path'"!=""&"`by_path'"!="all"){
	
	capture confirm integer number `by_path'
	if _rc{
	di ""	
	di as error "The input of the by_path() option has to be {it:all} or a positive integer!"
	di as input _continue ""
	
	exit
	}
	
if `by_path'<=0{
	di ""
	di as error "The input of the by_path() option has to be {it:all} or a positive integer!"
	di as input _continue ""
	
	exit
}
}

if "`bootstrap'"!=""&`continuous'==0{
	di as error "You specified the bootstrap option without the continuous option. Please be aware that we strongly recommend to only compute bootstraped standard errors when you want to use the continuous option. Otherwise you can rely on the (much faster) computation of analytical standard errors."	
	di as input _continue ""
}

if "`bootstrap'"==""&`continuous'>0{
	di as error "You specified the continuous option without the bootstrap option. Please be aware that we recommend to compute bootstraped standard errors when you are using the continuous option as the analytical standard errors can be liberal in that case."	
	di as input _continue ""
}

// Allow to specify seed with bootstrap
if "`bootstrap'"==""{
	local bootstrap_XX = 0 // allowing to keep the structure from before
}	
if "`bootstrap'"!=""{
	if strpos("`bootstrap'", ",") == 0 {
		di as error ""
		di as error "Syntax error in bootstrap option"
		di as error "Comma required"
		di as input _continue ""
		exit
	}

	local bootstrap_XX = strtrim(substr("`bootstrap'", 1, strpos("`bootstrap'", ",") - 1))
	local seed_temp_XX = strtrim(substr("`bootstrap'", strpos("`bootstrap'", ",") + 1, .))
	
	// set default to 50 reps
	if "`bootstrap_XX'"==""{
		local bootstrap_XX = 50
	}
	
	// set default to no seed
	if "`seed_temp_XX'"==""{
		local seed_XX ""
	}
	if "`seed_temp_XX'"!=""{
		local seed_XX "seed(`seed_temp_XX')"
	}
}

// Check that bootstrap input is correct 
if `bootstrap_XX'!=0{
	
	local controls_bs_orig_XX="`controls'"
	
	capture confirm integer number `bootstrap_XX'
	if _rc{
	di ""	
	di as error "The input of the bootstrap() option has to be a positive integer greater than 1"
	di as input _continue ""
	exit
	}
	
if `bootstrap_XX'<2{
	di ""
	di as error "The input of the bootstrap() option has to be a positive integer greater than 1"
	di as input _continue ""
	exit
}
}

///// Add a Warning that same_switchers_pl only works when same_switchers is specified.
if "`same_switchers_pl'"!=""&"`same_switchers'"==""{
	di ""
	di as error "The same_switchers_pl option only works if same_switchers is specified as well!"
	di as input _continue ""
	
	exit
}

///// Add a Warning that continous and the design option(s) should not be specified simulataneously
if `continuous'>0&"`design'"!=""{
	di ""
	di as error "The design option can not be specified together with the continuous option!"
	di as input _continue ""

	exit
}

if "`more_granular_demeaning'" != "" {
	local less_conservative_se "less_conservative_se"
}

////////// 2. Data preparation steps	
	
///// Storing path of initial dataset
local dataset_name_XX `c(filename)'

if "`save_sample'" != "" {
	cap drop _did_sample*
}

///// Preserving data 
preserve

///// Quietly
	qui{

	
///// capture drop of several variables created below
capture drop outcome_non_diff_XX				
capture drop outcome_XX
capture drop group_XX
capture drop time_XX
capture drop treatment_XX
capture drop d_sq_XX
capture drop d_fg_XX
capture drop diff_from_sq_XX
capture drop ever_change_d_XX
capture drop never_change_d_XX
capture drop temp_F_g_XX
capture drop F_g_XX
capture drop t_min_XX
capture drop T_max_XX
capture drop N_gt_XX
capture drop T_g_XX
capture drop controls_time_XX
capture drop _fillin
capture drop time_d_nonmiss_XX
capture drop min_time_d_nonmiss_XX
capture drop d_F_g_temp_XX
capture drop d_F_g_XX
capture drop S_g_XX
capture drop S_g_het_XX
capture drop L_g_XX
capture drop U_Gg_plus_XX
capture drop U_Gg_minus_XX
capture drop U_Gg_global_XX
capture drop w_plus_XX
capture drop first_obs_by_gp_XX
capture drop first_obs_by_clust_XX
capture drop avg_post_switch_treat_XX
capture drop count_time_post_switch_XX_temp
capture drop count_time_post_switch_XX
capture drop ever_strict_increase_XX
capture drop ever_strict_decrease_XX
capture drop U_Gg_var_plus_XX
capture drop U_Gg_var_minus_XX
capture drop U_Gg_var_global_XX
capture drop U_Gg_var_global_2_XX
capture drop weight_XX
capture drop delta_XX
capture drop aux_XX

// ereturn clear //Modif Felix

tokenize `varlist'

///// Patching the cluster option: by default, command clusters at group level. 
///// If user specified clustering according to group, replace it by empty. 
if "`cluster'"=="`2'" local cluster ""

///// Selecting the sample 

//dropping observations with missing group or time
drop if `2'==.|`3'==.
//dropping observations with missing controls
	if "`controls'" !=""{
	foreach var of varlist `controls'{
	drop if `var'==.
	}
	}
//dropping observations not included in the if condition
	if "`if'" !=""{
	keep `if'
	}
// dropping observations not included in the in condition
	if "`in'" !=""{
	keep `in'
	} 
	
	
///// Collapse and weight
	
// Checking wether data has to be collapsed, because at a more disaggregated level than group*time.
 
capture drop counter_XX
capture drop counter_temp_XX
gen counter_temp_XX=1
bys `2' `3' : gegen counter_XX=count(counter_temp_XX)
sum counter_XX
scalar aggregated_data=0
if r(max)==1{
scalar aggregated_data=1
}

// Creating the weight variable.

if("`weight'"==""){
gen weight_XX = 1
}
else{
gen weight_XX = `weight'
}

replace weight_XX=0 if weight_XX==.

// Collapsing the data when necessary

if scalar(aggregated_data)==0{
	
	replace weight_XX=10^(-38) if `1'==.
	capture drop tag_y_miss_XX
	gen tag_y_miss_XX=(`1'==.)
	
	if "`1'"!="`4'"{			
		collapse (mean) `1' `4' tag_y_miss_XX `controls' `trends_nonparam' `cluster' `by' `predict_het' (count) weight_XX [pw=weight_XX], by(`2' `3') 
	}
	
	if "`1'"=="`4'"{
				
		collapse (mean) `1' `controls' `trends_nonparam' `cluster' `by' `predict_het' (count) weight_XX [pw=weight_XX], by(`2' `3')
	}	
	
	replace weight_XX=0 if tag_y_miss_XX==1
}
	
//// Modif Felix 26.05.2025 -> tempvar to use the correct weights when we use by_path and the data is collapsed becausee the group variable is coarser than the panel identifier
if "`by_path'" != ""{
	tempvar weight_path_XX
	generate `weight_path_XX' = weight_XX
}	
	
///// by option for heterogeneous treatment effects analysis

// define local so we run the loop once if by is not specified
local levels_byvar_XX 1 

// checking that by variable is time-invariant
scalar by_XX  = 1
if "`by'" !=""{
	xtset `2' `3'
	xtsum `by'
	if `r(sd_w)'!=0{
		display as error "The variable specified in the option by is time-varying. That variable should be time-invariant."
		di as input _continue ""
		exit
	}
}

if "`by'" !=""{
// save data to load again in the multiple repetitions of the estimation
tempfile by_data_XX
save "`by_data_XX'.dta", replace
// store local with number of estimations we need to run
levelsof `by', local(levels_byvar_XX)
local `by'_lab_XX: value label `by'
}
// Loop to run estimation multiple times in case of by
} // end of the quietly condition
foreach l_by of local levels_byvar_XX { 
	qui{
if "`by'" !=""{	
* Check if we have value labels	
ds `by', has(vallabel)
local `by'_has_vallabel_XX=r(varlist)
if "``by'_has_vallabel_XX'"!="."{
	local val_lab_int_XX "`: label ``by'_lab_XX' `l_by''"
} 
if "``by'_has_vallabel_XX'"=="."{
	local val_lab_int_XX "`l_by'"
}
// sample restriction for the by option
	keep if `by' == `l_by'
}		

///// Further sample selection steps

// dropping observations with a missing clustering variable
	if "`cluster'" !=""{
	drop if `cluster'==.
	}
//dropping groups with always missing treatment or outcome
capture drop mean_d_XX
capture drop mean_y_XX
bys `2': gegen mean_d_XX=mean(`4')
bys `2': gegen mean_y_XX=mean(`1')
drop if mean_d_XX==.|mean_y_XX==.
drop mean_d_XX mean_y_XX

// save data for bootstrap
if `bootstrap_XX'!=0{
	
	qui tempfile data_bootstrap_XX
	qui save "`data_bootstrap_XX'.dta", replace

	if "`less_conservative_se'"!=""{
		di as error ""
		di as error "The less_conservative_se option has no effect when bootstrap is specified, therefore it is ignored in this case to save computation time."
		di as input _continue ""
		local less_conservative_se ""
}
}

///// predict_het option for heterogeneous treatment effects analysis

if "`predict_het'"!=""{

if strpos("`predict_het'", ",") == 0 {
		di as error ""
		di as error "Syntax error in predict_het option"
		di as error "Comma required"
		di as input _continue ""
		exit
	}

local het_vars_XX = strtrim(substr("`predict_het'", 1, strpos("`predict_het'", ",") - 1))
local het_nums_XX = strtrim(substr("`predict_het'", strpos("`predict_het'", ",") + 1, .))

	// Modif Felix: Add checks for correct inputs in predict_het
	if "`het_vars_XX'"==""{
		di as error "You did not specify any variables to be used in the predict_het option, therefore this option will be ignored"
		di as input _continue ""
	}
	
	if "`het_nums_XX'"==""{
		local het_nums_XX
	}
	
	local pred_het `het_vars_XX'
	local het_effects `het_nums_XX'	

* Check if predict_het and normalized are specified simulataneously
if "`normalized'"!=""{
	local pred_het ""
	local het_effects ""
	local predict_het ""
	di as error ""
	di as error "The options normalized and predict_het can not be specified at the same time, therefore predict_het will be ignored."
	di as input _continue ""
}

* Check if predict_het and controls are specified simulataneously
if "`predict_het'"!=""&"`controls'"!=""{
	local pred_het ""
	local het_effects ""
	local predict_het ""
	di as error ""
	di as error "The options controls() and predict_het can not be specified at the same time, therefore predict_het will be ignored."
	di as input _continue ""
}
}	

* Check if only time-invariant variables are specified in predict_het
if "`pred_het'" !=""{
	
	local predict_het_bad=""
	local predict_het_good=""
	
	foreach var in `pred_het'{
	bys `2': sum `var'
	
	if r(sd)!=0{
		local predict_het_bad="`predict_het_bad' `var'"
	}
	if r(sd)==0{
		local predict_het_good="`predict_het_good' `var'"
	}
	}
	
	if "`predict_het_bad'"!=""{
	display as error "The variable(s) (`predict_het_bad') specified in the option predict_het"
	display as error "is(are) time-varying, the command will therefore ignore them."
	di as input _continue ""
	}
}

*Limit use of predict_het_hc2bm to cases where predict_het is specified
if "`predict_het'"=="" & "`predict_het_hc2bm'" != "" {
	di as error ""
	di as error "Option predict_het_hc2bm only available when predict_het is specified."
	
	di as input _continue ""
	
	exit
}


///// Creating Y G T D variables

gen outcome_XX=`1'
gegen group_XX=group(`2')
gegen time_XX=group(`3')
gen treatment_XX=`4'

///// Declaring data as (G,T) panel

xtset group_XX time_XX

///// Creating variables useful to deal with imbalanced panels:

// G's first and last date when D not missing

gen time_d_nonmiss_XX=time_XX if treatment_XX!=.
bys group_XX: gegen min_time_d_nonmiss_XX=min(time_d_nonmiss_XX)
capture drop max_time_d_nonmiss_XX
bys group_XX: gegen max_time_d_nonmiss_XX=max(time_d_nonmiss_XX)

// G's first date when Y not missing

capture drop time_y_nonmiss_XX
capture drop min_time_y_nonmiss_XX
gen time_y_nonmiss_XX=time_XX if outcome_XX!=.
bys group_XX: gegen min_time_y_nonmiss_XX=min(time_y_nonmiss_XX)

// G's first date when D missing after Y has been not missing

capture drop time_d_miss_XX
capture drop min_time_d_miss_aft_ynm_XX
gen time_d_miss_XX=time_XX if treatment_XX==.&time_XX>=min_time_y_nonmiss_XX
bys group_XX: gegen min_time_d_miss_aft_ynm_XX=min(time_d_miss_XX)
drop time_d_nonmiss_XX time_y_nonmiss_XX time_d_miss_XX 

///// Creating baseline treatment
///// D_{g,1} in paper, redefined to account for imbalanced panels: 
///// g's treatment at first period where g's treatment non missing

gen d_sq_XX_temp=treatment_XX if time_XX==min_time_d_nonmiss_XX
bys group_XX: gegen d_sq_XX=mean(d_sq_XX_temp)
drop d_sq_XX_temp 

///// Enforcing Design Restriction 2 in the paper.
///// If the option dont_drop_larger_lower was not specified, 
///// drop (g,t) cells such that at t, g has experienced both a strictly lower 
///// and a strictly higher treatment than its baseline treatment.

gen diff_from_sq_XX=(treatment_XX-d_sq_XX)

if "`dont_drop_larger_lower'"==""{		
	sort group_XX time_XX
	gen ever_strict_increase_XX=(diff_from_sq_XX>0&treatment_XX!=.) 
	replace ever_strict_increase_XX=1 if ever_strict_increase_XX[_n-1]==1&group_XX==group_XX[_n-1]

	gen ever_strict_decrease_XX=(diff_from_sq_XX<0&treatment_XX!=.) 
	replace ever_strict_decrease_XX=1 if ever_strict_decrease_XX[_n-1]==1&group_XX==group_XX[_n-1]

	drop if ever_strict_increase_XX==1 & ever_strict_decrease_XX==1
	
	drop ever_strict_increase_XX ever_strict_decrease_XX
		
}

///// Sort data
sort group_XX time_XX

///// Counting number of groups
//sum group_XX
//scalar G_XX=r(max)

///// Ever changed treatment
gen ever_change_d_XX=(abs(diff_from_sq_XX)>0&treatment_XX!=.)
replace ever_change_d_XX=1 if ever_change_d_XX[_n-1]==1&group_XX==group_XX[_n-1]

///// Creating date of first treatment change
sort group_XX time_XX
gen temp_F_g_XX=time_XX if ever_change_d_XX==1&ever_change_d_XX[_n-1]==0 
replace temp_F_g_XX=0 if temp_F_g_XX==.
bys group_XX: gegen F_g_XX=max(temp_F_g_XX)
drop temp_F_g_XX

///// If continuous option specified, generating polynomials of D_{g,1},
///// storing D_{g,1} somewhere, and replacing it by 0.
if `continuous'>0{
	
	scalar degree_pol = `continuous'
	
forvalues pol_level = 1/`=degree_pol'{
	scalar pol_level_XX = `pol_level'
	capture drop d_sq_`pol_level'_XX 
	gen d_sq_`pol_level'_XX = d_sq_XX^scalar(pol_level_XX)
}
capture drop d_sq_XX_orig
gen d_sq_XX_orig=d_sq_XX
replace d_sq_XX=0
}

// Create a new variable = to d_sq_XX expressed in integers
capture drop d_sq_int_XX

gegen d_sq_int_XX = group(d_sq_XX)

///// Dropping the values of the baseline treatment such that no variance in F_g within those values.
capture drop var_F_g_XX
bys d_sq_XX `trends_nonparam': gegen var_F_g_XX=sd(F_g_XX)
drop if var_F_g_XX==0
drop var_F_g_XX

//// CHANGE BELOW - tks
//// Counting number of groups after the drop above
cap drop group2_XX
egen group2_XX = group(group_XX)
sum group2_XX
scalar G_XX=r(max)
////

///// Error message if Design Restriction 1 is not met.
count
if r(N)==0{
	di as error ""
	di as error "No treatment effect can be estimated. This is because Design Restriction 1 in de Chaisemartin & D'Haultfoeuille (2024) is not satisfied in the data, given the options requested. This may be due to the fact that groups' period-one treatment is continuous, or takes a large number of values, and you have not specified the continuous option. If so, you can try to specify this option. If the issue persists even with this option, this means that all groups experience their first treatment change at the same date, a situation where the estimators of de Chaisemartin & D'Haultfoeuille (2024) cannot be used."
	di as input _continue ""

	exit
}

///// For each value of the baseline treatment, we drop time periods such that we do not have any control with that baseline treatment after that period
///// This means the panel is no longer balanced, though it is balanced within values of the baseline treatment
gen never_change_d_XX=1-ever_change_d_XX
bys time_XX d_sq_XX `trends_nonparam': gegen controls_time_XX=max(never_change_d_XX)
drop if controls_time_XX==0

///// Computing t_min_XX, T_max_XX, and replacing F_g_XX by last period plus one for those that never change treatment
sum time_XX
scalar t_min_XX=r(min)
scalar T_max_XX=r(max)
replace F_g_XX=T_max_XX+1 if F_g_XX==0

///// Dealing with missing treatments: most conservative option

*Let FMD_g denote the first date when g's treatment is missing while y has been not missing at least once, so that we know for sure that g already exists. If that date is before the first period when g's treatment changes, we do not know when g's treatment has changed for the first time. Then, a conservative option is to drop all of g's outcomes starting at FMD_g.

if "`drop_if_d_miss_before_first_switch'"!=""{
replace outcome_XX=. if min_time_d_miss_aft_ynm_XX<F_g_XX&time_XX>=min_time_d_miss_aft_ynm_XX
}

///// Dealing with missing treatments: most liberal option

*Let FD_g and LD_g respectively denote the first and last period where a group's treatment is non missing. Let FY_g denote the first period where a group's outcome is non missing.

*For groups that experience at least one treatment change, let LDBF_g denote the last date before F_g where g's treatment is non missing. We have FD_g<=LDBF_g<F_g<=LD_g, and we will deal with missing treatments depending on when they occur with respect to those four dates.

capture drop last_obs_D_bef_switch_t_XX
capture drop last_obs_D_bef_switch_XX
gen last_obs_D_bef_switch_t_XX=time_XX if time_XX<F_g_XX&treatment_XX!=.
bys group_XX: gegen last_obs_D_bef_switch_XX=max(last_obs_D_bef_switch_t_XX)

*For groups that do not experience a treatment change, we just have FD_g<=LD_g, and we will deal with missing treatments depending on when they occur with respect to those two dates.

*For t<FD_g, by default we are going to consider that g joins the panel at FD_g: any non-missing outcome before FD_g replaced as missing, but all non-missing outcomes after FD_g are kept. For groups such that FY_g<FD_g, this is a "liberal" convention: those groups exist before FD_g, so one could argue that their status quo treatment is missing and they should be dropped from the analysis. We give the user the option to do that, with drop_if_d_miss_before_first_switch option
  
replace outcome_XX=. if time_XX<min_time_d_nonmiss_XX 

*For groups that experience a treatment change, if D_gt missing at FD_g<t<LDBF_g, we replace their missing treatment by their status-quo treatment. Again, this is a liberal convention, so we give the user the option to not use those observations, with drop_if_d_miss_before_first_switch option.

replace treatment_XX=d_sq_XX if F_g_XX<T_max_XX+1&treatment_XX==.&time_XX<last_obs_D_bef_switch_XX&time_XX>min_time_d_nonmiss_XX

*For groups that experience a treatment change, if D_gt missing at LDBF_g<t<F_g (equivalent to LDBF_g<F_g-1), we cannot know the exact date when their treatment has changed, even in a binary and staggered design. Therefore, we set their outcomes at missing starting at LDBF_g+1. We also redefine their F_g as T+1 because they are effectively control groups. We also define the trunc_control_XX as LDBF_g+1 for them, because they can only be used as controls till that date.

replace outcome_XX=. if F_g_XX<T_max_XX+1&time_XX>last_obs_D_bef_switch_XX&last_obs_D_bef_switch_XX<F_g_XX-1
capture drop trunc_control_XX
gen trunc_control_XX=last_obs_D_bef_switch_XX+1 if F_g_XX<T_max_XX+1&last_obs_D_bef_switch_XX<F_g_XX-1
replace F_g_XX=T_max_XX+1 if F_g_XX<T_max_XX+1&last_obs_D_bef_switch_XX<F_g_XX-1

*For groups that experience a treatment change, if D_gt missing at F_g<t, we replace their missing treatment by D(g,F_g). This is again a liberal convention, but it is innocuous for the reduced-form parameters DID_l, so we do not give the user the option to overrule it (Note that overruling it could make the same_switchers option fail)

gen d_F_g_temp_XX = treatment_XX if time_XX==F_g_XX
bys group_XX: gegen d_F_g_XX = mean(d_F_g_temp_XX)
replace treatment_XX=d_F_g_XX if F_g_XX<T_max_XX+1&treatment_XX==.&time_XX>F_g_XX&last_obs_D_bef_switch_XX==F_g_XX-1

*For groups that do not experience a treatment change, if D_gt missing at FD_g<t<LD_g, we replace their missing treatment by D_g1. This is again a liberal convention, so we give the user the option to not use those observations, with drop_if_d_miss_before_first_switch option.

replace treatment_XX=d_sq_XX if F_g_XX==T_max_XX+1&treatment_XX==.&time_XX>min_time_d_nonmiss_XX&time_XX<max_time_d_nonmiss_XX

*For groups that do not experience a treatment change, we replace all their outcomes by missing at t>LD_g. Even in a binary and staggered design, we cannot infer their treatment at t>LD_g.

replace outcome_XX=. if F_g_XX==T_max_XX+1&time_XX>max_time_d_nonmiss_XX
replace trunc_control_XX=max_time_d_nonmiss_XX+1 if F_g_XX==T_max_XX+1

///// Store the outcome in levels, will be useful later when predict_het and trends_lin specified
if "`predict_het_good'"!=""{
gen outcome_non_diff_XX=outcome_XX
}

///// If trends_lin option specified, drop units for which F_g_XX==2, and redefine 
///// outcome and controls in first difference.
if "`trends_lin'"!=""{
	drop if F_g_XX==2
	xtset group_XX time_XX
	capture drop FD_outcome_XX
	gen FD_outcome_XX =outcome_XX-L.outcome_XX
	replace outcome_XX = FD_outcome_XX
	drop FD_outcome_XX
	
	// adapt controls for case where trends_lin by and continuous specified together
	if "`trends_lin'"!=""&"`by'"!=""&`continuous'!=0{
	local controls = "`controls_orig_XX'"
}
	
	if "`controls'" !=""{
	foreach var of varlist `controls'{
	capture drop FD_`var'_XX
	gen FD_`var'_XX =`var' -L.`var'
	replace `var' = FD_`var'_XX
	drop FD_`var'_XX
	}
	}
	drop if time_XX==1
	sum time_XX
	scalar t_min_XX=r(min)
}

///// Balancing the panel
cap rename _merge _merge_og
fillin group_XX time_XX
capture drop d_sq_XX_new
bys group_XX: gegen d_sq_XX_new=mean(d_sq_XX)
drop d_sq_XX
rename d_sq_XX_new d_sq_XX

///// CHANGE BELOW - tks

bys group_XX: gegen d_sq_int_XX_new = mean(d_sq_int_XX)
drop d_sq_int_XX
rename d_sq_int_XX_new d_sq_int_XX

bys group_XX: gegen F_g_XX_new = mean(F_g_XX)
drop F_g_XX
rename F_g_XX_new F_g_XX

/////

///// Defining N_gt, the weight of each cell (g,t)
gen N_gt_XX=1
replace N_gt_XX = N_gt_XX*weight_XX  //
replace N_gt_XX=0 if outcome_XX==.|treatment_XX==.

///// Determining last period where g still has a control group:
///// There is still a group with same 
///// treatment as g's in period 1 and whose treatment has not changed since 
///// start of panel. Definition adapted from the paper, to account for 
///// imbalanced panel.

capture drop F_g_trunc_XX
gen F_g_trunc_XX=F_g_XX
replace F_g_trunc_XX=min(F_g_XX,trunc_control_XX) if trunc_control_XX!=.
bys d_sq_XX `trends_nonparam': gegen T_g_XX = max(F_g_trunc_XX) 
replace T_g_XX = T_g_XX-1

///// Defining S_g: 
///// an indicator variable for groups whose average post switch 
///// treatment value is larger than their initial treatment D_{g,1}. 
///// They will be considered switchers in. If S_g==0, the group is a switcher out. 
///// For never-switchers, S_g is undefined.
///// Definition of S_g matches that in paper, unless dont_drop_larger_lower specified.

bys group_XX : gegen avg_post_switch_treat_XX_temp = total(treatment_XX) if time_XX>=F_g_XX&time_XX<=T_g_XX
gen count_time_post_switch_XX_temp=(treatment_XX!=.) if time_XX>=F_g_XX&time_XX<=T_g_XX
bysort group_XX : gegen count_time_post_switch_XX=total(count_time_post_switch_XX_temp)
replace avg_post_switch_treat_XX_temp =avg_post_switch_treat_XX_temp/count_time_post_switch_XX
bys group_XX: gegen avg_post_switch_treat_XX=mean(avg_post_switch_treat_XX_temp)
drop avg_post_switch_treat_XX_temp

// When a group is a switching group, but its average post-treatment treatment 
// value is exactly equal to its baseline treatment, we cannnot classify it as 
// a swicher in or a switcher out, but it is not a control either. 
// As such, we drop it from the estimation. Those groups are referred to 
// as no-first-stage-switchers. This issue can only arise 
// if dont_drop_larger_lower specified. 
// if continuous is specified we do this according to the original 
// baseline treatment and not to the one set to 0 to correctly
// track if a group is switcher in or switcher out.
if `continuous'==0{
	drop if avg_post_switch_treat_XX==d_sq_XX&F_g_XX!=T_g_XX+1
	gen S_g_XX=(avg_post_switch_treat_XX>d_sq_XX) if F_g_XX!=T_max_XX+1
}
else if `continuous'>0{
	drop if avg_post_switch_treat_XX==d_sq_XX_orig&F_g_XX!=T_g_XX+1
	gen S_g_XX=(avg_post_switch_treat_XX>d_sq_XX_orig) if F_g_XX!=T_max_XX+1
}	
// correct for the fact that if all avg_post_switch_treat_XX are missing it is considered as +inf
egen aux_XX = min(avg_post_switch_treat_XX), by(group_XX)
replace S_g_XX = . if aux_XX ==.

// Define another version where S_g=-1 for switchers out, which we need 
// when predict_het or continuous specified.
if "`predict_het_good'"!="" | `continuous'>0{
gen S_g_het_XX=S_g_XX
replace S_g_het_XX=-1 if S_g_het_XX==0
if "`predict_het_good'"!=""{
scalar heterogeneity_scalar_XX=1
}
}


///// If continuous option specified: binarizing and staggerizing treatment,
///// and adding time_FEs interacted with D_{g,1} as controls
if `continuous'>0{

//Binarizing and staggerizing treatment
capture drop treatment_XX_temp
gen treatment_XX_temp = (F_g_XX<=time_XX)*S_g_het_XX if S_g_het_XX!=.
capture drop treatment_XX_orig
gen treatment_XX_orig=treatment_XX
replace treatment_XX = treatment_XX_temp 

//Enriching controls
tab time_XX, gen(time_fe_XX_)
local max_time_XX=r(r)
// Modif Clément: actually, to compute the estimator with the continuous option we need to control for polynomial in D_{g,1} interacted with \sum_{k=1}^t'1\{t=k}, sum of period FEs up to t'.
forvalue i=2/`max_time_XX'{
replace time_fe_XX_`i'=(time_XX>=`i')
}
// End modif Clément
foreach var of varlist time_fe_XX_2-time_fe_XX_`max_time_XX'{
	forvalues pol_level = 1/`=degree_pol'{
		gen `var'_bt`pol_level'_XX = `var'*d_sq_`pol_level'_XX
		local controls "`controls' `var'_bt`pol_level'_XX"
	}
	capture drop `var'
}
capture drop time_fe_XX_1
}

///// Creating treatment at F_g: D_{g,F_g}
gen d_fg_XX_temp=treatment_XX if time_XX==F_g_XX
bys group_XX: gegen d_fg_XX=mean(d_fg_XX_temp)
replace d_fg_XX=d_sq_XX if d_fg_XX==. & F_g_XX==T_max_XX+1
drop d_fg_XX_temp 

///// Creating the variable L_g_XX = T_g_XX - F_g_XX, so that we can compute L_u or L_a afterwards.
gen L_g_XX = T_g_XX - F_g_XX +1

///// Creating the equivalent variable L_g_placebo_XX for the placebos.
if `placebo'!=0{
capture drop L_g_placebo_XX
gen L_g_placebo_XX = min(L_g_XX, F_g_XX-2) if F_g_XX>=3
}

///// Tagging first observation of each group_XX.
sort group_XX time_XX
bysort group_XX : gen first_obs_by_gp_XX = (_n==1)

///// If cluster option specified, tagging first observation of each cluster, 
///// and checking clustering variable weakly coarser than group variable.

if "`cluster'"!=""{
		
	// FELIX 06-03-25 -> I guess this is how Tom solved it 
	cap drop cluster_group_XX
	bysort group_XX: gegen cluster_group_XX=min(`cluster') 
	bysort group_XX: replace `cluster'=cluster_group_XX if `cluster'==.
	// Can do this because clusters have to be coarser than groups
	
	capture drop cluster_var_g_XX
	
	bysort `cluster' (group_XX time_XX) : gen first_obs_by_clust_XX = (_n==1) if `cluster'!=. // FELIX 05_03_25 -> This is then not needed anymore
	
	//Error message for clustering: non-nested case
	bysort group_XX: gegen cluster_var_g_XX = sd(`cluster')
	sum cluster_var_g_XX
	scalar max_cluster_var_XX = `r(max)'
	if (scalar(max_cluster_var_XX) >0){
	di as error ""
	di as error "The group variable should be nested within the clustering variable."
	di as input _continue ""
	exit
	}
	
}

///// Declaring data as panel //
xtset group_XX time_XX

///// Creating first-differenced outcome and treatment//
capture drop diff_y_XX
capture drop diff_d_XX
gen diff_y_XX = d.outcome_XX
g diff_d_XX = d.treatment_XX

////////// 3. Necessary pre-estimation steps when the controls option is specified.	

if "`controls'" !=""{
local count_controls=0

capture drop fd_X_all_non_missing_XX
gen fd_X_all_non_missing_XX=1

foreach var of varlist `controls'{

local count_controls=`count_controls'+1

capture drop diff_X`count_controls'_XX
capture drop avg_diff_X`count_controls'_XX
capture drop resid_X`count_controls'_time_FE_XX

///// Computing the first differences of the control variables

xtset group_XX time_XX
gen diff_X`count_controls'_XX=D.`var'
replace fd_X_all_non_missing_XX=0 if diff_X`count_controls'_XX==.
}

///// Residualization, and computation of term entering variance. 

local count_controls=0
local mycontrols_XX ""
local prod_controls_y ""

foreach var of varlist `controls'{
local count_controls=`count_controls'+1

capture drop sum_weights_control_XX //So as to consider the weighted regressions.

// Computing  \Delta X_{.,t}: average of controls' first difference at time t, 
// among groups whose treatment has not changed yet.

bys time_XX d_sq_XX `trends_nonparam' : gegen sum_weights_control_XX = total(N_gt_XX) if ever_change_d_XX==0&diff_y_XX!=.&fd_X_all_non_missing_XX==1
bys time_XX d_sq_XX `trends_nonparam' : gegen avg_diff_X`count_controls'_XX = total(N_gt_XX*diff_X`count_controls'_XX) if ever_change_d_XX==0&diff_y_XX!=.&fd_X_all_non_missing_XX==1

bys time_XX d_sq_XX `trends_nonparam' : replace avg_diff_X`count_controls'_XX = avg_diff_X`count_controls'_XX/sum_weights_control_XX

// Computing \Delta\Dot{X}_{g,t}, the difference between the first differences 
// of covariates and the average of their first-difference, which gives us 
// the residuals of a regression of covariates on time fixed effects. 
//Multiply by sqrt(N_gt_XX) to replicate weighted regression.

gen resid_X`count_controls'_time_FE_XX = sqrt(N_gt_XX)*(diff_X`count_controls'_XX - avg_diff_X`count_controls'_XX)

replace resid_X`count_controls'_time_FE_XX=0 if resid_X`count_controls'_time_FE_XX==.

// Storing the obtained residuals for the computation of theta_d
local mycontrols_XX "`mycontrols_XX' resid_X`count_controls'_time_FE_XX"

// Generating the product between \Delta\Dot{X}_{g,t} and \Delta Y_{g,t}
//Multiply by sqrt(N_gt_XX) to replicate weighted regression
capture drop prod_X`count_controls'_Ngt_XX
// Delete those we do not need anymore
capture drop prod_X`count_controls'_diff_y_temp_XX
capture drop prod_X`count_controls'_diff_y_XX
capture drop diff_y_wXX
gen diff_y_wXX = sqrt(N_gt_XX)*diff_y_XX

* Note: resid_X`count_controls'_time_FE_XX is already multiplied by sqrt(N_gt_XX)
gen prod_X`count_controls'_Ngt_XX=resid_X`count_controls'_time_FE_XX*sqrt(N_gt_XX)
replace prod_X`count_controls'_Ngt_XX=0 if prod_X`count_controls'_Ngt_XX==. 

}

///// Computing the Den_d matrices and their inverts,
///// and creating locals storing the status quos for which Den_d^{-1} not defined. 

local store_singular_XX ""
local store_noresidualization_XX ""
local levels_d_sq_XX_final ""

levelsof d_sq_int_XX, local(levels_d_sq_XX)

foreach l of local levels_d_sq_XX {
	
capture drop E_y_hat_gt_int_`l'_XX		

// Running residualization regression to compute predicted values

local controlsXX ""
forv i = 1/`count_controls'{
	local controlsXX "`controlsXX' diff_X`i'_XX"
}

*Modif Clément 30/6/2025
*cap reg diff_y_XX `controlsXX' ibn.time_XX [aw=N_gt_XX] if d_sq_int_XX==`l'&time_XX<F_g_XX, noconst
if "`trends_nonparam'" == "" {
cap reg diff_y_XX `controlsXX' ibn.time_XX [aw=N_gt_XX] if d_sq_int_XX==`l'&time_XX<F_g_XX, noconst
}
if "`trends_nonparam'" != "" {
capture drop trends_nonparam_temp_XX
gegen trends_nonparam_temp_XX=group(`trends_nonparam')	
cap reg diff_y_XX `controlsXX' ibn.time_XX#ibn.trends_nonparam_temp_XX [aw=N_gt_XX] if d_sq_int_XX==`l'&time_XX<F_g_XX, noconst
}
*End Modif Clément 19/6/2025

if (_rc == 0) {
predict E_y_hat_gt_int_`l'_XX if d_sq_int_XX==`l'&time_XX<F_g_XX
*Modif Clément 19/6/2025
capture drop trends_nonparam_temp_XX
*End Modif Clément 19/6/2025
	tempfile data_XX
	save "`data_XX'.dta", replace
	
// A baseline treatment is relevant iff it is taken by at least two groups with different values of F_g_XX
// and non-missing diff_y_XX, otherwise we do not need to perform the residualization for this specific baseline treatment.

scalar store_singular_`l'_XX = 0 
tab F_g_XX if d_sq_int_XX==`l'  

scalar useful_res_`l'_XX = `r(r)'
	if (scalar(useful_res_`l'_XX)>1){

// Isolate the observations used in the computation of theta_d

keep if ever_change_d_XX==0&diff_y_XX!=.&fd_X_all_non_missing_XX==1&d_sq_int_XX==`l'

// Using the matrix accum function, to regress the first difference of outcome on the first differences of covariates. We will obtain the vectors of coefficients \theta_d s, where d indexes values of the baseline treatment.

capture matrix accum overall_XX = diff_y_wXX `mycontrols_XX'
scalar rc_XX=_rc

if scalar(rc_XX)!=0{
scalar store_singular_`l'_XX = 1
local store_noresidualization_XX "`store_noresidualization_XX' `l'"
scalar useful_res_`l'_XX=1
}

else {
	
// Isolate the parts of the matrix we need
matrix didmgt_XX = overall_XX[2..`=`count_controls'+1',2..`=`count_controls'+1']
matrix didmgt_Xy = overall_XX[2..`=`count_controls'+1',1]

// Computing the vectors of coefficients \theta_d s for each value of the baseline treatment
matrix coefs_sq_`l'_XX = invsym(didmgt_XX)*didmgt_Xy
local levels_d_sq_XX_final "`levels_d_sq_XX_final' `l'" // Added local so as to avoid adding to the final block non-existent coefficients.

		// Computing the matrix Den^{-1}_d
			//Check first if the matrix is invertible, invsym() inverts a matrix even if it is singular
			capture drop scalar det_XX
			scalar det_XX = det(didmgt_XX)

		if (abs(scalar(det_XX))<=10^(-16)){ 
					scalar store_singular_`l'_XX = 1
					scalar drop det_XX
		}

		// Changes Diego 27-03-25: N_c_`l'_XX adjustment
		sum F_g_XX
		gen N_c_`l'_temp_XX = time_XX >= 2 & time_XX <= r(max) - 1 & time_XX < F_g_XX & diff_y_XX < .
		sum N_gt_XX if N_c_`l'_temp_XX == 1

		matrix inv_Denom_`l'_XX = invsym(didmgt_XX)*r(sum)*G_XX
		drop N_c_`l'_temp_XX
}

use "`data_XX'.dta", clear

}
}
else {
*Modif Clément 19/6/2025
capture drop trends_nonparam_temp_XX
*End Modif Clément 19/6/2025	
    drop if d_sq_int_XX == `l'
    noi di "Baseline Treatment Level `l' dropped because of insufficient observations."
}

}

//Fill up store_singular_XX, with correct values of statu quo and not the levels -> Modif Felix
levelsof d_sq_XX, local(levels_d_sq_bis_XX)
scalar index_sing_XX = 0
foreach l of local levels_d_sq_bis_XX {
	scalar index_sing_XX = scalar(index_sing_XX)+1
	capture confirm scalar store_singular_`=index_sing_XX'_XX
	if _rc==0{
		if(scalar(store_singular_`=index_sing_XX'_XX) == 1){
		local store_singular_XX = "`store_singular_XX' `l'"
		}
	}
}

///// Display errors if one of the Den_d^{-1} is not defined
if ("`store_singular_XX'"!=""){
	
	di as error "Some control variables are not taken into account for groups with baseline treatment equal to: [`store_singular_XX']"
	di as error "This may occur in the following situations:"
	di as error "1. For groups with those values of the baseline treatment, the regression of the outcome first difference on the controls' first differences and time fixed effects has fewer observations than variables. Note that for each value of the baseline treatment, those regressions are estimated among (g,t)s such that g has not changed treatment yet at t."
	di as error "2. For groups with those values of the baseline treatment, two or more of your control variables are perfectly collinear in the sample where the regression is run, for instance because those control variables do not vary over time."
	di as input _continue ""
}

///// Values of baseline treatment such that residualization could not be performed at all are dropped.
foreach l of local store_noresidualization_XX {
drop if d_sq_int_XX==`l'
}

} // end of the if "`controls'" !="" condition

////////// 4. Performing the estimation and storing the results

///// Computing L_u/L_a, maximum number of event-study effects that can be computed
///// for the switchers in/out, to compare them to number of effects requested,
///// and finally determine the number of effects to be estimated.
///// Same thing for the placebos.

// Initializing L_u_XX and L_a_XX with default values.
scalar L_u_XX=.
scalar L_a_XX=.

// For switchers in
if "`switchers'"==""|"`switchers'"=="in"{ 
sum L_g_XX if S_g_XX==1
scalar L_u_XX=r(max)
*For placebos
if `placebo'!=0{
sum L_g_placebo_XX if S_g_XX==1
scalar L_placebo_u_XX=r(max)

// If the trends_lin option was specified, L_placebo_u_XX should be decreased by 1
// because data starts at period 2 instead of 1.
if "`trends_lin'"!=""{
	scalar L_placebo_u_XX=L_placebo_u_XX-1
}
if L_placebo_u_XX==. {
    scalar L_placebo_u_XX = 0
}
}
}

// For switchers out
if "`switchers'"==""|"`switchers'"=="out"{
sum L_g_XX if S_g_XX==0
scalar L_a_XX=r(max)
*For placebos
if `placebo'!=0{
sum L_g_placebo_XX if S_g_XX==0
scalar L_placebo_a_XX=r(max)
if "`trends_lin'"!=""{
	scalar L_placebo_a_XX=L_placebo_a_XX-1
}
if L_placebo_a_XX==. {
    scalar L_placebo_a_XX = 0
}
}
}

// Error message if Design Restriction 1 is not met

if ("`switchers'"=="in"&(L_u_XX==.|L_u_XX==0))|("`switchers'"=="out"&(L_a_XX==.|L_a_XX==0))|("`switchers'"==""&(L_u_XX==.|L_u_XX==0)&(L_a_XX==.|L_a_XX==0)){
	di as error ""
	di as error "No treatment effect can be estimated. This is because Design Restriction 1 in de Chaisemartin & D'Haultfoeuille (2024) is not satisfied in the data, given the options requested. This may be due to the fact that groups' period-one treatment is continuous, or takes a large number of values, and you have not specified the continuous option. If so, you can try to specify this option. If the issue persists even with this option, this means that all groups experience their first treatment change at the same date, a situation where the estimators of de Chaisemartin & D'Haultfoeuille (2024) cannot be used."
	di as input _continue ""
		
	exit
}

// Checking that the number of dynamic and placebo effects requested by user
// are feasible, and correcting them if they are not. 

if "`switchers'"==""{
scalar l_XX=max(L_a_XX, L_u_XX)
scalar l_XX=min(l_XX, `effects')

if `placebo'!=0{
scalar l_placebo_XX=max(L_placebo_a_XX, L_placebo_u_XX)
scalar l_placebo_XX=min(l_placebo_XX, `placebo')
// The number of placebos computed cannot be greater than the number of effects computed:
scalar l_placebo_XX=min(l_placebo_XX, `effects')
}
else{
scalar l_placebo_XX=0
}
}

if "`switchers'"=="in"{
	scalar l_XX=min(`effects', L_u_XX)
	if `placebo'!=0{
		scalar l_placebo_XX=min(`placebo', L_placebo_u_XX)
		// The number of placebos computed cannot be greater than the number of effects computed:
		scalar l_placebo_XX=min(l_placebo_XX, `effects') 
		}
else{
	scalar l_placebo_XX=0
}
}

if "`switchers'"=="out"{
	scalar l_XX=min(`effects', L_a_XX)
		if `placebo'!=0{
			scalar l_placebo_XX=min(`placebo', L_placebo_a_XX)
			// The number of placebos computed cannot be greater than the number of effects computed:
			scalar l_placebo_XX=min(l_placebo_XX, `effects')
		}
else{
	scalar l_placebo_XX=0
}
}

//If the number of effects or placebos initially asked by user was too large, 
// display an error message.
if l_XX<`effects'{
	di as error ""
	di as error "The number of effects requested is too large. The number of effects which can be estimated is at most " l_XX ". The command will therefore try to estimate " l_XX " effect(s)."
	di as input _continue ""
}

if `placebo'!=0{
	if l_placebo_XX<`placebo'&`effects'>=`placebo'{
		di as error ""
		di as error "The number of placebos which can be estimated is at most " l_placebo_XX ". The command will therefore try to estimate " l_placebo_XX " placebo(s)."
		di as input _continue ""
	}

	if `effects'<`placebo'{
		di as error ""
		di as error "The number of placebo requested cannot be larger than the number of effects requested. The command cannot compute more than " l_placebo_XX " placebo(s)."
		di as input _continue ""
	}
}

// Adjustment to add more placebos (did_multiplegt_dyn_all_pl)
local max_pl_u_XX = 0
local max_pl_a_XX = 0
local max_pl_gap_u_XX = 0
local max_pl_gap_a_XX = 0
gen pl_gap_XX = (F_g_XX - 2) - L_g_XX if S_g_XX != .
if ("`switchers'" == "" | "`switchers'" == "in") {
	sum F_g_XX if S_g_XX == 1
	local max_pl_u_XX = r(max) - 2
	sum pl_gap_XX if S_g_XX == 1
	local max_pl_gap_u_XX = r(max)
}
if ("`switchers'" == "" | "`switchers'" == "out") {
	sum F_g_XX if S_g_XX == 0
	local max_pl_a_XX = r(max) - 2
	sum pl_gap_XX if S_g_XX == 0
	local max_pl_gap_a_XX = r(max)
}
scalar max_pl_XX = max(`max_pl_u_XX', `max_pl_a_XX')
scalar max_pl_gap_XX = max(`max_pl_gap_u_XX', `max_pl_gap_a_XX')
drop pl_gap_XX

///// Start the by_path option here -> after defining l_XX etc. but before calling the core program
if "`by_path'"!=""{

// Disentangle all the treatment paths and match switchers with those paths with not yet switchers with the same baseline that can be used as controls
forvalues k=0/`=l_XX'{
	gen dummy_treat_`k'_temp=treatment_XX if time_XX==F_g_XX-1+`k'
	gegen dummy_treat_`k'=max(dummy_treat_`k'_temp), by(group_XX)
	capture drop dummy_treat_`k'_temp
}
// Variable grouping all groups that follow exactly the same treatment path from F_g-1 to F_g-1+\ell
gegen different_paths_temp_XX=group(dummy_treat_*) if time_XX<=F_g_XX-1+`=l_XX'
gegen different_paths_XX=max(different_paths_temp_XX), by(group_XX)

// Modify for never switchers
replace different_paths_XX=. if F_g_XX==T_g_XX+1
sum different_paths_XX
local num_paths_XX=r(max) // count number of different switchers paths
// Issue, controls can be controls for multiple treatment paths (as long as they have the same baseline treatment)
// And as not yet switchers are also controls there will be some groups for which less than the l_XX effects will be estimated -> specify same_switchers option

if "`by_path'"=="all"{ // case where we dont specify a numeric value
	local by_path=`num_paths_XX'
}

// Warnings
if `by_path'>`num_paths_XX'{
	di ""
	di as error "You specified a number that exceeds the number of treatment paths in the by_path() option! The number of paths is automatically set to the number of treatment paths in the sample."
	di as input _continue ""
	
	local by_path=`num_paths_XX'
}

// count number of groups within each path and sort the numbers accordingly in ascending order (1 -> most, 2 -> second most etc...)
if `by_path'<`num_paths_XX'{
gen count_path_XX=. 
forvalues k=1/`num_paths_XX'{
	count if different_paths_XX==`k'&first_obs_by_gp_XX==1 // important to count each group just once in case of unbalanced panel!
	replace count_path_XX=r(N) if different_paths_XX==`k' 
}

gegen rank_paths_XX=group(-count_path_XX different_paths_XX) if count_path_XX!=.
replace different_paths_XX=rank_paths_XX
}


// prevent \ell to be overwritten, should be checked each repetition again 
	scalar l_XX_orig=scalar(l_XX)
	scalar l_placebo_XX_orig=scalar(l_placebo_XX)

// create dummies which state for which path you can act as control depending on the same F_g-1 and not having switched (yet) at F_g_XX-1+l_XX
global graph_by_path ""
forvalues k=1/`by_path'{
	sum d_sq_XX if different_paths_XX==`k'
	local path_d_sq_XX=r(mean)
	sum F_g_XX if different_paths_XX==`k'
	local path_min_Fg_XX=r(min)
	
//  keep everyone prior to switch (and same baseline)
	capture drop cont_path_alt_XX
	gen cont_path_alt_XX=(time_XX<F_g_XX&d_sq_XX==`path_d_sq_XX')
	
// re-initialize l_XX in case it got changed in one repetition 
	scalar l_XX=scalar(l_XX_orig)
	scalar l_placebo_XX=scalar(l_placebo_XX_orig)	
	
// generate a global which stores the number of groups in each path
	count if different_paths_XX==`k'&first_obs_by_gp_XX==1
	global num_g_path_`k'_XX=r(N)	
	
// specify a local with the treatment path to display in Matrix
	global k=`k'
	
	local path_`k'_XX ""
	forvalues j=0/`=l_XX'{
		sum treatment_XX if time_XX==F_g_XX-1+`j'&different_paths_XX==`k'
		local inp_XX=r(mean)
		if `j'==0{
		local path_`k'_XX "`path_`k'_XX'`inp_XX'"
		}
		else{
		local path_`k'_XX "`path_`k'_XX', `inp_XX'"	
		}
	}
	global path_`k'_XX "`path_`k'_XX'"
	
	
global l_graph_XX=`=l_XX'		
global l_placebo_graph_XX=`=-l_placebo_XX'	

capture graph drop graph_`k'_XX

// call did_multiplegt_dyn with the corresponding options // Modif Felix 26.05.2025 -> change the weight to the tempvar
noisily did_multiplegt_dyn `1' `2' `3' `4' if (different_paths_XX==`k'|cont_path_alt_XX==1), effects(`=l_XX') placebo(`=l_placebo_XX') same_switchers switchers(`switchers') `only_never_switchers' controls(`controls') trends_nonparam(`trends_nonparam') weight(`weight_path_XX') `dont_drop_larger_lower' `normalized' cluster(`cluster') `same_switchers_pl' `trends_lin' by(`by_var_path') predict_het(`predict_het') ci_level(`ci_level') design(`design') date_first_switch(`date_first_switch') continuous(`continuous') `less_conservative_se' `normalized_weights' `graph_off' `save_sample' graphoptions(xlabel(`=-l_placebo_XX'[1]`=l_XX') title(Treatment path (${path_`k'_XX}); ${num_g_path_`k'_XX} switchers, size(small)) xtitle(Relative time to last period before treatment changes (t=0), size(small)) graphregion(color(white)) plotregion(color(white)) legend(pos(6) order(`graph_options_int') rows(1) size(small)) name(graph_`k'_XX) legend(off))

// Save global with all the graph names to do the combine 
global graph_by_path "$graph_by_path graph_`k'_XX"
}

if "`graph_off'"==""{
// check if all graphs are created
graph dir 
local in_memory_XX=r(list)	
	
	local count_graphs_XX=0
	foreach b of global graph_by_path{ 
    if `: list b in in_memory_XX' { 
         local count_graphs_XX=`count_graphs_XX'
    } 
    else{
		local count_graphs_XX=`count_graphs_XX'+1
	} 
 } 

// combine the graphs if all of them could be produced 
if `count_graphs_XX'==0{
graph combine $graph_by_path, title(DID from last period before treatment changes (t=0) to t, size(medium)) ycom
}
else{
	di as error ""
	di as error "Estimation of at least one treatment path was not possible, therefore no combined graph is produced."
	di as input _continue ""
}
}

// Reset the global so we do not always see path in table 
global k ""

exit // end the command here for now -> do not run "the full sample"
} // end "by_path" option


///// Generating default values for the variables which will be aggregated 
///// after Program 2 below has been run for switchers in and for switchers out.

forvalue i=1/`=l_XX'{
	
	capture drop U_Gg`i'_plus_XX 
	capture drop U_Gg`i'_minus_XX
	capture drop count`i'_plus_XX
	capture drop count`i'_minus_XX
	capture drop U_Gg`i'_global_XX
	capture drop count`i'_global_XX
	capture drop N_effect_`i'_XX
	capture drop DID_`i'_XX
	capture drop U_Gg_var_`i'_in_XX
	capture drop U_Gg_var_`i'_out_XX
	capture drop U_Gg_var_glob_`i'_XX
	capture drop U_Gg_var_glob_eff`i'_sqrd_XX
	capture drop delta_D_g_`i'_plus_XX 
	capture drop delta_D_g_`i'_minus_XX 
	
	gen U_Gg`i'_plus_XX = 0
	gen U_Gg`i'_minus_XX = 0
	gen count`i'_plus_XX = 0
	gen count`i'_minus_XX = 0
	scalar N1_`i'_XX=0
	scalar N0_`i'_XX=0
	gen U_Gg_var_`i'_in_XX=0
	gen U_Gg_var_`i'_out_XX=0
	scalar sum_for_var_in_XX=0
	scalar sum_for_var_out_XX=0
	gen delta_D_g_`i'_plus_XX=0 
	gen delta_D_g_`i'_minus_XX=0 
	
	scalar N0_`i'_XX_new=0
	scalar N1_`i'_XX_new=0
	
	if "`weight'"!=""{
		scalar N0_`i'_XX_weight=0
		scalar N1_`i'_XX_weight=0
	}
	
	if "`normalized'"!=""{
		scalar delta_D_`i'_in_XX = 0
		scalar delta_D_`i'_out_XX = 0
	}
}

if l_placebo_XX!=0{

	forvalue i=1/`=l_placebo_XX'{	
	
	capture drop U_Gg_pl_`i'_plus_XX  
	capture drop U_Gg_pl_`i'_global_XX
	capture drop U_Gg_pl_`i'_minus_XX 
	capture drop DID_placebo_`i'_XX
	capture drop N_placebo_`i'_XX
	capture drop U_Gg_var_pl_`i'_in_XX
	capture drop U_Gg_var_pl_`i'_out_XX
	capture drop U_Gg_var_glob_pl_`i'_XX
	capture drop U_Gg_var_glob_pl_`i'_2_XX
	capture drop count`i'_pl_plus_XX
	capture drop  count`i'_pl_minus_XX
	capture drop count`i'_pl_global_XX
	
	gen U_Gg_pl_`i'_plus_XX   = 0
	gen U_Gg_pl_`i'_minus_XX   = 0
	scalar N1_placebo_`i'_XX=0
	scalar N0_placebo_`i'_XX=0
	gen U_Gg_var_pl_`i'_in_XX=0
	gen U_Gg_var_pl_`i'_out_XX=0
	scalar sum_for_var_placebo_in_XX=0
	scalar sum_for_var_placebo_out_XX=0
	gen count`i'_pl_plus_XX = 0
	gen count`i'_pl_minus_XX = 0
	
	
	scalar N0_placebo_`i'_XX_new=0
	scalar N1_placebo_`i'_XX_new=0
	
	if "`weight'"!=""{
		scalar N0_`i'_pl_XX_weight=0
		scalar N1_`i'_pl_XX_weight=0
	}
	
	if "`normalized'"!=""{
		scalar delta_D_pl_`i'_in_XX = 0
		scalar delta_D_pl_`i'_out_XX = 0
	}
	
}
}

gen U_Gg_plus_XX = 0
gen U_Gg_minus_XX = 0
scalar U_Gg_den_plus_XX = 0
scalar U_Gg_den_minus_XX = 0
scalar sum_N1_l_XX = 0
scalar sum_N0_l_XX = 0
gen U_Gg_var_plus_XX = 0
gen U_Gg_var_minus_XX = 0

// Initialize switchers tag variable to earmark switchers after did_multiplegt_dyn_core 
capture drop switcher_tag_XX
gen switcher_tag_XX = .

///// Perform the estimation: call the program did_multiplegt_dyn_core, 
///// for switchers in and for switchers out, and store the results.

// Modif. Diego: tailor the upper bounds of the switcher_tag loops to the actual number of post periods requested/feasible for switchers-in and switchers-out
scalar L_u_XX_tag = min(L_u_XX, `effects')
scalar L_a_XX_tag = min(L_a_XX, `effects')

// For switchers in
if ("`switchers'"==""|"`switchers'"=="in"){
if L_u_XX!=.&L_u_XX!=0{

***** MODIF FELIX: Replaced all the weights by weight(weight_XX) instead of weight(`weight') due to the fact that `weights' does not exist anymore after collapsing

* Perform the estimation of effects and placebos outside of the loop on 
* number of effects if trends_lin not specified
if "`trends_lin'"==""{
	did_multiplegt_dyn_core_new outcome_XX group_XX time_XX treatment_XX, effects(`=l_XX') placebo(`=l_placebo_XX') switchers_core(in) `only_never_switchers' controls(`controls') trends_nonparam(`trends_nonparam') `normalized' `same_switchers' `same_switchers_pl' continuous(`continuous') `less_conservative_se' weight(weight_XX) cluster(`cluster')

	// Store the number of the event-study effect for switchers-in
		forv k = 1/`=L_u_XX_tag' {
			replace switcher_tag_XX = `k' if distance_to_switch_`k'_XX == 1
		}
}	
	
forvalue i=1/`=l_XX'{

* Perform the estimation of effects inside of the loop on number of effects 
* if trends_lin is specified
* Note that if the option trends_lin was specified, same_switchers must also be specified.
if "`trends_lin'"!=""{
	did_multiplegt_dyn_core_new outcome_XX group_XX time_XX treatment_XX, effects(`i') switchers_core(in) `only_never_switchers' controls(`controls') trends_nonparam(`trends_nonparam') `normalized' same_switchers  trends_lin continuous(`continuous') `less_conservative_se' weight(weight_XX) cluster(`cluster')

	// Store the number of the event-study effect for switchers-in
	replace switcher_tag_XX = `i' if distance_to_switch_`i'_XX == 1
	}
* Store variables necessary for computation of effects.
* N.B.: in the case of unbalanced panels, it can happen that the U_Gg`i'_XX are not computed by program 2 (for example when y is missing). Consequently, for the command not to display an error message and continue running, we need to verify the variable is created, which is conditional on  N1_`i'_XX!=0.
if N1_`i'_XX!=0{
		replace U_Gg`i'_plus_XX = U_Gg`i'_XX
		replace count`i'_plus_XX= count`i'_core_XX
		replace U_Gg_var_`i'_in_XX=U_Gg`i'_var_XX
		scalar N1_`i'_XX_new=N1_`i'_XX

		if "`normalized'"!=""{
			scalar delta_D_`i'_in_XX = delta_norm_`i'_XX
		}
		
		if "`trends_lin'"==""{
			replace delta_D_g_`i'_plus_XX = delta_D_g_`i'_XX
		}
	}
}

*Same as above, for placebos.
if l_placebo_XX!=0{

	forvalue i=1/`=l_placebo_XX'{
		
		if "`trends_lin'"!=""{
	did_multiplegt_dyn_core_new outcome_XX group_XX time_XX treatment_XX, effects(`i') placebo(`i') switchers_core(in) `only_never_switchers' controls(`controls') trends_nonparam(`trends_nonparam') `normalized' same_switchers same_switchers_pl trends_lin continuous(`continuous') `less_conservative_se' weight(weight_XX) cluster(`cluster')
}
				
		if N1_placebo_`i'_XX!=0{
				replace U_Gg_pl_`i'_plus_XX  = U_Gg_placebo_`i'_XX
				replace count`i'_pl_plus_XX= count`i'_pl_core_XX
				replace U_Gg_var_pl_`i'_in_XX=U_Gg_pl_`i'_var_XX
				scalar N1_placebo_`i'_XX_new=N1_placebo_`i'_XX
				
				if "`normalized'"!=""{
					scalar delta_D_pl_`i'_in_XX = delta_norm_pl_`i'_XX
				}

		}
		
	}	
}

* Store variables necessary for computation of average effect.
	if "`trends_lin'"==""{
	if sum_N1_l_XX!=0{
	replace U_Gg_plus_XX = U_Gg_XX
	scalar U_Gg_den_plus_XX=U_Gg_den_XX
	replace U_Gg_var_plus_XX = U_Gg_var_XX
	}
	}

}
	
} // end of the condition for switchers in

// Same thing as above, for switchers out

if ("`switchers'"==""|"`switchers'"=="out"){
if L_a_XX!=.&L_a_XX!=0{
	
if "`trends_lin'"==""{	
did_multiplegt_dyn_core_new outcome_XX group_XX time_XX treatment_XX, effects(`=l_XX') placebo(`=l_placebo_XX') switchers_core(out) `only_never_switchers' controls(`controls')  trends_nonparam(`trends_nonparam') `normalized' `same_switchers' `same_switchers_pl' continuous(`continuous') `less_conservative_se' weight(weight_XX) cluster(`cluster')

	// Store the number of the event-study effect for switchers-out
		forv k = 1/`=L_a_XX_tag' {
			replace switcher_tag_XX = `k' if distance_to_switch_`k'_XX == 1
		}

}
	
forvalue i=1/`=l_XX'{
	
if "`trends_lin'"!=""{	
	did_multiplegt_dyn_core_new outcome_XX group_XX time_XX treatment_XX, effects(`i') switchers_core(out) `only_never_switchers' controls(`controls') trends_nonparam(`trends_nonparam') `normalized' same_switchers trends_lin continuous(`continuous') `less_conservative_se' weight(weight_XX) cluster(`cluster')

	// Store the number of the event-study effect for switchers-out
	replace switcher_tag_XX = `i' if distance_to_switch_`i'_XX == 1
}
	
if N0_`i'_XX!=0{
		replace U_Gg`i'_minus_XX = - U_Gg`i'_XX
		replace count`i'_minus_XX= count`i'_core_XX
		//// CHANGE BELOW - tks (add minus sign)
		replace U_Gg_var_`i'_out_XX= - U_Gg`i'_var_XX
		scalar N0_`i'_XX_new=N0_`i'_XX
		
		if "`normalized'"!=""{
			scalar delta_D_`i'_out_XX = delta_norm_`i'_XX
		}
		
		if "`trends_lin'"==""{
		replace delta_D_g_`i'_minus_XX = delta_D_g_`i'_XX
		}
	}
	
}


if l_placebo_XX!=0{

	forvalue i=1/`=l_placebo_XX'{
		
if "`trends_lin'"!=""{	
	did_multiplegt_dyn_core_new outcome_XX group_XX time_XX treatment_XX, effects(`i') placebo(`i') switchers_core(out) `only_never_switchers' controls(`controls') trends_nonparam(`trends_nonparam') `normalized' same_switchers same_switchers_pl trends_lin continuous(`continuous') `less_conservative_se' weight(weight_XX) cluster(`cluster')
}
		
	if N0_placebo_`i'_XX!=0{
			replace U_Gg_pl_`i'_minus_XX  = -U_Gg_placebo_`i'_XX
			replace count`i'_pl_minus_XX= count`i'_pl_core_XX
			//// CHANGE BELOW - tks (add minus sign)
			replace U_Gg_var_pl_`i'_out_XX= - U_Gg_pl_`i'_var_XX 
			scalar N0_placebo_`i'_XX_new=N0_placebo_`i'_XX
			
			if "`normalized'"!=""{
					scalar delta_D_pl_`i'_out_XX = delta_norm_pl_`i'_XX
	}

	}
	}		
}

	if "`trends_lin'"==""{
	if sum_N0_l_XX!=0{
	replace U_Gg_minus_XX = - U_Gg_XX
	scalar U_Gg_den_minus_XX=U_Gg_den_XX
	replace U_Gg_var_minus_XX = - U_Gg_var_XX
	}
	}

}
	
} // end of the loop for switchers out 

scalar drop L_u_XX_tag L_a_XX_tag

////////// 5. Computing the estimators and their variances

/////  Computing DID_\ell

// Creation of the matrix which stores all the estimators (DID_l, DID_pl, delta, etc.), their sd and the CIs

matrix mat_res_XX = J(l_XX+l_placebo_XX+1,7,.) 

// Loop over the number of effects to be estimated

	// Create a local to store effects we cant estimate 
	local missing_eff_XX ""	

forvalue i=1/`=l_XX'{

// Aggregating the U_(g,l) for switchers in and out
	
gen U_Gg`i'_global_XX = (N1_`i'_XX_new/(N1_`i'_XX_new+N0_`i'_XX_new))*U_Gg`i'_plus_XX +(N0_`i'_XX_new/(N1_`i'_XX_new+N0_`i'_XX_new))*U_Gg`i'_minus_XX
replace U_Gg`i'_global_XX=. if first_obs_by_gp_XX==0

gen count`i'_global_XX = max(count`i'_plus_XX, count`i'_minus_XX)

// Computing aggregated delta_D (difference between treatments received wrt status quo, from F_g-1 to F_g-1+\ell), only needed for the normalized estimator 
	if "`normalized'"!=""{
	scalar delta_D_`i'_global_XX = (N1_`i'_XX_new/(N1_`i'_XX_new+N0_`i'_XX_new))*delta_D_`i'_in_XX+(N0_`i'_XX_new/(N1_`i'_XX_new+N0_`i'_XX_new))*delta_D_`i'_out_XX
	}

// Storing the results and the number of switchers into the matrix mat_res_XX
local rownames "`rownames' Effect_`i'"

// Counting number of switchers DID_\ell applies to

scalar N_switchers_effect_`i'_XX=N1_`i'_XX_new+N0_`i'_XX_new
matrix mat_res_XX[`i',6]=N_switchers_effect_`i'_XX
matrix mat_res_XX[`i',7]=`i'
ereturn scalar N_switchers_effect_`i' = N_switchers_effect_`i'_XX

// Counting number of observations used in the computation of DID_\ell

gegen N_effect_`i'_XX = total(count`i'_global_XX) 
scalar N_effect_`i'_XX = N_effect_`i'_XX
ereturn scalar N_effect_`i' = N_effect_`i'_XX
matrix mat_res_XX[`i',5]=N_effect_`i'_XX

// Modif Felix weight -> Display correct N with weights
if "`weight'"!=""{
	matrix mat_res_XX[`i',6]=N1_`i'_XX_weight+N0_`i'_XX_weight
	ereturn scalar N_switchers_effect_`i'_w= N1_`i'_XX_weight+N0_`i'_XX_weight

	capture drop count`i'_global_XX_w
	capture drop N_effect_`i'_XX_w
	
	gen count`i'_global_XX_w=count`i'_global_XX/weight_XX
	gegen N_effect_`i'_XX_w = total(count`i'_global_XX_w) 
	scalar N_effect_`i'_XX_w = N_effect_`i'_XX_w
	ereturn scalar N_effect_`i'_w = N_effect_`i'_XX_w
	matrix mat_res_XX[`i',5]=N_effect_`i'_XX_w
}

// Error message if DID_\ell cannot be estimated

if N_switchers_effect_`i'_XX==0|N_effect_`i'_XX==0{
	di as error ""
	di as error "Effect_"`i' " cannot be estimated."
	di as error "There is no switcher or no control"
	di as error "for this effect."
	di as input _continue ""
	
	local missing_eff_XX `missing_eff_XX' `i'
}

// Averaging the U_Gg\ell to compute DID_\ell
 
gegen DID_`i'_XX = total(U_Gg`i'_global_XX) 
replace  DID_`i'_XX = DID_`i'_XX/G_XX
scalar DID_`i'_XX = DID_`i'_XX

// Computing DID_n_l"  if the option was specified

if "`normalized'"!=""{
	scalar DID_`i'_XX = scalar(DID_`i'_XX)/scalar(delta_D_`i'_global_XX)
}

// Set DID_\ell missing when there is no switcher or no control

if ("`switchers'"==""&N1_`i'_XX_new==0&N0_`i'_XX_new==0)|("`switchers'"=="out"&N0_`i'_XX_new==0)|("`switchers'"=="in"&N1_`i'_XX_new==0){
	scalar DID_`i'_XX=.
}

// Store DID_\ell estimates in the ereturn and in the results matrix which will be printed out

ereturn scalar Effect_`i' = scalar(DID_`i'_XX)
matrix mat_res_XX[`i',1]= scalar(DID_`i'_XX) 

}

///// Computing the average total effect

// The average effect cannot be estimated when the trends_lin option is specified so the whole part will be skipped in that case

if "`trends_lin'"==""{

// Computing the weight w_+.

if "`switchers'"==""{
	scalar w_plus_XX = U_Gg_den_plus_XX*sum_N1_l_XX/(U_Gg_den_plus_XX*sum_N1_l_XX+U_Gg_den_minus_XX*sum_N0_l_XX)
}

* When the "switchers" option is used, full weight is put to either switchers out or in
if "`switchers'"=="out"{
	scalar w_plus_XX=0
}

if "`switchers'"=="in"{
	scalar w_plus_XX=1
}

// Aggregating the U_{G,g} for switchers in and out

gen U_Gg_global_XX = w_plus_XX*U_Gg_plus_XX +(1-w_plus_XX)*U_Gg_minus_XX
replace U_Gg_global_XX=. if first_obs_by_gp_XX==0

// Averaging the U_Gg to compute average total effect

gegen delta_XX = total(U_Gg_global_XX)
replace delta_XX = delta_XX/G_XX
scalar delta_XX = delta_XX
ereturn scalar Av_tot_effect = scalar(delta_XX)

// Completing the results matrix

* Storing the results
matrix mat_res_XX[l_XX+1,1]= scalar(delta_XX)
local rownames "`rownames' Av_tot_eff" 
* Number of switchers
scalar N_switchers_effect_XX=0
forvalue i=1/`=l_XX'{
	scalar N_switchers_effect_XX = N_switchers_effect_XX + N_switchers_effect_`i'_XX
}
matrix mat_res_XX[l_XX+1,6]=N_switchers_effect_XX
matrix mat_res_XX[l_XX+1,7]=0
ereturn scalar N_switchers_effect_average = N_switchers_effect_XX
 
* Number of observations used in the estimation
capture drop count_global_XX
capture drop N_effect_XX
gen count_global_XX=0
forvalue i=1/`=l_XX'{
	replace count_global_XX= max(count_global_XX, count`i'_global_XX)
}
sum count_global_XX
scalar N_effect_XX=r(sum)
matrix mat_res_XX[l_XX+1,5]=scalar(N_effect_XX)
ereturn scalar N_avg_total_effect = scalar(N_effect_XX)

}

// Modif Felix weight
if "`weight'"!=""{
	scalar N_switchers_effect_XX_w=0
	forvalue i=1/`=l_XX'{
		scalar N_switchers_effect_XX_w = N_switchers_effect_XX_w + N1_`i'_XX_weight+N0_`i'_XX_weight
	}
	
	matrix mat_res_XX[l_XX+1,6]=N_switchers_effect_XX_w
	ereturn scalar N_switchers_effect_average_w=N_switchers_effect_XX_w
	
	capture drop count_global_XX_w
	capture drop N_effect_XX_w
	gen count_global_XX_w=0
	forvalue i=1/`=l_XX'{
		replace count_global_XX_w= max(count_global_XX_w, count`i'_global_XX_w)
	}
	
	sum count_global_XX_w
	scalar N_effect_XX_w=r(sum)
	matrix mat_res_XX[l_XX+1,5]=scalar(N_effect_XX_w)
	ereturn scalar N_avg_total_effect_w = scalar(N_effect_XX_w)
}

// Add time_to_treat to the results matrix even when computation of average total effect skipped with trends_lin, important fot the graph later.
if "`trends_lin'"!=""{
	local rownames "`rownames' Av_tot_eff" 
	matrix mat_res_XX[l_XX+1,7]=0
}

// Generate string local for sotable
local str_effects "Effect_1"
if l_XX > 1 {
	forv j = 2/`=l_XX' {
		local str_effects "`str_effects' Effect_`j'"
	}
}
	
///// Computing the placebo estimators (same steps as for the DID_\ell, not commented)

if l_placebo_XX!=0{
	
	// Create a local to store effects we cant estimate 
	local missing_pl_XX ""

forvalue i=1/`=l_placebo_XX'{	

gen U_Gg_pl_`i'_global_XX = (N1_placebo_`i'_XX_new/(N1_placebo_`i'_XX_new+N0_placebo_`i'_XX_new))*U_Gg_pl_`i'_plus_XX  +(N0_placebo_`i'_XX_new/(N1_placebo_`i'_XX_new+N0_placebo_`i'_XX_new))*U_Gg_pl_`i'_minus_XX 
replace U_Gg_pl_`i'_global_XX=. if first_obs_by_gp_XX==0

gen count`i'_pl_global_XX=max(count`i'_pl_plus_XX, count`i'_pl_minus_XX)

if "`normalized'"!=""{
	scalar delta_D_pl_`i'_global_XX = (N1_placebo_`i'_XX_new/(N1_placebo_`i'_XX_new+N0_placebo_`i'_XX_new))*delta_D_pl_`i'_in_XX+(N0_placebo_`i'_XX_new/(N1_placebo_`i'_XX_new+N0_placebo_`i'_XX_new))*delta_D_pl_`i'_out_XX
}

gegen DID_placebo_`i'_XX = total(U_Gg_pl_`i'_global_XX) 
replace  DID_placebo_`i'_XX= DID_placebo_`i'_XX/G_XX
scalar DID_placebo_`i'_XX = DID_placebo_`i'_XX

if "`normalized'"!=""{
	scalar DID_placebo_`i'_XX = scalar(DID_placebo_`i'_XX)/scalar(delta_D_pl_`i'_global_XX)
}

if ("`switchers'"==""&N1_placebo_`i'_XX_new==0&N0_placebo_`i'_XX_new==0)|("`switchers'"=="out"&N0_placebo_`i'_XX_new==0)|("`switchers'"=="in"&N1_placebo_`i'_XX_new==0){
	scalar DID_placebo_`i'_XX=.
}

ereturn scalar Placebo_`i' = scalar(DID_placebo_`i'_XX)

matrix mat_res_XX[`=l_XX'+ 1 + `i',1]=scalar(DID_placebo_`i'_XX)
local rownames "`rownames' Placebo_`i'"

scalar N_switchers_placebo_`i'_XX=N1_placebo_`i'_XX_new+N0_placebo_`i'_XX_new
matrix mat_res_XX[`=l_XX'+ 1 + `i',7]= `=-`i''
ereturn scalar N_switchers_placebo_`i' = N_switchers_placebo_`i'_XX

gegen N_placebo_`i'_XX = total(count`i'_pl_global_XX)
scalar N_placebo_`i'_XX = N_placebo_`i'_XX

if N_switchers_placebo_`i'_XX==0|N_placebo_`i'_XX==0{
	di as error ""
	di as error "Placebo_"`i' " cannot be estimated."
	di as error "There is no switcher or no control"
	di as error "for this placebo."
	di as input _continue ""
	
	local missing_pl_XX `missing_pl_XX' `i'
}
if "`normalized'"!=""{
	if (delta_D_pl_`i'_global_XX==0|delta_D_pl_`i'_global_XX==.){
	di as error ""
	di as error "Placebo_"`i' " cannot be estimated."
	di as error "The denominator is missing for this normalized placebo."
	di as input _continue ""
	
	local missing_pl_XX `missing_pl_XX' `i'
	
	scalar N_placebo_`i'_XX=0
	scalar N_switchers_placebo_`i'_XX=0
	}
}

ereturn scalar N_placebo_`i' = N_placebo_`i'_XX
matrix mat_res_XX[`=l_XX' + 1 + `i',5]=N_placebo_`i'_XX
matrix mat_res_XX[`=l_XX'+ 1 + `i',6]=N_switchers_placebo_`i'_XX

// Modif Felix weight
if "`weight'"!=""{
	matrix mat_res_XX[`=l_XX'+ 1 + `i',6]=N1_`i'_pl_XX_weight+N0_`i'_pl_XX_weight
	ereturn scalar N_switchers_placebo_`i'_w=N1_`i'_pl_XX_weight+N0_`i'_pl_XX_weight
	
	capture drop count`i'_pl_global_XX_w
	capture drop N_placebo_`i'_XX_w
	
	gen count`i'_pl_global_XX_w=count`i'_pl_global_XX/weight_XX
	gegen N_placebo_`i'_XX_w = total(count`i'_pl_global_XX_w) 
	scalar N_placebo_`i'_XX_w = N_placebo_`i'_XX_w
	ereturn scalar N_placebo_`i'_w = N_placebo_`i'_XX_w
	matrix mat_res_XX[`=l_XX' + 1 + `i',5]=N_placebo_`i'_XX_w
}

}

// Generate string local for sotable
local str_placebo "Placebo_1"
if l_placebo_XX > 1 {
	forv j = 2/`=l_placebo_XX' {
		local str_placebo "`str_placebo' Placebo_`j'"
	}
}

}

//// Extract estimates vector (without avg_effect) for honestdid
matrix didmgt_results_no_avg1_XX=J(`=l_XX',1,.)
matrix didmgt_results_no_avg1_XX=mat_res_XX[1 .. `=l_XX',1]
matrix didmgt_results_no_avg_XX = didmgt_results_no_avg1_XX

if `=l_placebo_XX'>0{
matrix didmgt_results_no_avg2_XX=J(`=l_placebo_XX',1,.)
matrix didmgt_results_no_avg2_XX=mat_res_XX[`=l_XX'+2 .. `=l_XX'+`=l_placebo_XX'+1,1]
matrix didmgt_results_no_avg_XX = didmgt_results_no_avg1_XX \ didmgt_results_no_avg2_XX
}

matrix didmgt_results_no_avg_XX=didmgt_results_no_avg_XX'

capture matrix drop didmgt_results_no_avg1_XX didmgt_results_no_avg2_XX

//// Add bootstrap 
if `bootstrap_XX'!=0{
	
	// Save "pre-bootstraped" results matrix so it does not get overwritten 
	matrix mat_res_bs_XX=mat_res_XX
	
	// Save "pre-bootstrap" data 
	qui tempfile data_pre_bootstrap_XX
	qui save "`data_pre_bootstrap_XX'.dta", replace
	
	// Load original data 
	use "`data_bootstrap_XX'.dta", clear
	
	// gen numlists
	numlist "1/`=l_XX'"
	local num_eff_XX "`r(numlist)'"
	// Adjust for missings
	local num_eff_XX: list num_eff_XX- missing_eff_XX
	
	if `=l_placebo_XX'>0{
	numlist "1/`=l_placebo_XX'"
	local num_pl_XX "`r(numlist)'"
	// Adjust for missings
	local num_pl_XX: list num_pl_XX- missing_pl_XX
	}
	
	//Store the coefficients to bootstrap
	local coefs ""
	foreach i of numlist `num_eff_XX'{
		local coefs = "`coefs' e(Effect_`i')"
	}
	
	if `=l_placebo_XX'>0{
	foreach i of numlist `num_pl_XX'{
		local coefs = "`coefs' e(Placebo_`i')"
	}
	}
	
	if "`trends_lin'"==""{
	local coefs "`coefs' e(Av_tot_effect)"
	}
	
	xtset, clear // need this to ensure bootstrap is running correctly
	
	// Save pre-bootstraped coefficients matrix for residualization (controls) and matrix for the actual coefficient matrix to be used in the ereturn post 
	foreach u of local levels_d_sq_XX {
		matrix coefs_sq_bs_`u'_XX = coefs_sq_`u'_XX
	}
	matrix didmgt_bs_results_no_avg_XX=didmgt_results_no_avg_XX // Modif Felix
	
	// Modif Felix: Ensure that we do not add missings in this process
	gen existing_pre_XX=1
	
	// Save pre-bootstraped scalars 
	mata: scalars_XX = st_dir("global", "numscalar", "*_XX")	
	getmata (scalars_XX*) = scalars_XX, force					
	// Fix this by counting the number of observations before and after we add all the scalars in and then delete the additional cells we might add!!!				
	
	levelsof(scalars_XX), local(scalars_save)
	foreach w of local scalars_save {
		local k = subinstr("`w'","_XX","_XY",.)
		scalar `k' = `w'
	}
	capture drop scalars_XX* 
	
		
	// Modif Felix: Ensure that we do not add missings in this process
	drop if existing_pre_XX==.
	drop existing_pre_XX
	
	global bootstrap_on_XX "yes"
	
	// bootstrap the whole command
	if "`cluster'"==""{
		bootstrap "`coefs'", reps(`bootstrap_XX') cluster(`2') `seed_XX': did_multiplegt_dyn `1' `2' `3' `4', effects(`=l_XY') placebo(`=l_placebo_XY') `same_switchers' switchers(`switchers') `only_never_switchers' controls(`controls_bs_orig_XX') trends_nonparam(`trends_nonparam') weight(`weight') `dont_drop_larger_lower' `drop_if_d_miss_before_first_switch' `normalized' `same_switchers_pl' `trends_lin' continuous(`continuous') graph_off
	}
	else if "`cluster'"!=""{
		bootstrap "`coefs'", reps(`bootstrap_XX') cluster(`cluster') `seed_XX': did_multiplegt_dyn `1' `2' `3' `4', effects(`=l_XY') placebo(`=l_placebo_XY') `same_switchers' switchers(`switchers') `only_never_switchers' controls(`controls_bs_orig_XX') trends_nonparam(`trends_nonparam') weight(`weight') `dont_drop_larger_lower' `normalized' `same_switchers_pl' `drop_if_d_miss_before_first_switch' `trends_lin' continuous(`continuous') graph_off
	}
	
	//global bootstrap_on_XX ""
	
	// store bootstrap results matrix
	matrix bs_out_XX=r(table)
	matrix bs_var_cov_XX=e(V) // Modif Felix: also fetch var/cov Matrix
	
	
	// Restore residualization matrix 
	foreach u of local levels_d_sq_XX {
		matrix coefs_sq_`u'_XX = coefs_sq_bs_`u'_XX
	}
	matrix didmgt_results_no_avg_XX=didmgt_bs_results_no_avg_XX // Modif Felix
	
	// Restore scalars
	mata: scalars_XY = st_dir("global", "numscalar", "*_XY")
	getmata (scalars_XY*) = scalars_XY, force
	levelsof(scalars_XY), local(scalars_save)
	foreach w of local scalars_save {
		local k = subinstr("`w'","_XY","_XX",.)
		scalar `k' = `w'
	}
	capture drop scalars_XY*
	
	// row 1 to l_XX are the effects 
	mat se_eff_bs_XX=J(`=l_XX',1,.)
	local j=1 // local to find the correct row in the bootstrap output matrix
	foreach k of numlist `num_eff_XX'{
		mat se_eff_bs_XX[`k',1]=bs_out_XX[2,`j']
		local j=`j'+1
	}
	
	// row l_XX+1 to l_XX+l_placebo_XX are the placebos
	if l_placebo_XX!=0{
		mat se_pl_bs_XX=J(`=l_placebo_XX',1,.)
		local for_count_2_XX=`=l_XX'+`=l_placebo_XX' // Felix: obsolete
		local for_count_3_XX=`=l_XX'+1 // Felix: obsolete
		foreach k of numlist `num_pl_XX'{
			mat se_pl_bs_XX[`k',1]=bs_out_XX[2,`j']
			local j=`j'+1
		}
	}
	
	if "`trends_lin'"==""{
	// row l_XX+l_placebo_XX+1 isvaverage cummulative (total) effect
	mat se_avg_eff_bs_XX=bs_out_XX[2,`j']
	}
	
	// Reload results matrix 
	matrix mat_res_XX=mat_res_bs_XX
	
	// Reload the "pre-bootstrap" data 
	use "`data_pre_bootstrap_XX'.dta", clear
}
global bootstrap_on_XX "" // Modif Felix

/////  Computing the variance of DID_\ell

// Loop over the number of effects to be estimated

forvalue i=1/`=l_XX'{

// Check that the effect can be estimated	

if ("`switchers'"==""&(N1_`i'_XX_new!=0|N0_`i'_XX_new!=0))|("`switchers'"=="out"&N0_`i'_XX_new!=0)|("`switchers'"=="in"&N1_`i'_XX_new!=0){

// Aggregating the U_Gg_var_\ell for switchers in and out

gen U_Gg_var_glob_`i'_XX = U_Gg_var_`i'_in_XX * (scalar(N1_`i'_XX_new)/(scalar(N1_`i'_XX_new)+scalar(N0_`i'_XX_new))) + U_Gg_var_`i'_out_XX* (scalar(N0_`i'_XX_new)/(scalar(N1_`i'_XX_new)+scalar(N0_`i'_XX_new)))

// Compute sigma_hat_2_l without clustering

if "`cluster'"==""{
	gen U_Gg_var_glob_eff`i'_sqrd_XX = U_Gg_var_glob_`i'_XX^2*first_obs_by_gp_XX

	sum U_Gg_var_glob_eff`i'_sqrd_XX
	scalar sum_for_var_`i'_XX=r(sum)/G_XX^2
}

// Compute sigma_hat_2_l with clustering: sum U_Gg_var_l within a cluster, and then take average of square. 

if "`cluster'"!=""{
	
	capture drop clust_U_Gg_var_glob_`i'_XX
	capture drop clust_U_Gg_var_glob_`i'_2_XX
	
	replace U_Gg_var_glob_`i'_XX = U_Gg_var_glob_`i'_XX*first_obs_by_gp_XX
	
	* Sum within cluster
	bys `cluster' : gegen clust_U_Gg_var_glob_`i'_XX = total(U_Gg_var_glob_`i'_XX)
	
	* Compute average of square
	gen clust_U_Gg_var_glob_`i'_2_XX=clust_U_Gg_var_glob_`i'_XX^2*first_obs_by_clust_XX
	
	sum clust_U_Gg_var_glob_`i'_2_XX
	scalar sum_for_var_`i'_XX=r(sum)/G_XX^2
	
	replace U_Gg_var_glob_`i'_XX = clust_U_Gg_var_glob_`i'_XX
	
}

// Compute SE
scalar se_`i'_XX = sqrt(sum_for_var_`i'_XX)

// Normalize SE if normalized option was specified
if "`normalized'"!=""{
	scalar se_`i'_XX = scalar(se_`i'_XX)/scalar(delta_D_`i'_global_XX)
}

// Adapting the se scalars with the bootstrap
if `bootstrap_XX'!=0{
	scalar se_`i'_XX=se_eff_bs_XX[`i',1]
}

// Storing the results

* SE
matrix mat_res_XX[`i',2]=scalar(se_`i'_XX)
ereturn scalar se_effect_`i'=scalar(se_`i'_XX)

* ci_level
scalar ci_level = `ci_level'/100
scalar z_level = invnormal(scalar(ci_level) + (1-scalar(ci_level))/2)
	
* Lower bound of the confidence interval
scalar LB_CI_`i'_XX = scalar(DID_`i'_XX) - scalar(z_level)*scalar(se_`i'_XX)
matrix mat_res_XX[`i',3]= scalar(LB_CI_`i'_XX)

* Upper bound of the confidence interval
scalar UB_CI_`i'_XX = scalar(DID_`i'_XX) + scalar(z_level)*scalar(se_`i'_XX)
matrix mat_res_XX[`i',4]=scalar(UB_CI_`i'_XX)

}
}


/////  Computing the variances of the placebo estimators (same steps as for the DID_\ell, not commented)

if l_placebo_XX!=0{

forvalue i=1/`=l_placebo_XX'{
if ("`switchers'"==""&(N1_placebo_`i'_XX_new!=0|N0_placebo_`i'_XX_new!=0))|("`switchers'"=="out"&N0_placebo_`i'_XX_new!=0)|("`switchers'"=="in"&N1_placebo_`i'_XX_new!=0){ 

gen U_Gg_var_glob_pl_`i'_XX = U_Gg_var_pl_`i'_in_XX * (N1_placebo_`i'_XX_new/(N1_placebo_`i'_XX_new+N0_placebo_`i'_XX_new)) + U_Gg_var_pl_`i'_out_XX* (N0_placebo_`i'_XX_new/(N1_placebo_`i'_XX_new+N0_placebo_`i'_XX_new))

if "`cluster'"==""{

gen U_Gg_var_glob_pl_`i'_2_XX = U_Gg_var_glob_pl_`i'_XX^2*first_obs_by_gp_XX

sum U_Gg_var_glob_pl_`i'_2_XX
scalar sum_for_var_placebo_`i'_XX=r(sum)/G_XX^2
}

if "`cluster'"!=""{
	capture drop clust_U_Gg_var_glob_pl_`i'_XX
	capture drop clust_U_Gg_var_glob_pl_`i'_2_XX
	
	replace U_Gg_var_glob_pl_`i'_XX = U_Gg_var_glob_pl_`i'_XX*first_obs_by_gp_XX

	bys `cluster' : gegen clust_U_Gg_var_glob_pl_`i'_XX= total(U_Gg_var_glob_pl_`i'_XX)

	gen clust_U_Gg_var_glob_pl_`i'_2_XX=clust_U_Gg_var_glob_pl_`i'_XX^2*first_obs_by_clust_XX
	
	sum clust_U_Gg_var_glob_pl_`i'_2_XX
	scalar sum_for_var_placebo_`i'_XX=r(sum)/G_XX^2
	
	replace U_Gg_var_glob_pl_`i'_XX = clust_U_Gg_var_glob_pl_`i'_XX  
}

scalar se_placebo_`i'_XX = sqrt(sum_for_var_placebo_`i'_XX)

if "`normalized'"!=""{
	scalar se_placebo_`i'_XX = scalar(se_placebo_`i'_XX)/scalar(delta_D_pl_`i'_global_XX)
}

// Adapting the se scalars with the bootstrap
if `bootstrap_XX'!=0{
	scalar se_placebo_`i'_XX=se_pl_bs_XX[`i',1]
}

matrix mat_res_XX[`=l_XX' + 1 + `i',2]=scalar(se_placebo_`i'_XX)
ereturn scalar se_placebo_`i'=scalar(se_placebo_`i'_XX)
	
scalar LB_CI_placebo_`i'_XX = scalar(DID_placebo_`i'_XX) - scalar(z_level)*scalar(se_placebo_`i'_XX)
matrix mat_res_XX[`=l_XX'+ 1 + `i', 3]= scalar(LB_CI_placebo_`i'_XX)

scalar UB_CI_placebo_`i'_XX = scalar(DID_placebo_`i'_XX) + scalar(z_level)*scalar(se_placebo_`i'_XX)
matrix mat_res_XX[`=l_XX'+ 1 + `i',4]=scalar(UB_CI_placebo_`i'_XX)

}
}
}

/////  Computing the variance of the average total effect (same steps as for the DID_\ell, not commented)

if "`trends_lin'"==""{

if ("`switchers'"==""&(sum_N1_l_XX!=0|sum_N0_l_XX!=0))|("`switchers'"=="out"&sum_N0_l_XX!=0)|("`switchers'"=="in"&sum_N1_l_XX!=0){

gen U_Gg_var_global_XX = w_plus_XX * U_Gg_var_plus_XX + (1 - w_plus_XX) * U_Gg_var_minus_XX

if "`cluster'"==""{
capture drop U_Gg_var_global_2_XX
gen U_Gg_var_global_2_XX = U_Gg_var_global_XX^2*first_obs_by_gp_XX

sum U_Gg_var_global_2_XX
scalar sum_for_var_XX=r(sum)/G_XX^2
}

if "`cluster'"!=""{
	capture drop clust_U_Gg_var_global_XX
	
	replace U_Gg_var_global_XX=U_Gg_var_global_XX*first_obs_by_gp_XX
	
	bys `cluster' : gegen clust_U_Gg_var_global_XX= total(U_Gg_var_global_XX)
	
	replace clust_U_Gg_var_global_XX=clust_U_Gg_var_global_XX^2*first_obs_by_clust_XX
	
	sum clust_U_Gg_var_global_XX
	scalar sum_for_var_XX=r(sum)/G_XX^2
}

scalar se_XX = sqrt(sum_for_var_XX)

// Adapting the se scalars with the bootstrap
if `bootstrap_XX'!=0{
	scalar se_XX=se_avg_eff_bs_XX[1,1]
}

matrix mat_res_XX[l_XX+1,2]=se_XX
ereturn scalar se_avg_total_effect = se_XX

scalar LB_CI_XX = delta_XX - scalar(z_level)*se_XX
matrix mat_res_XX[l_XX+1,3]= LB_CI_XX

scalar UB_CI_XX = delta_XX + scalar(z_level)*se_XX
matrix mat_res_XX[l_XX+1,4]= UB_CI_XX

}
}

///// Storing results in a way ensuring that command compatible with "event_plot" command 

// Extract Estimates

matrix didmgt_b2=mat_res_XX[1...,1..1]

// Extract Variances

matrix didmgt_var2=J(`=l_placebo_XX'+`=l_XX'+1,1,0)
forvalue i=1/`=`=l_placebo_XX'+`=l_XX'+1'{
matrix didmgt_var2[`i',1]=mat_res_XX[`i',2]^2
}

// Add reference period

local rownames2 "`rownames' Effect_0"
matrix zero=J(1,1,0)
matrix didmgt_b2=didmgt_b2\zero
matrix didmgt_var2=didmgt_var2\zero

// Add correct names

matrix rownames didmgt_var2= `rownames2'
matrix rownames didmgt_b2 = `rownames2'

// For compatibility with old do files:

matrix compatibility_variances = didmgt_var2

// Set up the output to match event_plot
/*
ereturn matrix estimates=didmgt_b2 
ereturn matrix didmgt_variances=didmgt_var2
ereturn matrix variances= compatibility_variances
ereturn local cmd "did_multiplegt_dyn"
*/


////////// 6. Computing p-values from the tests 

//// Generate combined Variance-Covariance Matrix for honestdid
// Note: Check if all of them are defined!
// If test is feasible, initalize scalar at 0
		scalar all_Ns_pl_not_zero=0
		scalar all_delta_pl_not_zero=0
		
		// Count the number of placebos that can be estimated
		forvalue i=1/`=l_placebo_XX'{
		if ("`switchers'"==""&(N1_placebo_`i'_XX_new!=0|N0_placebo_`i'_XX_new!=0))|("`switchers'"=="out"&N0_placebo_`i'_XX_new!=0)|("`switchers'"=="in"&N1_placebo_`i'_XX_new!=0){
			scalar all_Ns_pl_not_zero=all_Ns_pl_not_zero+1
			
		}
		// Count the number of normalized placebos that can be estimated
		if "`normalized'"!=""{
		if (delta_D_pl_`i'_global_XX!=0 & delta_D_pl_`i'_global_XX!=.){
			scalar all_delta_pl_not_zero=all_delta_pl_not_zero+1
		}
		}
		}
		
		scalar all_Ns_not_zero=0
	
		forvalue i=1/`=l_XX'{
		if ("`switchers'"==""&(N1_`i'_XX_new!=0|N0_`i'_XX_new!=0))|("`switchers'"=="out"&N0_`i'_XX_new!=0)|("`switchers'"=="in"&N1_`i'_XX_new!=0){
			scalar all_Ns_not_zero=all_Ns_not_zero+1
			}
	}
		
if ((all_Ns_pl_not_zero==l_placebo_XX & "`normalized'"=="")|(all_Ns_pl_not_zero==l_placebo_XX & "`normalized'"!="" & all_delta_pl_not_zero==l_placebo_XX))&all_Ns_not_zero==l_XX{		

// Creating a matrix where the variances and the covariances will be stored.
matrix didmgt_Var_all_XX=J(`=l_XX'+`=l_placebo_XX',`=l_XX'+`=l_placebo_XX',0)

local total_num_estimates_XX=`=l_XX'+`=l_placebo_XX'

// combine effects and placebos in "one" variable with running index (similarly for scalars storing the se's)
capture drop U_Gg_var_comb_*_XX
forvalue i=1/`=l_XX'{
	gen U_Gg_var_comb_`i'_XX=U_Gg_var_glob_`i'_XX
	scalar se_all_`i'_XX=se_`i'_XX
	if "`normalized'"!=""{
		scalar delta_D_all_`i'_XX=delta_D_`i'_global_XX
	}
}
forvalue i=`=l_XX+1'/`total_num_estimates_XX'{
	local j=`i'-`=l_XX'
	gen U_Gg_var_comb_`i'_XX=U_Gg_var_glob_pl_`j'_XX
	scalar se_all_`i'_XX=se_placebo_`j'_XX
	if "`normalized'"!=""{
		scalar delta_D_all_`i'_XX=delta_D_pl_`j'_global_XX
	}
}


// If the option cluster is specified, we have previously replaced U_Gg_var_glob_pl_`i'_XX by clust_U_Gg_var_glob_pl_`i'_XX, and U_Gg_var_glob_`i'_XX by clust_U_Gg_var_glob_`i'_XX. 
// Now, we must also replace first_obs_by_gp_XX by first_obs_by_clust_XX
if "`cluster'"!=""{
replace first_obs_by_gp_XX=first_obs_by_clust_XX
}

forvalue i=1/`total_num_estimates_XX'{
		
		// Fill the diagonal with the Variances
		matrix didmgt_Var_all_XX[`i',`i']= scalar(se_all_`i'_XX)^2
	
		// Fill off diagonal with covariances
		if `i'<`total_num_estimates_XX'{
		forvalue j=`=`i'+1'/`total_num_estimates_XX'{
			
			* Create variables necessary to compute the covariances
			capture drop U_Gg_var_comb_`i'_`j'_XX
			capture drop U_Gg_var_comb_`i'_`j'_2_XX
					
			if ("`normalized'"==""){
			gen U_Gg_var_comb_`i'_`j'_XX = U_Gg_var_comb_`i'_XX + U_Gg_var_comb_`j'_XX
		}
			
			// Also need adjusted delta_D for all effects!!!
			if "`normalized'"!=""{
			gen U_Gg_var_comb_`i'_`j'_XX = U_Gg_var_comb_`i'_XX/scalar(delta_D_all_`i'_XX) + U_Gg_var_comb_`j'_XX/scalar(delta_D_all_`j'_XX)
		}

			* Estimate the covariances
			gen U_Gg_var_comb_`i'_`j'_2_XX = U_Gg_var_comb_`i'_`j'_XX^2*first_obs_by_gp_XX
			
			
			sum U_Gg_var_comb_`i'_`j'_2_XX
			scalar var_sum_comb_`i'_`j'_XX=r(sum)/G_XX^2
			
			scalar cov_comb_`i'_`j'_XX = (scalar(var_sum_comb_`i'_`j'_XX) - scalar(se_all_`i'_XX)^2 - scalar(se_all_`j'_XX)^2)/2
	
			* Store the results
			matrix didmgt_Var_all_XX[`i',`j']= scalar(cov_comb_`i'_`j'_XX)
			matrix didmgt_Var_all_XX[`j',`i']= scalar(cov_comb_`i'_`j'_XX)

		}
	}
	
}

*** Modif Felix: replace by bootstrap var/cov matrix
if `bootstrap_XX'!=0{
	matrix didmgt_Var_all_XX=bs_var_cov_XX
}

}
	// Error message if not all of the specified effects/placebos could be estimated 
	else{
		di as error ""
		di as error "Some placebos/effects could not be estimated. Therefore, the command will not be compatible with the honestdid command."
		di as input _continue ""
	}

///// Performing a test that all placebo are jointly equal to 0.

// initalize scalar
scalar all_Ns_pl_not_zero=.
scalar all_delta_pl_not_zero=.

// Test can only be run when at least two placebos requested:
if (l_placebo_XX!=0)&l_placebo_XX>1{

		// If test is feasible, initalize scalar at 0
		scalar all_Ns_pl_not_zero=0
		scalar all_delta_pl_not_zero=0
		
		// Count the number of placebos that can be estimated
		forvalue i=1/`=l_placebo_XX'{
		if ("`switchers'"==""&(N1_placebo_`i'_XX_new!=0|N0_placebo_`i'_XX_new!=0))|("`switchers'"=="out"&N0_placebo_`i'_XX_new!=0)|("`switchers'"=="in"&N1_placebo_`i'_XX_new!=0){
			scalar all_Ns_pl_not_zero=all_Ns_pl_not_zero+1
			
		}
		// Count the number of normalized placebos that can be estimated
		if "`normalized'"!=""{
		if (delta_D_pl_`i'_global_XX!=0 & delta_D_pl_`i'_global_XX!=.){
			scalar all_delta_pl_not_zero=all_delta_pl_not_zero+1
		}
		}
		}

	// Test can only be run when all requested placebos could be computed:
	// Add condition with normalized
	if (all_Ns_pl_not_zero==l_placebo_XX & "`normalized'"=="")|(all_Ns_pl_not_zero==l_placebo_XX & "`normalized'"!="" & all_delta_pl_not_zero==l_placebo_XX){

	// Creating a vector with all placebo estimates
	matrix didmgt_Placebo=J(l_placebo_XX,1,0)
	
	// Creating a matrix where the variances and the covariances of the placebos will be stored.
	matrix didmgt_Var_Placebo=J(l_placebo_XX,l_placebo_XX,0)
	
	// Fill those matrices
	forvalue i=1/`=l_placebo_XX'{
		
		matrix didmgt_Placebo[`i',1]=scalar(DID_placebo_`i'_XX)
		matrix didmgt_Var_Placebo[`i',`i']= scalar(se_placebo_`i'_XX)^2
	
		if `i'<`=l_placebo_XX'{
		forvalue j=`=`i'+1'/`=l_placebo_XX'{
			
			* Create variables necessary to compute the covariances
			capture drop U_Gg_var_pl_`i'_`j'_XX
			capture drop U_Gg_var_pl_`i'_`j'_2_XX
					
			if ("`normalized'"==""){
			gen U_Gg_var_pl_`i'_`j'_XX = U_Gg_var_glob_pl_`i'_XX + U_Gg_var_glob_pl_`j'_XX
		}
			
			if "`normalized'"!=""{
			gen U_Gg_var_pl_`i'_`j'_XX = U_Gg_var_glob_pl_`i'_XX/scalar(delta_D_pl_`i'_global_XX) + U_Gg_var_glob_pl_`j'_XX/scalar(delta_D_pl_`j'_global_XX)
		}

			* Estimate the covariances
			gen U_Gg_var_pl_`i'_`j'_2_XX = U_Gg_var_pl_`i'_`j'_XX^2*first_obs_by_gp_XX
			
			sum U_Gg_var_pl_`i'_`j'_2_XX
			scalar var_sum_pla_`i'_`j'_XX=r(sum)/G_XX^2
			
			scalar cov_pl_`i'_`j'_XX = (scalar(var_sum_pla_`i'_`j'_XX) - scalar(se_placebo_`i'_XX)^2 - scalar(se_placebo_`j'_XX)^2)/2
	
			* Store the results
			matrix didmgt_Var_Placebo[`i',`j']= scalar(cov_pl_`i'_`j'_XX)
			matrix didmgt_Var_Placebo[`j',`i']= scalar(cov_pl_`i'_`j'_XX)

		}
	}
	
	}
	
	*** Modif Felix: replace matrix with bootstrap
	if `bootstrap_XX'!=0{
		matrix didmgt_Var_Placebo=bs_var_cov_XX["_bs_`=l_XX+1'".."_bs_`=l_XX+l_placebo_XX'","_bs_`=l_XX+1'".."_bs_`=l_XX+l_placebo_XX'"]
	}	
	
	// Compute P-value for the F-test on joint nullity of all placebos
	matrix didmgt_Var_Placebo_inv=invsym(didmgt_Var_Placebo)
	matrix didmgt_Placebo_t=didmgt_Placebo'
	matrix didmgt_chi2placebo=didmgt_Placebo_t*didmgt_Var_Placebo_inv*didmgt_Placebo
	scalar p_jointplacebo=1-chi2(l_placebo_XX,didmgt_chi2placebo[1,1])
	ereturn scalar p_jointplacebo=1-chi2(l_placebo_XX,didmgt_chi2placebo[1,1])

	}
	
	// Error message if not all of the specified placebos could be estimated 
	else{
		di as error ""
		di as error "Some placebos could not be estimated. Therefore, the test of joint nullity of the placebos could not be computed."
		di as input _continue ""
	}
	
}


///// Performing a test that all DID_\ell effects are equal to 0 
// Felix: This will be automatically reported in case there are at least two effects requested and none are missign

// initalize scalar
scalar all_Ns_not_zero=.
scalar all_delta_not_zero=.

// Test can only be run when at least two placebos requested:
if l_XX>1{

		// If test is feasible, initalize scalar at 0
		scalar all_Ns_not_zero=0
		scalar all_delta_not_zero=0
		
		// Count the number of effects that can be estimated
		forvalue i=1/`=l_XX'{
		if ("`switchers'"==""&(N1_`i'_XX_new!=0|N0_`i'_XX_new!=0))|("`switchers'"=="out"&N0_`i'_XX_new!=0)|("`switchers'"=="in"&N1_`i'_XX_new!=0){
			scalar all_Ns_not_zero=all_Ns_not_zero+1
			
		}
		// Count the number of normalized effects that can be estimated
		if "`normalized'"!=""{
		if (delta_D_`i'_global_XX!=0 & delta_D_`i'_global_XX!=.){
			scalar all_delta_not_zero=all_delta_not_zero+1
		}
		}
		}

	// Test can only be run when all requested effects could be computed:
	// Add condition with normalized
	if (all_Ns_not_zero==l_XX & "`normalized'"=="")|(all_Ns_not_zero==l_XX & "`normalized'"!="" & all_delta_not_zero==l_XX){

	// Creating a vector with all placebo estimates
	matrix didmgt_effects=J(l_XX,1,0)
	
	// Creating a matrix where the variances and the covariances of the placebos will be stored.
	matrix didmgt_Var_effects=J(l_XX,l_XX,0)
	
	// Fill those matrices
	forvalue i=1/`=l_XX'{
		
		matrix didmgt_effects[`i',1]=scalar(DID_`i'_XX)
		matrix didmgt_Var_effects[`i',`i']= scalar(se_`i'_XX)^2
	
		if `i'<`=l_XX'{
		forvalue j=`=`i'+1'/`=l_XX'{
			
			* Create variables necessary to compute the covariances
			capture drop U_Gg_var_`i'_`j'_XX
			capture drop U_Gg_var_`i'_`j'_2_XX
					
			if ("`normalized'"==""){
			gen U_Gg_var_`i'_`j'_XX = U_Gg_var_glob_`i'_XX + U_Gg_var_glob_`j'_XX
		}
			
			if "`normalized'"!=""{
			gen U_Gg_var_`i'_`j'_XX = U_Gg_var_glob_`i'_XX/scalar(delta_D_`i'_global_XX) + U_Gg_var_glob_`j'_XX/scalar(delta_D_`j'_global_XX)
		}

			* Estimate the covariances
			gen U_Gg_var_`i'_`j'_2_XX = U_Gg_var_`i'_`j'_XX^2*first_obs_by_gp_XX
			
			sum U_Gg_var_`i'_`j'_2_XX
			scalar var_sum_`i'_`j'_XX=r(sum)/G_XX^2
			
			scalar cov_`i'_`j'_XX = (scalar(var_sum_`i'_`j'_XX) - scalar(se_`i'_XX)^2 - scalar(se_`j'_XX)^2)/2
	
			* Store the results
			matrix didmgt_Var_effects[`i',`j']= scalar(cov_`i'_`j'_XX)
			matrix didmgt_Var_effects[`j',`i']= scalar(cov_`i'_`j'_XX)

		}
	}
	
	}
	
	*** Modif Felix: replace matrix with bootstrap
	if `bootstrap_XX'!=0{
		matrix didmgt_Var_eff=bs_var_cov_XX["_bs_`=1'".."_bs_`=l_XX'","_bs_`=1'".."_bs_`=l_XX'"]
	}	
	
	// Compute P-value for the F-test on joint nullity of all effects
	matrix didmgt_Var_effects_inv=invsym(didmgt_Var_effects)
	matrix didmgt_effects_t=didmgt_effects'
	matrix didmgt_chi2effects=didmgt_effects_t*didmgt_Var_effects_inv*didmgt_effects
	scalar p_jointeffects=1-chi2(l_XX,didmgt_chi2effects[1,1])
	ereturn scalar p_jointeffects=1-chi2(l_XX,didmgt_chi2effects[1,1])

	}
	
	// Error message if not all of the specified placebos could be estimated 
	else{
		di as error ""
		di as error "Some effects could not be estimated. Therefore, the test of joint nullity of the effects could not be computed."
		di as input _continue ""
	}
	
}

// LBX change below 

///// Performing a test that all DID_\ell effects are equal (similar structure as test on placebos, not commented, except for the small differences with placebos)
// scalar all_Ns_not_zero=. // Modif Felix: this scalar is already initalized above and would now be overwritten when it should not be	
// 	effects_equal(string)

local ee_called = 0

// Check and Extract Lower and Upper Bound from the Option
if ("`effects_equal'")!="" & l_XX>1{
	
	* when all is not specified 
	if "`effects_equal'" != "all"{ 
		
		* Exit when option wrongly called without separator
		if strpos("`effects_equal'", ",") == 0 {
			di as error "Syntax error in effects_equal option"
			di as error "Comma required"
			di as input _continue ""
			exit
		}
		
		* Extract lower and upper bounds of interest 
		local ee_lb = strtrim(substr("`effects_equal'", 1, strpos("`effects_equal'", ",") - 1))
		local ee_ub = strtrim(substr("`effects_equal'", strpos("`effects_equal'", ",") + 1, .))
		
				* Exit when option wrongly call with non-integer inputs
		capture confirm integer number `ee_lb'
		if _rc{
		di ""	
		di as error "The input of the effects_equal() option has to be a positive integer in the range of estimated effects."
		di as input _continue ""
		exit
		}
		capture confirm integer number `ee_ub'
		if _rc{
		di ""	
		di as error "The input of the effects_equal() option has to be a positive integer in the range of estimated effects."
		di as input _continue ""
		exit
		}
	
		* Exit if lower and upper bounds do not make sense
		if `ee_ub' <= `ee_lb' | `ee_lb' < 1 | `ee_ub' > l_XX{
			di as error "Syntax error in effect_equal option: The bounds specifed are out of range."
			exit
		}
	}
	
	* when all is specified: default range
	else{
		local ee_lb = 1
		local ee_ub = l_XX
	}
	
	
	* when effects_equal effectively called
	local ee_called = 1
}

// When option correctly called 
if `ee_called' == 1 & l_XX>1 {  

	scalar all_Ns_not_zero_equal_test=0
	
	// Check how many periods of effects have sufficient observations for the test to proceed. 	
	forvalue i=`ee_lb'/`ee_ub'{
		if ("`switchers'"==""&(N1_`i'_XX_new!=0|N0_`i'_XX_new!=0))|("`switchers'"=="out"&N0_`i'_XX_new!=0)|("`switchers'"=="in"&N1_`i'_XX_new!=0){
			scalar all_Ns_not_zero_equal_test=all_Ns_not_zero_equal_test+1 
			}
	}
	

	* track num of periods of interest
	local length=`ee_ub'-`ee_lb'+1
	
	* when all periods of interest have sufficient obs 
	
	
	if all_Ns_not_zero_equal_test==`length'{
		
		
		* Extract the first column of rows up to l_xx, hold DID_\ell to be tested for equality. 
		matrix didmgt_Effects = mat_res_XX[`ee_lb'..`ee_ub', 1]
		* Initialize the variance-covariance matrix for the effects of l_XX size. 
		matrix didmgt_Var_Effects=J(`length', `length', 0)
		* Create a partial identity matrix. 
		matrix didmgt_identity = J(`=`length'-1', `length',0 )  
		
		* loop over each dyn effects of interest
		
		forvalue i=`ee_lb'/`ee_ub'{
		
			* condition on sufficient obs for this dyn effect
			if ("`switchers'"==""&(N1_`i'_XX_new!=0|N0_`i'_XX_new!=0))|("`switchers'"=="out"&N0_`i'_XX_new!=0)|("`switchers'"=="in"&N1_`i'_XX_new!=0){
				
				* Location tracking in the sub matrices j:submatrix, i:full matrix. 
				local j=`i'-`ee_lb'+1
				
				* Update diagonal elements of didmgt_Var Effects
				matrix didmgt_Var_Effects[`j',`j']=scalar(se_`i'_XX)^2
				* Update diagonal element of the identity matrix for matrix of demeaned effects
				if `i' < `ee_ub'{
					matrix didmgt_identity[`j',`j']=1
				}
				
				* Update the variance-covariance matrix of selected effects
				if `i' < `ee_ub'{

					/* Nested loop iterate over rest periods to be paired with i from i+1 to upper bound */
					forvalue k=`=`i'+1'/`=`ee_ub''{ 
						
						* Location tracking in the sub matrices s:submatrix, k:full matrix. 
						local s=`k'-`ee_lb'+1
						
						capture drop U_Gg_var_`i'_`k'_XX
						capture drop U_Gg_var_`i'_`k'_2_XX	
						
						if ("`normalized'"==""){
							gen U_Gg_var_`i'_`k'_XX = U_Gg_var_glob_`i'_XX+ U_Gg_var_glob_`k'_XX
						}
						if "`normalized'"!=""{
							gen U_Gg_var_`i'_`k'_XX = U_Gg_var_glob_`i'_XX/scalar(delta_D_`i'_global_XX) + U_Gg_var_glob_`k'_XX/scalar(delta_D_`k'_global_XX)
						}
						
						gen U_Gg_var_`i'_`k'_2_XX = U_Gg_var_`i'_`k'_XX^2*first_obs_by_gp_XX

						sum U_Gg_var_`i'_`k'_2_XX
						scalar var_sum_`i'_`k'_XX=r(sum)/G_XX^2
						scalar cov_`i'_`k'_XX = (scalar(var_sum_`i'_`k'_XX) - scalar(se_`i'_XX)^2 - scalar(se_`k'_XX)^2)/2
							
						* Update covariance of i,k pair in submatrix
						matrix didmgt_Var_Effects[`j',`s']= scalar(cov_`i'_`k'_XX)
						matrix didmgt_Var_Effects[`s',`j']= scalar(cov_`i'_`k'_XX)
						
					}
				/* End nested loop */ 
				}
				}
				}
				
				
 		if `bootstrap_XX'!=0{
 			matrix didmgt_Var_Effects=bs_var_cov_XX["_bs_1".."_bs_`=l_XX'","_bs_1".."_bs_`=l_XX'"]
			matrix didmgt_Var_Effects = didmgt_Var_Effects[`ee_lb'..`ee_ub', `ee_lb'..`ee_ub']
 			}	


			/* Just subsetting everything */			
			//matrix didmgt_D = (1+1/(`=`length'-1')) * didmgt_identity - J(`=`length'-1', `=`length'', 1/(`=`length'-1'))
			matrix didmgt_D = didmgt_identity - J(`=`length'-1', `=`length'', 1/(`=`length''))
			* Demeaned effect matrix vector
			matrix didmgt_test_effects = didmgt_D * didmgt_Effects
			* Variance Covariance matrix of demeaned effects
			matrix didmgt_test_var = didmgt_D * didmgt_Var_Effects * didmgt_D'
			* Enforcing symmetry
			matrix didmgt_test_var = (didmgt_test_var + didmgt_test_var') / 2 
			matrix didmgt_test_var_inv = invsym(didmgt_test_var)
			matrix didmgt_test_effects_t = didmgt_test_effects'
			
			* chi-squared statistics : effectstranspose * (Variance)^(-1)* effects
			matrix didmgt_chi2_equal_ef = didmgt_test_effects_t * didmgt_test_var_inv * didmgt_test_effects
			* Calculate p-value: comparing the chi-square statistic (didmgt_chi2_equal_ef[1,1]) with the chi-square distribution 
			scalar p_equality_effects = 1 - chi2(`=`length'-1', didmgt_chi2_equal_ef[1, 1])  
			ereturn scalar p_equality_effects = 1 - chi2(`=`length'-1', didmgt_chi2_equal_ef[1, 1])	
			
		}
		else{
		di as error ""
		di as error "Some effects could not be estimated. Therefore, the test of equality of the effects could not be computed."
		di as input _continue ""
	}
}
	}

// LBX change above
		
////////// 7. Output the final tables
	
// Supress output during the bootstrap replications
if "$bootstrap_on_XX"==""{
	
///// Generating the output that will be shown in the console

// Adjust the row and cloumn names for the output tables
matrix rownames mat_res_XX= `rownames'
matrix colnames mat_res_XX= "Estimate" "SE" "LB CI" "UB CI" "N" "Switchers" "time_to_treat"

///// Table for the DID_l estimates
display _newline
di "{hline 80}"
di _skip(13) "{bf:Estimation of treatment effects: Event-study effects}"
if "`by'" !=""{	
* Add description with the by variable level if by is specified
di _skip(35) "{bf:By: `by' = `val_lab_int_XX'}"
}	
if "$k"!=""{
	local k=$k
	local nb_obs_adj_XX = strlen("${path_`k'_XX}")
	di _skip(`=42-`nb_obs_adj_XX'') "{bf:Path (${path_`k'_XX})}"
}
di "{hline 80}"
noisily matlist mat_res_XX[1..l_XX, 1..6]
di "{hline 80}"
// Felix: Add test on effects jointly = 0
if (l_XX>1&all_Ns_not_zero==l_XX&all_delta_not_zero==l_XX&"`normalized'"!="")|(l_XX>1&all_Ns_not_zero==l_XX&"`normalized'"==""){
di as text "{it:Test of joint nullity of the effects : p-value =} " scalar(p_jointeffects)
}

if `ee_called' == 1 & l_XX>1 {
// LBX Change below 
if l_XX>1&"`effects_equal'"!=""&all_Ns_not_zero_equal_test== `length'{ 
* When effects_equal is specified show P-value here	
if `length' == l_XX{
	di as text "{it:Test of equality of the effects : p-value =} " scalar(p_equality_effects)
}
else{
		di as text "{it:Test of equality of the effects of having been exposed to treatment for `ee_lb' to `ee_ub' periods: p-value =} " scalar(p_equality_effects) 
}
}

if all_Ns_not_zero_equal_test != `length'{ 
di as text "{it: Not enough observations in some period to perform the test of equality of the effects.}"
}
}
// LBX Change above

///// Table for the Average total effect estimate
if "`trends_lin'"==""{
* Not computed and therefore not shown when trends_lin is used	

matrix mat_res_avg_XX=mat_res_XX[l_XX+1, 1..6]
matrix mat_res_avg_XX=(mat_res_avg_XX, .z )
matrix colnames mat_res_avg_XX= "Estimate" "SE" "LB CI" "UB CI" "N" "Switch" "x Periods"
display _newline
di "{hline 80}"
di _skip(15) "{bf:Average cumulative (total) effect per treatment unit}"
if "`by'" !=""{	
* Add description with the by variable level if by is specified
di _skip(35) "{bf:By: `by' = `val_lab_int_XX'}"
}
if "$k"!=""{
	local k=$k
	local nb_obs_adj_XX = strlen("${path_`k'_XX}")
	di _skip(`=42-`nb_obs_adj_XX'') "{bf:Path (${path_`k'_XX})}"
}	
di "{hline 80}"
noisily matlist mat_res_avg_XX, nodotz
di "{hline 80}"

///// Preparing computation of average number of cumulated effects
// Determine max number of effects by group
qui{
// unify delta_D_g_`i'_plus_XX and delta_D_g_`i'_minus_XX
capture drop delta_D_g_1_XX - delta_D_g_`=l_XX'_XX
	
// generate M_g_XX, maximum number of estimable effects by group	
capture drop var_l_XX_*
gen var_l_XX_1=`=l_XX'
gen var_l_XX_2=T_g_XX-F_g_XX+1
gegen M_g_XX=min(var_l_XX_1 var_l_XX_2), by(group_XX)

// generate one variable that stores all the different delta_D_g_`i'_XX
capture drop delta_D_g_XX

gen delta_D_g_XX=.
forvalues j=1/`=l_XX'{
	capture drop delta_D_g_`j'_XX
	gen delta_D_g_`j'_XX=delta_D_g_`j'_plus_XX
	replace delta_D_g_`j'_XX=delta_D_g_`j'_minus_XX if delta_D_g_`j'_XX==0 // Note delta_D_g_`j'_plus_XX/_minus_XX were initiallized with value 0
	replace delta_D_g_`j'_XX=. if delta_D_g_`j'_XX==0 // does this matter? 
	
	// Replace the "general" variable with the corresponding delta_D for all effects included in the estimation
	replace delta_D_g_XX=delta_D_g_`j'_XX if switcher_tag_XX==`j'	
}

// Multiply by the corresponding L
gen delta_D_g_num_XX=delta_D_g_XX*[(M_g_XX-(switcher_tag_XX-1))]

// take the total over periods and groups 
gegen delta_D_num_total_XX=total(delta_D_g_num_XX)
sum delta_D_num_total_XX
local delta_D_num_total_XX=r(mean)

gegen delta_D_denom_total_XX=total(delta_D_g_XX)
sum delta_D_denom_total_XX
local delta_D_denom_total_XX=r(mean)
}
di as text "{it:Average number of time periods over which a treatment's effect is accumulated = }"  `delta_D_num_total_XX'/`delta_D_denom_total_XX'

}

// Output in case trends_lin is specified
if "`trends_lin'"!=""{
display _newline
di "{hline 80}"
di _skip(4) "{bf:When the option {it:trends_lin} is specified no average effects are reported}"
di "{hline 80}"
}	

///// Table for the placebo estimates
if l_placebo_XX!=0{
* Only shown when some placebos are requested

display _newline
di "{hline 80}"
di _skip(10) "{bf:Testing the parallel trends and no anticipation assumptions}"
if "`by'" !=""{	
* Add description with the by variable level if by is specified	
di _skip(35) "{bf:By: `by' = `val_lab_int_XX'}"
}
if "$k"!=""{
	local k=$k
	local nb_obs_adj_XX = strlen("${path_`k'_XX}")
	di  _skip(`=42-`nb_obs_adj_XX'') "{bf:Path (${path_`k'_XX})}"
}	
di "{hline 80}"
matlist mat_res_XX[l_XX+2...,1..6]
di "{hline 80}"
if (l_placebo_XX>1&all_Ns_pl_not_zero==l_placebo_XX&all_delta_pl_not_zero==l_placebo_XX&"`normalized'"!="")|(l_placebo_XX>1&all_Ns_pl_not_zero==l_placebo_XX&"`normalized'"==""){
di as text "{it:Test of joint nullity of the placebos : p-value =} " scalar(p_jointplacebo)
}
}

} // end bootstrap condition
////////// 8. Output other post estimation options

///// Testing for effect heterogeneity
if "`predict_het_good'"!=""{ 
qui{
	
sort group_XX time_XX	

// Define number of effects we want to calculate
if "`het_effects'"=="all"{
local all_effects_XX ""	
// Take all effects corresponding to the number of requested effect estimates
forvalues i=1/`=l_XX'{
local all_effects_XX "`all_effects_XX' `i'"
}
}

if "`het_effects'"!="all"{ 
// allow to only show some effects specified in the option
local all_effects_XX "`het_effects'"
local test_eff_XX : subinstr local het_effects " " ",", all 
local count_length_XX: word count `all_effects_XX'
if `count_length_XX'==1{
	local max_test_eff_XX=`test_eff_XX'
}
else{
local max_test_eff_XX = max(`test_eff_XX')
}
	if `max_test_eff_XX'>`=l_XX'{
	* error if specified effects not matching with those actually calculated	
		di as error ""
		di as error "You specified some numbers in predict_het that exceed the number of effects possible to estimate!"
		di as error "Please specify only numbers that are smaller or equal to the number you request in effects()."
		di as input _continue ""
		exit
	}
}

local all_effects_XX_e "`all_effects_XX'" // Modif Felix (delta_XX error)

// Loop the procedure over all requested effects for which potential heterogeneity should be predicted
foreach i in `all_effects_XX'{
	
capture drop Yg_Fg_min_1_XX
capture drop Yg_Fg_min_2_XX
capture drop Yg_Fg_`i'_XX
capture drop prod_het_`i'_XX
capture drop interact_var_XX
capture drop group_interact_var_XX
capture drop d_sq_group_XX	

// Defining the parts needed for the long difference 
* Yg,Fg−1
gen Yg_Fg_min1_XX_temp=outcome_non_diff_XX if time_XX==F_g_XX-1
bys group_XX: gegen Yg_Fg_min_1_XX=mean(Yg_Fg_min1_XX_temp)
capture drop Yg_Fg_min1_XX_temp

* Yg,Fg−2 
if "`trends_lin'" != ""{
gen Yg_Fg_min2_XX_temp=outcome_non_diff_XX if time_XX==F_g_XX-2
bys group_XX: gegen Yg_Fg_min_2_XX=mean(Yg_Fg_min2_XX_temp)
capture drop Yg_Fg_min2_XX_temp
}

* Yg,Fg−1+ℓ
gen Yg_Fg_`i'_XX_temp=outcome_non_diff_XX if time_XX==F_g_XX-1+`i'
bys group_XX: gegen Yg_Fg_`i'_XX=mean(Yg_Fg_`i'_XX_temp)
capture drop Yg_Fg_`i'_XX_temp

// Now generate Sg*(Yg,Fg−1+ℓ − Yg,Fg−1) 
gen prod_het_`i'_XX=(Yg_Fg_`i'_XX - Yg_Fg_min_1_XX)
if "`trends_lin'" != ""{
	replace prod_het_`i'_XX=prod_het_`i'_XX-`i'*(Yg_Fg_min_1_XX-Yg_Fg_min_2_XX)
}
replace prod_het_`i'_XX=S_g_het_XX*prod_het_`i'_XX

* keep one observation by group to not artificially increase sample
bys group_XX: replace prod_het_`i'_XX = . if switcher_tag_XX != `i' // Change by DAC to fix issue with the number of units

* F_g_XX#d_sq_XX#S_g_het_XX
gegen d_sq_group_XX=group(d_sq_XX)

//Define clusters for the SE in the predict_het in case the predict_het_hc2bm option is specified
if "`cluster'" != ""{
	local cluster_het `cluster'
}
if "`cluster'" == ""{
	local cluster_het group_XX
}
//Define SE type according to if predict_het_hc2bm was specified or not
if "`predict_het_hc2bm'" == ""{
	local ses = "hc2"
}
if "`predict_het_hc2bm'" != ""{
	local ses = "hc2 `cluster_het', dfadjust"
} 


// Run regression of interest 
if "`trends_nonparam'" == "" {
reg prod_het_`i'_XX `predict_het_good' F_g_XX#d_sq_group_XX#S_g_XX [aw=N_gt_XX] if F_g_XX-1+`i'<=T_g_XX, level(`ci_level') vce(`ses')
}
if "`trends_nonparam'" != "" {
gegen trends_nonparam_temp_XX=group(`trends_nonparam')	
reg prod_het_`i'_XX `predict_het_good' F_g_XX#d_sq_group_XX#S_g_XX#trends_nonparam_temp_XX [aw=N_gt_XX] if F_g_XX-1+`i'<=T_g_XX, level(`ci_level') vce(`ses')
capture drop trends_nonparam_temp_XX
}

// Retrieve the results from the regression
forvalues j=1/`:word count `predict_het_good''{
scalar beta_het_`j'_eff`i'_hat_XX=_b["`: word `j' of `predict_het_good''"]
scalar se_het_`j'_eff`i'_hat_XX=_se["`: word `j' of `predict_het_good''"]
scalar lb_het_`j'_eff`i'_hat_XX=_b["`: word `j' of `predict_het_good''"]+invt(e(N)-e(df_m)-1,0.025)*_se["`: word `j' of `predict_het_good''"]
scalar ub_het_`j'_eff`i'_hat_XX=_b["`: word `j' of `predict_het_good''"]+invt(e(N)-e(df_m)-1,0.975)*_se["`: word `j' of `predict_het_good''"]
scalar t_het_`j'_eff`i'_hat_XX=scalar(beta_het_`j'_eff`i'_hat_XX)/scalar(se_het_`j'_eff`i'_hat_XX)
}
scalar N_het_`i'_hat_XX=e(N)


// Add F-test on joint significance
test `predict_het_good'
scalar p_het_`i'_hat_XX=r(p)
scalar F_het_`i'_hat_XX=r(F)
}
}

// Output Part of the predict_het option

// Generate results matrix
foreach i in `all_effects_XX'{
matrix effect_het_`i'_XX=J(`:word count `predict_het_good'',6,.)

local effect_het_rownames_`i'_XX ""

forvalue j=1/`:word count `predict_het_good''{
matrix effect_het_`i'_XX[`j',1]=scalar(beta_het_`j'_eff`i'_hat_XX)
matrix effect_het_`i'_XX[`j',2]=scalar(se_het_`j'_eff`i'_hat_XX)
matrix effect_het_`i'_XX[`j',3]=scalar(t_het_`j'_eff`i'_hat_XX)
matrix effect_het_`i'_XX[`j',4]=scalar(lb_het_`j'_eff`i'_hat_XX)
matrix effect_het_`i'_XX[`j',5]=scalar(ub_het_`j'_eff`i'_hat_XX)
matrix effect_het_`i'_XX[`j',6]=scalar(N_het_`i'_hat_XX)

// Generate local with rownames
local effect_het_rownames_`i'_XX "`effect_het_rownames_`i'_XX' "`: word `j' of `predict_het_good''""
}

// Assign names
matrix colnames effect_het_`i'_XX= "Estimate" "SE" "t" "LB CI" "UB CI" "N"
matrix rownames effect_het_`i'_XX=`effect_het_rownames_`i'_XX'

}	


// Output the table
display _newline
di "{hline 80}"
di _skip(25) "{bf:Predicting effect heterogeneity}"
if "`by'" !=""{
* Add description with the by variable level if by is specified		
di _skip(35) "{bf:By: `by' = `val_lab_int_XX'}"
}	
di "{hline 80}"

// output only the specified effects
foreach i in `all_effects_XX'{
display _newline
di "{hline 80}"	
di _skip(37) "{bf:Effect_`i'}"
di "{hline 80}"
matlist effect_het_`i'_XX
di "{hline 80}"
di as text "{it:Test of joint nullity of the estimates : p-value =} " p_het_`i'_hat_XX
}



*****************************************************
// Note: We need a step to ensure that we only do this for the number of pre peridos that are available in the data -> For now just allow as mayn placebos as specified

// Note: we need to specify "all", correct that in the help file and fix the issue when we specify a numlist

if `=l_placebo_XX'>0{
local all_effects_XX_dyn "`all_effects_XX'"
qui{		
if "`het_effects'"=="all"{
local all_effects_XX ""	
// Take all effects corresponding to the number of requested effect estimates
forvalues i=1/`=l_placebo_XX'{
local all_effects_XX "`all_effects_XX' `i'"
}
}

if "`het_effects'"!="all"{ 
// allow to only show some effects specified in the option
local all_effects_XX "`het_effects'"
local test_eff_XX : subinstr local het_effects " " ",", all 
local count_length_XX: word count `all_effects_XX'
if `count_length_XX'==1{
	local max_test_eff_XX=`test_eff_XX'
}
else{
local max_test_eff_XX = max(`test_eff_XX')
}
	if `max_test_eff_XX'>`=l_placebo_XX'{
	* error if specified effects not matching with those actually calculated	
		di as error ""
		di as error "You specified some numbers in predict_het that exceed the number of placebos possible to estimate!"
		di as error "Please specify only numbers that are smaller or equal to the number you request in placebo()."
		di as input _continue ""
		exit
	}
}

local all_effects_XX_pl "`all_effects_XX'" // Modif Felix (delta_XX error)

foreach i in `all_effects_XX'{

// Modif Felix: Yg,Fg−1-ℓ
gen Yg_Fg_pl_`i'_XX_temp=outcome_non_diff_XX if time_XX==F_g_XX-1-`i'
bys group_XX: gegen Yg_Fg_pl_`i'_XX=mean(Yg_Fg_pl_`i'_XX_temp)
capture drop Yg_Fg_pl_`i'_XX_temp

// Modif Felix: Now generate Sg*(Yg,Fg−1-ℓ − Yg,Fg−1) 
gen prod_het_pl_`i'_XX=(Yg_Fg_pl_`i'_XX - Yg_Fg_min_1_XX)
if "`trends_lin'" != ""{
	replace prod_het_pl_`i'_XX=prod_het_pl_`i'_XX-`i'*(Yg_Fg_min_1_XX-Yg_Fg_min_2_XX)
}
replace prod_het_pl_`i'_XX=S_g_het_XX*prod_het_pl_`i'_XX

* keep one observation by group to not artificially increase sample
bys group_XX: replace prod_het_pl_`i'_XX = . if switcher_tag_XX != `i' // Change by DAC to fix issue with the number of units

// Run regression of interest 
if "`trends_nonparam'" == "" {
reg prod_het_pl_`i'_XX `predict_het_good' F_g_XX#d_sq_group_XX#S_g_XX [aw=N_gt_XX] if F_g_XX-1+`i'<=T_g_XX, level(`ci_level') vce(`ses') // leave the F_g_XX-1+`i'<=T_g_XX condition as it is to ensure that we only use treated groups or is this already taken into account by the S_g_XX?
}
if "`trends_nonparam'" != "" {
gegen trends_nonparam_temp_XX=group(`trends_nonparam')	
reg prod_het_pl_`i'_XX `predict_het_good' F_g_XX#d_sq_group_XX#S_g_XX#trends_nonparam_temp_XX [aw=N_gt_XX] if F_g_XX-1+`i'<=T_g_XX, level(`ci_level') vce(`ses')
capture drop trends_nonparam_temp_XX
}

// Retrieve the results from the regression
forvalues j=1/`:word count `predict_het_good''{
scalar beta_het_`j'_pl`i'_hat_XX=_b["`: word `j' of `predict_het_good''"]
scalar se_het_`j'_pl`i'_hat_XX=_se["`: word `j' of `predict_het_good''"]
scalar lb_het_`j'_pl`i'_hat_XX=_b["`: word `j' of `predict_het_good''"]+invt(e(N)-e(df_m)-1,0.025)*_se["`: word `j' of `predict_het_good''"]
scalar ub_het_`j'_pl`i'_hat_XX=_b["`: word `j' of `predict_het_good''"]+invt(e(N)-e(df_m)-1,0.975)*_se["`: word `j' of `predict_het_good''"]
scalar t_het_`j'_pl`i'_hat_XX=scalar(beta_het_`j'_pl`i'_hat_XX)/scalar(se_het_`j'_pl`i'_hat_XX)
}
scalar N_het_pl_`i'_hat_XX=e(N)

// Add F-test on joint significance
test `predict_het_good'
scalar p_het_pl_`i'_hat_XX=r(p)
scalar F_het_pl_`i'_hat_XX=r(F)
}
}

// Generate results matrix
foreach i in `all_effects_XX'{
matrix effect_het_pl_`i'_XX=J(`:word count `predict_het_good'',6,.)

local effect_het_pl_rownames_`i'_XX ""

forvalue j=1/`:word count `predict_het_good''{
matrix effect_het_pl_`i'_XX[`j',1]=scalar(beta_het_`j'_pl`i'_hat_XX)
matrix effect_het_pl_`i'_XX[`j',2]=scalar(se_het_`j'_pl`i'_hat_XX)
matrix effect_het_pl_`i'_XX[`j',3]=scalar(t_het_`j'_pl`i'_hat_XX)
matrix effect_het_pl_`i'_XX[`j',4]=scalar(lb_het_`j'_pl`i'_hat_XX)
matrix effect_het_pl_`i'_XX[`j',5]=scalar(ub_het_`j'_pl`i'_hat_XX)
matrix effect_het_pl_`i'_XX[`j',6]=scalar(N_het_pl_`i'_hat_XX)

// Generate local with rownames
local pl_het_rownames_`i'_XX "`pl_het_rownames_`i'_XX' "`: word `j' of `predict_het_good''""
}

// Assign names
matrix colnames effect_het_pl_`i'_XX= "Estimate" "SE" "t" "LB CI" "UB CI" "N"
matrix rownames effect_het_pl_`i'_XX=`pl_het_rownames_`i'_XX'

}

// Output the table
display _newline
di "{hline 80}"
di _skip(18) "{bf:Predicting effect heterogeneity - Placebos}"
if "`by'" !=""{
* Add description with the by variable level if by is specified		
di _skip(35) "{bf:By: `by' = `val_lab_int_XX'}"
}	
di "{hline 80}"

// output only the specified effects
foreach i in `all_effects_XX'{
display _newline
di "{hline 80}"	
di _skip(37) "{bf:Placebo_`i'}"
di "{hline 80}"
matlist effect_het_pl_`i'_XX
di "{hline 80}"
di as text "{it:Test of joint nullity of the estimates : p-value =} " p_het_pl_`i'_hat_XX
}
} // End condition that at least one placebo specified
else{
	noi di "No placebos were specified so predict_het will also not report placebo estimates"
}

*****************************************************


}





///// Store all the ereturns after the last reg was called 
ereturn clear 

// Modif Felix: Put ereturn post before all the other ereturns and adjust row/col names

/* -> Problems with format -> didmgt_results_no_avg_XX does not take out missings, didmgt_Var_all_XX does, so we would need to change one of them to make it work (plus we create b V with all those missings, would need to change it too)
	// Adjust for missings
	local minus ""
	
	if "`missing_eff_XX'" != ""{
		foreach i of numlist `missing_eff_XX'{
			local minus = "`minus' Effect_`i'"
		}
	}
	
	if "`missing_pl_XX'" != ""{
		foreach i of numlist `missing_pl_XX'{
			local minus = "`minus' Placebo_`i'"
		}
	}
	
local minus " `minus' Av_tot_eff"
local rownames_alt : list rownames-minus
*/

local minus Av_tot_eff
local rownames_alt : list rownames-minus

////// Integration with esttab
if "`trends_lin'"==""{ // Modif Felix (delta_XX error)
	matrix b=(didmgt_results_no_avg_XX, scalar(delta_XX))
}
else if "`trends_lin'"!=""{ // Modif Felix (delta_XX error) -> Also check if I need to change the way in which the variances are retrieved from didmgt_Var_all_XX -> no because avg effect not in there!
	matrix b=(didmgt_results_no_avg_XX)
}	

local nc = colsof(b)
matrix V = J(`nc', `nc', 0)
if "`bootstrap'"==""{ // Modif Felix: Adapt Matrix for bootstrap

if "`trends_lin'"==""{ // Modif Felix (delta_XX error)
	forv i=1/`=`nc'-1' {
		forv j =1/`=`nc'-1' {
			mat V[`i', `j'] = didmgt_Var_all_XX[`i', `j']
		}
	}
}
else if "`trends_lin'"!=""{ // Modif Felix (delta_XX error)
forv i=1/`=`nc'' {
		forv j =1/`=`nc'' {
			mat V[`i', `j'] = didmgt_Var_all_XX[`i', `j']
		}
	}
}

}
if "`bootstrap'"!=""{ // Modif Felix: Adapt Matrix for bootstrap
	mat V=didmgt_Var_all_XX
}

// Note Felix: Drop this bootstrap distiction after we reeplaced  didmgt_Var_all_XX by bs_var_cov_XX above so we get the "correct" variance covariance matrix with and without bootstrap.

// Modif Felix: Adjust for cases where SE cant be computed -> with bootstrap we already have the Avg_tot_eff in there
if "`trends_lin'"==""{ // Modif Felix (delta_XX error)
	if "`bootstrap'"==""{
		if scalar(se_XX)!=. {
			mat V[`nc', `nc'] = scalar(se_XX)^2
		}
		if scalar(se_XX)==. {
			mat V[`nc', `nc'] = 0
		}
	}
}	


if "`trends_lin'"==""{ // Modif Felix (delta_XX error)
	matrix colnames b= `rownames_alt' Av_tot_eff
	matrix rownames V = `rownames_alt' Av_tot_eff
	matrix colnames V = `rownames_alt' Av_tot_eff
}
else if "`trends_lin'"!=""{ // Modif Felix (delta_XX error)
	matrix colnames b= `rownames_alt'
	matrix rownames V = `rownames_alt'
	matrix colnames V = `rownames_alt'
}

// Modif Felix (delta_XX error) -> does the following section also need to be changed??? -> yes
if "`predict_het'" != "" {
	local het_rown ""
	
	if "`trends_lin'"==""{ // Modif Felix (delta_XX error)
		local rownames_het "`rownames_alt' Av_tot_eff"
	}
	else if "`trends_lin'"!=""{ // Modif Felix (delta_XX error)
		local rownames_het "`rownames_alt'"
	}
	
	foreach v in `rownames_het' {
		local het_rown "`het_rown' `1':`v'"
	}
	
	foreach i in `all_effects_XX_e'{ // Modif Felix (delta_XX error)
		local nc = rowsof(effect_het_`i'_XX)
		local nv = rowsof(V)
		mat V`i' = J(`nc', `nc', 0)
		forv j = 1/`nc' {
			mat b = (b, scalar(effect_het_`i'_XX[`j', 1]))
			mat V`i'[`j', `j'] = scalar(effect_het_`i'_XX[`j', 2])^2
		}
		mat V = (V, J(`nv',`nc', 0)\J(`nc', `nv', 0), V`i')
		local rnames ""
		foreach rn in `effect_het_rownames_`i'_XX' {
			local rnames "`rnames' Effect_`i':`rn'"
		}
		local het_rown "`het_rown' `rnames'"
	}	
	
	// split this to add both effects and placebos to the ereturn
	foreach i in `all_effects_XX_pl'{ // Modif Felix (delta_XX error)
		local nc = rowsof(effect_het_pl_`i'_XX)
		local nv = rowsof(V)
		mat V`i' = J(`nc', `nc', 0)
		forv j = 1/`nc' {
			mat b = (b, scalar(effect_het_pl_`i'_XX[`j', 1]))
			mat V`i'[`j', `j'] = scalar(effect_het_pl_`i'_XX[`j', 2])^2
		}
		mat V = (V, J(`nv',`nc', 0)\J(`nc', `nv', 0), V`i')
		local rnames ""
		foreach rn in `pl_het_rownames_`i'_XX' {
			local rnames "`rnames' Placebo_`i':`rn'"
		}
		local het_rown "`het_rown' `rnames'"
	}
	
	mat coln b = `het_rown'
	mat rown V = `het_rown'
	mat coln V = `het_rown'
}

// Felix -> Lets add cap and print error message when we are in the case with some missings in the matrices (cases where we have missings in the output table)
cap ereturn post b V 
if _rc!=0{
	di ""
	di as error "Warning: some of the effects/placebos could not be computed so e(b)/e(V) will not be defined!"
}

// Felix: Continue by adding Av_tot_eff to the Var/Cov matrix

forvalue i=1/`=l_XX'{
capture ereturn scalar Effect_`i' = scalar(DID_`i'_XX)
capture ereturn scalar N_effect_`i' = N_effect_`i'_XX
capture ereturn scalar N_switchers_effect_`i' = N_switchers_effect_`i'_XX
capture ereturn scalar se_effect_`i'=scalar(se_`i'_XX)
}
ereturn local effects `"`str_effects'"'

// Modif Felix 12.05.25
if (l_XX>1&all_Ns_not_zero==l_XX&all_delta_not_zero==l_XX&"`normalized'"!="")|(l_XX>1&all_Ns_not_zero==l_XX&"`normalized'"==""){
capture ereturn scalar p_jointeffects=scalar(p_jointeffects)
}

if "`effects_equal'" != ""{
capture ereturn scalar p_equality_effects=scalar(p_equality_effects)
}

if l_placebo_XX!=0{
ereturn local placebo `"`str_placebo'"'
forvalue i=1/`=l_placebo_XX'{
capture ereturn scalar Placebo_`i' = scalar(DID_placebo_`i'_XX)
capture ereturn scalar N_placebo_`i' = N_placebo_`i'_XX
capture ereturn scalar N_switchers_placebo_`i' = N_switchers_placebo_`i'_XX
capture ereturn scalar se_placebo_`i'=scalar(se_placebo_`i'_XX)
}
if (l_placebo_XX>1&all_Ns_pl_not_zero==l_placebo_XX&all_delta_pl_not_zero==l_placebo_XX&"`normalized'"!="")|(l_placebo_XX>1&all_Ns_pl_not_zero==l_placebo_XX&"`normalized'"==""){
capture ereturn scalar p_jointplacebo=scalar(p_jointplacebo)
}
}

if "`trends_lin'"==""{
capture ereturn scalar Av_tot_effect = scalar(delta_XX)
capture ereturn scalar N_avg_total_effect = scalar(N_effect_XX)
capture ereturn scalar N_switchers_effect_average = N_switchers_effect_XX
capture ereturn scalar se_avg_total_effect = se_XX
}

ereturn local cmd "did_multiplegt_dyn"
ereturn matrix estimates=didmgt_b2 
//ereturn matrix didmgt_variances=didmgt_var2
ereturn matrix variances=compatibility_variances

if "`predict_het'"!=""{
if `=l_placebo_XX' == 0 {
	foreach i in `all_effects_XX'{
	ereturn matrix effect_het_`i'_XX=effect_het_`i'_XX
	}	
} 
else {
	foreach i in `all_effects_XX_dyn'{
	ereturn matrix effect_het_`i'_XX=effect_het_`i'_XX
	}	
	foreach i in `all_effects_XX_pl'{
	ereturn matrix effect_het_pl_`i'_XX=effect_het_pl_`i'_XX
	}	
}
}

///// Option that shows a table with weights attached to each normalized effect
if "`normalized_weights'" != "" {
	qui {
		// Error as normalized_weights only works in combination with normalized
		if "`normalized'" == "" {
			noi di as err "normalized option required to compute normalized_weights"
			di as input _continue ""
			exit
		}

	// Set up the matrix for the output table
	mat define weight_mat = J(`=l_XX + 1', `=l_XX', .)
	local cols ""

	forv i = 1/`=l_XX'{	
		local m = "ℓ=`i'"
		local cols `cols' `m'
		gen N_gt_`i'_temp_XX = N_gt_XX if time_XX == F_g_XX - 1 + `i' & `i' <= L_g_XX & N_gt_control_`i'_XX > 0 & N_gt_control_`i'_XX != . & diff_y_`i'_XX != .
		gegen N_gt_`i'_XX = mean(N_gt_`i'_temp_XX), by(group_XX)
		drop N_gt_`i'_temp_XX
		forv k = 0/`=`i' - 1' {
			// Visualization by k
			local row = `k' + 1
			
			// Compute the delta_ℓ_k, if the continuous option is specified the original treatment values are used
			if `continuous'==0{
			gen delta_`i'_`k' = abs(treatment_XX - d_sq_XX) if time_XX == F_g_XX - 1 + `i' - `k' & F_g_XX - 1 + `i' <= T_g_XX
			}
			else if `continuous'>0{
			gen delta_`i'_`k' = abs(treatment_XX_orig - d_sq_XX_orig) if time_XX == F_g_XX - 1 + `i' - `k' & F_g_XX - 1 + `i' <= T_g_XX
			}
			replace delta_`i'_`k' = delta_`i'_`k' * N_gt_`i'_XX
			// Filling the weight matrix 
			sum delta_`i'_`k'
			mat weight_mat[`row', `i'] = (`r(sum)' / scalar(delta_D_`i'_global_XX)) / scalar(N_switchers_effect_`i'_XX)
			//mat weight_mat[`row', `i'] = (`r(sum)' / `denom_`i'_norm_XX')
		}		
		drop N_gt_`i'_XX
	}
	
	// Generating the row names 
	// Visualization by k 
	local rows ""
	forv j = 1/`=l_XX' {
			local r = "k=`=`j'-1'"
			local rows `rows' `r'
	}	

	local rows `rows' "Total"
	
	// Fill the values for the displayed table
	matrix define mat_temp = J(`=l_XX', `=l_XX', 0)	
	forv i = 1/`=l_XX' {
		forv j = 1/`=l_XX' {
			if weight_mat[`i', `j'] != . {
				mat mat_temp[`i', `j'] = weight_mat[`i', `j']
			}
		}
	}	
	
	// Getting the correct order
	matrix define mat_total = mat_temp' * J(`=l_XX', 1, 1)
	forv i = 1/`=l_XX' {
		mat weight_mat[`=l_XX+1', `i'] = mat_total[`i', 1]
	}
	
	// Add row and column names
	mat rown weight_mat = `rows'
	mat coln weight_mat = `cols'	
	}

	// Output the table
	display _newline
	di "{hline 70}"
	di _skip(15) "{bf:Weights on treatment lags}"
	if "`by'" !=""{	
	* Add description with the by variable level if by is specified	
	di _skip(25) "{bf:By: `by' = `val_lab_int_XX'}"
	}	
	di "{hline 70}"
	matlist weight_mat, format(%9.4fc) lines(rowtotal)
	display _newline		
}

// Saving data for further options as save sample drops variables 
qui tempfile data_predesign
qui save `data_predesign', replace


///// save_sample option that allows to see which groups were included in the estimation and if they were switcher in/out ot control

if "`save_sample'" != "" {
	qui {
		// keeping only group, time and switcher status (if not missing)
		keep if !missing(`2') & !missing(`3') 
		keep `2' `3' S_g_XX switcher_tag_XX
		
		if "`by'"==""{
			
		// redefine S_g_XX to show if group is switcher in/out or control	
		rename S_g_XX _did_sample
		rename switcher_tag_XX _effect
		replace _did_sample = -1 if _did_sample == 0
		replace _did_sample = 0 if _did_sample == .
		cap label drop switch_lab_XX 
		label define switch_lab_XX 0 "Never-switcher" 1 "Switcher-in" -1 "Switcher-out"
		label values _did_sample switch_lab_XX
		
		// Save dataset to be in the memory at the end
		tempfile did_sample
		save `did_sample', replace
		}
		
		// Same procedure, but with by we get multiple samples
		else if "`by'"!=""{
			
		// redefine S_g_XX to show if group is switcher in/out or control	
		rename S_g_XX _did_sample_`l_by'
		rename switcher_tag_XX _effect_`l_by'
		replace _did_sample_`l_by' = -1 if _did_sample_`l_by' == 0
		replace _did_sample_`l_by' = 0 if _did_sample_`l_by' == .
		cap label drop switch_lab_XX 
		label define switch_lab_XX 0 "Never-switcher" 1 "Switcher-in" -1 "Switcher-out"
		label values _did_sample_`l_by' switch_lab_XX
		
		// Save dataset to be in the memory at the end
		tempfile did_sample_`l_by'
		save `did_sample_`l_by'', replace
		}
	}
}

///// design option which shows the different treatment paths that switcher groups follow

if "`design'" != "" {	
	qui {
	
	// Releaod the data before it was changed in save_sample
	use `data_predesign', clear
	
	// Error message if the arguments in the option were specified wrong
	if strpos("`design'", ",") == 0 {
		di as error ""
		di as error "Syntax error in design option"
		di as error "Comma required"
		di as input _continue ""
		
		exit
	}
	
	// Fetch the arguments 
	local des_p = strtrim(substr("`design'", 1, strpos("`design'", ",") - 1))
	local des_path = strtrim(substr("`design'", strpos("`design'", ",") + 1, .))
	local des_n = l_XX 
	
	// When no percentage value is specified (for how many paths should be shown) it is set to a default of 100%
	if missing("`des_p'") {
		local des_p = 1
	}
	local des_per = `des_p' * 100
	local des_per : di %9.2fc `des_per'
	
	// keep periods up to ℓ periods after the first switch
	gen F_g_plus_n_XX = F_g_XX + `des_n' - 1
	keep if time_XX >= F_g_XX - 1 & time_XX <= F_g_plus_n_XX
	sort group_XX time_XX
	bys group_XX : gen time_l_XX = _n

	// Aggregate weights by group 
	keep group_XX time_l_XX weight_XX treatment_XX F_g_XX
	{
		if "`weight'" != "" {
			gegen g_weight_XX = sum(weight_XX), by(group_XX)
		}
		else {
			gen g_weight_XX = 1
		}
	}
	drop weight_XX
	// Reshape wide to represent the treatment path by time
	greshape wide treatment_XX, i(group_XX g_weight_XX F_g_XX) j(time_l_XX)
	
	// Drop missing treatments 
	foreach v of varlist treatment_XX* {
		drop if `v' == .
	}
	
	// Remove non switchers 
	count if F_g_XX > T_max_XX
	local non_switchers = r(N)
	drop if F_g_XX > T_max_XX
	
	// Creating varibale to store number of groups per treatment path and collapsing
	gen N_XX = 1
	sum g_weight_XX
	gen N_w_XX =  (g_weight_XX * N_XX) / r(sum)
	drop group_XX g_weight_XX
	gcollapse (sum) N_XX N_w_XX, by(treatment_XX*)
	order N* treat*	
	sum N_XX
	local tot_switch = r(sum)
	
	// Keep the observations amounting to p% of the detected treatment paths 
	gen neg_N_XX = -N_XX
	sort neg_N_XX treatment_XX*
	gen cum_sum_XX = sum(N_w_XX)
	gen in_table_XX = (cum_sum_XX <= `des_p')
	sort in_table_XX cum_sum_XX
	bys in_table_XX: gen id_XX = _n
	
	// Keep all observations up to the first exceeding the p%	
	keep if (in_table_XX == 1) | (in_table_XX == 0 & id_XX == 1) 	
	// Store the final % of groups included by the design option
	{
		if `des_p' < 1 {
			sum cum_sum_XX if in_table == 0
			local last_p : di %9.2fc 100 * r(min)
		}
		else {
			local last_p : di %9.2fc 100
		}
	}
	
	sort neg_N_XX treatment_XX*
	drop neg_N_XX in_table_XX id_XX cum_sum_XX
	
	// Prepare matrix for the output table
	mat define desmat = J(`=_N', `= 2 + 1 + `des_n'', .)
	local colnames
	label var N_XX "#Groups"
	label var N_w_XX "%Groups"
	replace N_w_XX=100*N_w_XX
	* display in % terms
	
	// Generate the column/row names and fill treatment path
	local j = 1
	foreach v of varlist _all {
		local m : var label `v'
		if strpos("`v'", "treatment_XX") > 0 {
			local k = substr("`v'", length("treatment_XX") + 1, .)
			local m = "ℓ=`=`k'-1'"
		}
		local colnames `colnames' `m'	
		
		// Fill treatement path
		forv i = 1/`=_N' {
			matrix desmat[`i', `j'] = `v'[`i']
		}
		local j = `j' + 1
		
	}
	local rownames
	forv i = 1/`=_N' {
		local rownames `rownames' "TreatPath`i'"
	}
	foreach v in col row {
		mat `v'n desmat = ``v'names'
	}
	}

	// output in console
	if "`des_path'" == "console" {		
		display _newline
		di "{hline 80}"
		di _skip(10) "{bf:Detection of treatment paths - `des_n' periods after first switch}"
		if "`by'" !=""{	
			* Add description with the by variable level if by is specified	
			di _skip(35) "{bf:By: `by' = `val_lab_int_XX'}"
		}	
		di "{hline 80}"
		matlist desmat, format(%9.4gc)
		di "{hline 80}"
		di "{it: Treatment paths detected in at least `=strtrim("`des_per'")'% of the switching groups `tot_switch' for which `effects' effects could be estimated.}"
		di "{it: Total % = `=strtrim("`last_p'")' %}"
		
		// Get values for the reference line that explains how to read the table 
		qui {
			local n_groups = N_XX[1]
			local d_start = treatment_XX1[1]
			if `des_n' == 1 {
				local d_vec : di %9.2gc treatment_XX2[1]
				local d_vec = strtrim("`d_vec'")
			}
			else {
				local d_vec = "vector ("
				forv j = 1/`des_n' {
					local k : di %9.2gc treatment_XX`=`j' + 1'[1]
					local k = strtrim("`k'")
					local d_vec = "`d_vec'`k', "
				}
				local d_vec = substr("`d_vec'", 1, length("`d_vec'")-2) + ")"
			}
		}
		di as text "{it: Design interpretation (first row):}"
		di as text "{it: `n_groups' groups started with treatment `d_start' and then experienced treatment `d_vec'.}"		
	}
	
	// output as excel
	else if "`des_path'" != "console" & "`des_path'" !=  "" {
		if "`by'"==""{
		qui {
			putexcel set "`des_path'", replace sheet("Design", replace)
			putexcel A1 = matrix(desmat, names)
		}
		 di "{bf:Design exported to `des_path'}"
		}
		else if "`by'"!=""{
			qui {
			* Create new sheet for each by level when by is specified	
			putexcel set "`des_path'", modify sheet("Design `by' = `val_lab_int_XX'", replace)
			putexcel A1 = matrix(desmat, names)
			}
		 di "{bf:Design exported to `des_path'}"
	}  
	}
}


///// Date first switch option: produces a table showing the number of groups switching for the first time by period

if "`date_first_switch'" != "" {
	qui {

	// Reload the data before it was changed in save_sample
	use `data_predesign', clear
	
	// Fetch the arguments 
	local dfs_opt = strtrim(substr("`date_first_switch'", 1, strpos("`date_first_switch'", ",") - 1))
	local dfs_path = strtrim(substr("`date_first_switch'", strpos("`date_first_switch'", ",") + 1, .))
	
	// Error message if the arguments in the option were specified wrong
	if "`dfs_opt'" != "" & "`dfs_opt'" != "by_baseline_treat" {
		di as error ""
		di as error "Only option {bf:by_baseline_treat} allowed"
		di as input _continue ""
		exit
	}
	
	// Drop non switchers
	drop if F_g_XX == T_max_XX+1 | F_g_XX == .
	keep if time_XX == F_g_XX 
	*Keep just 1 observation per group
	keep `2' `3' F_g_XX d_sq_XX
	}
	
	// When by_baseline_treat is not specified
	if missing("`dfs_opt'") {
		qui {
		// collapse the number of groups by time
		gcollapse (count) `2', by(`3')
		tostring `3', replace
		// generate the share of each group
		sum `2'
		local tot_s = r(sum)
		gen share_XX = `2'/`tot_s' * 100
		
		// make matrix with the number and share of groups by time
		mkmat `2' share_XX, matrix(dfs) rownames(`3')
		mat colnames dfs = #Groups %Groups
		}

		// output in console
		if "`dfs_path'" == "console" {		
			display _newline
			di "{hline 48}"
			di  _skip(10) "{bf:Switching dates}"
			if "`by'" !=""{	
				* Add description with the by variable level if by is specified
				di _skip(10) "{bf:By: `by' = `val_lab_int_XX'}"
			}	
			di "{hline 48}"
			matlist dfs, format(%9.4gc) rowtitle("Any status quo treat.") twidth(25) // New text
			di "{hline 48}"		

			ereturn scalar switch_dates = `=_N'
		}
		// output as excel
		else if "`dfs_path'" != "console" & "`dfs_path'" !=  "" {
			if "`by'"==""{
				qui putexcel set "`dfs_path'", replace sheet("Switching dates", replace)
				qui putexcel A1 = matrix(dfs, names)
				di "{bf:Switching dates exported to `dfs_path'}"
			}
			else if "`by'"!=""{
				* Create new sheet for each by level when by is specified	
				qui putexcel set "`dfs_path'", modify sheet("Switching dates `by' = `val_lab_int_XX'", replace)
				qui putexcel A1 = matrix(dfs, names)
				di "{bf:Switching dates exported to `dfs_path'}"
			}
		}		
	}
	
	// When by_baseline_treat is specified
	else {
		// collapse, but this time by time and status quo treatment
		qui {
		gcollapse (count) `2', by(`3' d_sq_XX)
		tostring `3', replace
		gegen tot_share_XX = sum(`2'), by(d_sq_XX)
		gen share_XX = `2'/tot_share_XX * 100
		sort d_sq_XX `3'
		}
		
		display _newline
		if "`dfs_path'" == "console" {
			di "{hline 48}"
			di _skip(10) "{bf:Switching dates}"
			di "{hline 48}"
		}
		
		// make matrix with the number and share of groups by time, one for each level of status quo treatment
		qui {
			tostring d_sq_XX, replace
			levelsof d_sq_XX, local(levels_d_sq_XX)
			local i = 1
		}
		foreach m of local levels_d_sq_XX {
			qui {
				mkmat `2' share_XX if d_sq_XX == "`m'", matrix(dfs`i') rownames(`3')
				mat colnames dfs`i' = #Groups %Groups
				sum `2'
				local tot_s = r(sum)
			}			
			// output in console
			if "`dfs_path'" == "console" {		
				matlist dfs`i', format(%9.4gc) rowtitle("Status quo treat. = `m'") twidth(25) // New
				di "{hline 48}"
				qui {
					sum `2' if d_sq_XX == "`m'"
					local tot_s = r(sum)
					count if d_sq_XX == "`m'"
					ereturn scalar switch_dates_Dsq_`l_by' = `r(N)' // New
				}
			}
			// output as excel
			else if "`dfs_path'" != "console" & "`dfs_path'" !=  "" {
				qui {
				if `i' == 1 {
					local rep_opt "replace"
				}
				else {
					local rep_opt "modify"
				}
				
				if "`by'"==""{
					putexcel set "`dfs_path'", `rep_opt' sheet("Switch. Dates Base treat `m'", replace)
					putexcel A1 = matrix(dfs`i', names)
					noi di _newline
				}
				else if "`by'"!=""{
					* Create new sheet for each by level when by is specified
					putexcel set "`dfs_path'", modify sheet("Dates Base treat=`m' & `by' = `val_lab_int_XX'", replace) //`by' = `val_lab_int_XX'
					putexcel A1 = matrix(dfs`i', names)
					noi di _newline
				}	
			}
		}
		local i = `i' + 1			
		}
		// consol message if output is written to excel
		if "`dfs_path'" != "console" & "`dfs_path'" != "" {
			di "Switching dates exported to `dfs_path'"
		}
	}
}

// Reload the data before it was changed in save_sample
qui use `data_predesign', clear // Modif Felix: Add qui

// If by specified, reload the original data before restricting the sample with by
if "`by'" !=""{	
use "`by_data_XX'.dta", clear
local rownames ""

// Save the matrices to generate multiple graphs
matrix mat_res_XX_by_`l_by'=mat_res_XX
}

} // End of the byvar loop

////////// 9. Event study graph

///// Producing a graph
qui {

// Initialize color pattern for multiple graphs wihen by is specified 
local col1 "midblue"
local col2 "midblue" 
local col3 "red"  
local col4 "red" 
local col5 "green" 
local col6 "green"    
local col7 "magenta" 
local col8 "magenta"   
local col9 "gold"   
local col10 "gold"    
local col11 "lime" 
local col12 "lime"      

local graph_input ""
local graph_options ""
local graph_options_int ""

if "`by'" ==""|scalar(by_XX)==0{
*Version with only one event study graph	
// Translate results matrix into data			
capture drop mat_res_XX1 mat_res_XX2 mat_res_XX3 mat_res_XX4 mat_res_XX5 mat_res_XX6 mat_res_XX7
svmat mat_res_XX

// artificially replacing the result of average effect by 0
forvalue index=1/6{
replace mat_res_XX`index'=0 if mat_res_XX7==0 
}

capture drop point_estimate
capture drop se_point_estimate
capture drop lb_CI_95
capture drop up_CI_95
capture drop N
capture drop N_switchers
capture drop time_to_treat

rename mat_res_XX1 point_estimate
rename mat_res_XX2 se_point_estimate
rename mat_res_XX3 lb_CI_95
rename mat_res_XX4 up_CI_95
rename mat_res_XX5 N
rename mat_res_XX6 N_switchers
rename mat_res_XX7 time_to_treat


sort time_to_treat

// Define fixed graph options as they are implemented as flexible locals
local graph_input "(connected point_estimate time_to_treat, lpattern(solid)) (rcap up_CI_95 lb_CI_95 time_to_treat)"
local graph_options "legend(off)"

}

if "`by'" !=""{
		
// Save tempfile because we will do sorting afterwards 
tempfile before_graph_XX
save "`before_graph_XX'.dta", replace		

// Create datasets from the results matrix for each level of the by variable		
local cnt_XX=1
local merge_count=1
foreach l of local levels_byvar_XX {
use "`before_graph_XX'.dta", clear		
	
capture drop mat_res_XX_by_`l'1 mat_res_XX_by_`l'2 mat_res_XX_by_`l'3 mat_res_XX_by_`l'4 mat_res_XX_by_`l'5 mat_res_XX_by_`l'6 mat_res_XX_by_`l'7
svmat mat_res_XX_by_`l'

// artificially replacing the result of average effect by 0
forvalue index=1/6{
replace mat_res_XX_by_`l'`index'=0 if mat_res_XX_by_`l'7==0
}

capture drop point_estimate_`l'
capture drop se_point_estimate_`l'
capture drop lb_CI_95_`l'
capture drop up_CI_95_`l'
capture drop N_`l'
capture drop N_switchers_`l'
capture drop time_to_treat

rename mat_res_XX_by_`l'1 point_estimate_`l'
rename mat_res_XX_by_`l'2 se_point_estimate_`l'
rename mat_res_XX_by_`l'3 lb_CI_95_`l'
rename mat_res_XX_by_`l'4 up_CI_95_`l'
rename mat_res_XX_by_`l'5 N_`l'
rename mat_res_XX_by_`l'6 N_switchers_`l'
rename mat_res_XX_by_`l'7 time_to_treat


// pick CI color matching the line color
local col_CI=`cnt_XX'+1

// Create a local that adjusts the graph automatically for ll the by levels
local graph_input "`graph_input' (connected point_estimate_`l' time_to_treat, lpattern(solid) lcolor(`col`cnt_XX'') mcolor(`col`cnt_XX'')) (rcap up_CI_95_`l' lb_CI_95_`l' time_to_treat, lcolor(`col`col_CI''))" 

// Store value labels for the legend if they exist, show the level otherwise
ds `by', has(vallabel)
local `by'_has_vallabel_XX=r(varlist)
if "``by'_has_vallabel_XX'"!="."{
	local val_lab_int_XX "`: label ``by'_lab_XX' `l''"
} 
if "``by'_has_vallabel_XX'"=="."{
	local val_lab_int_XX "`l'"
}

// Automatcially adjust the legend for each additional graph from the by option
local graph_options_int "`graph_options_int' `cnt_XX' "`by' = `val_lab_int_XX'""

// +2 to skip the CI's to get the correct color for the next event study graph
local cnt_XX=`cnt_XX'+2

// Save the specific dataset needed for the graphs
keep point_estimate_`l' se_point_estimate_`l' lb_CI_95_`l' up_CI_95_`l' N_`l' N_switchers_`l' time_to_treat
tempfile graph_`l'_XX
keep if time_to_treat!=. 
*to allow time_to_treat as id of the dataset and use merge 1:1
save "`graph_`l'_XX'.dta", replace	

// Merge the different datasts if there is more then one event study graph to be shown
if `merge_count'==1{
	use "`graph_`l'_XX'.dta", clear
	sort time_to_treat
	tempfile graph_merge_XX
	save "`graph_merge_XX'.dta", replace
}
if `merge_count'>1{
	use "`graph_merge_XX'.dta", clear
	merge 1:1 time_to_treat using "`graph_`l'_XX'.dta"
	capture drop _merge
	tempfile graph_merge_XX
	save "`graph_merge_XX'.dta", replace
}	
local merge_count=`merge_count'+1
} // end of the differentlevels_byvar_XX loop

// Add universal graph options
local graph_options "legend(pos(6) order(`graph_options_int') rows(1))"
}	

}

// Event Study graph without any additional graph options
if ("`graph_off'"==""){
if "`graphoptions'"==""{
twoway `graph_input', xlabel(`=-l_placebo_XX'[1]`=l_XX') xtitle("Relative time to last period before treatment changes (t=0)", size(large)) title("DID, from last period before treatment changes (t=0) to t", size(large)) graphregion(color(white)) plotregion(color(white)) `graph_options'
}
// Event Study graph with additional graph options
else{
global options="`graphoptions'"
twoway `graph_input', $options
}
} // end graph_off


///// Saving the results if requested
if "`save_results'"!=""{
quietly{
keep if time_to_treat!=.
}

// keep main results
if "`by'"==""{
keep time_to_treat point_estimate se_point_estimate lb_CI_95 up_CI_95 N N_switchers
}
else if "`by'"!=""{
keep time_to_treat point_estimate* se_point_estimate* lb_CI_95* up_CI_95* N* N_switchers*
}

// Store information about the dataset
label data "Stores did_multiplegt_dyn estimates' information: Type 'notes' for details." 
local command_options_XX effects(`effects') placebo(`placebo') switchers(`switchers') `only_never_switchers' controls(`controls') trends_nonparam(`trends_nonparam') weight(`weight') `dont_drop_larger_lower' `normalized' cluster(`cluster') `same_switchers' `trends_lin' by(`by') predict_het(`predict_het') ci_level(`ci_level') design(`design') date_first_switch(`date_first_switch') continuous(`continuous') `less_conservative_se' `normalized_weights'
notes : "{bf:{ul:Date of Run:}} `c(current_date)' at `c(current_time)'"
notes : "{bf:{ul:Command Syntax:}} did_multiplegt `1' `2' `3' `4' `if' `in', `command_options_XX' "
notes : "{bf:{ul:Path of the Used Dataset:}} `dataset_name_XX' "
save "`save_results'", replace

}

display _newline
di as text "The development of this package was funded by the European Union (ERC, REALLYCREDIBLE,GA N°101043899)."

// Restoring data
restore

///// When both save_sample and by were specified merge the different datasets for each level of the by variable into one dataset that will be in memory after the command ran
if "`save_sample'" != "" { 
	if "`by'"==""{
	qui merge m:1 `2' `3' using `did_sample', gen(_merge_XX)
	qui drop if _merge_XX == 2
	qui drop _merge_XX
	}
	else if "`by'"!=""{
	foreach l of local levels_byvar_XX {	
	qui merge m:1 `2' `3' using `did_sample_`l'', gen(_merge_XX)
	qui drop if _merge_XX == 2
	qui drop _merge_XX	
	}
	}
}
end

********************************************************************************
*                                 PROGRAM 2                                    *
********************************************************************************
*Main goal: compute the variables U_Gg_plus_XX, U_Gg_minus_XX, U_Gg_var_plus_XX, and U_Gg_var_minus_XX,
*which are necessary to compute the DID_\ell estimators and their variance.

capture program drop did_multiplegt_dyn_core_new

program did_multiplegt_dyn_core_new, eclass
	version 12.0
	syntax varlist(min=4 max=4 numeric) [, effects(integer 1) placebo(integer 0) switchers(string) only_never_switchers controls(varlist numeric) trends_nonparam(varlist numeric) weight(varlist numeric) dont_drop_larger_lower NORMALIZED cluster(varlist numeric) graphoptions(string) SAVe_results(string) graph_off same_switchers same_switchers_pl drop_if_d_miss_before_first_switch trends_lin ci_level(integer 95) by(varlist numeric max=1) predict_het(string) design(string) date_first_switch(string) NORMALIZED_weights CONTinuous(string) switchers_core(string) less_conservative_se]
	qui{
		
////////// 1. Scalars initialization

///// Initializing the number of effects and placebos to estimate, depending on whether we consider switchers in or switchers out. 
///// Initializing also a scalar for whether the estimation is for switchers in or out.
	
if "`switchers_core'"=="in"{
scalar l_u_a_XX=min(`effects', L_u_XX)

if `placebo'!=0{
	scalar l_placebo_u_a_XX=min(`placebo', L_placebo_u_XX)
	}

	scalar increase_XX=1
}

if "`switchers_core'"=="out"{
scalar l_u_a_XX=min(`effects', L_a_XX)

if `placebo'!=0{
	scalar l_placebo_u_a_XX=min(`placebo', L_placebo_a_XX)
	}

scalar increase_XX=0
}

///// Initializing values of baseline treatment
levelsof d_sq_int_XX, local(levels_d_sq_XX)

////////// 2. Data preparation steps to generate variables necessary for computation of event-study effects	

///// Drop some variables we only create once
capture drop num_g_paths_0_XX
capture drop cohort_fullpath_0_XX
capture drop path_0_XX
capture drop d_fg0_XX 

///// Loop over the number of dynamic effects_equal

forvalue i=1/`=l_u_a_XX'{

///// Capture drop of variables created below

capture drop distance_to_switch_`i'_XX
capture drop never_change_d_`i'_XX
capture drop N`=increase_XX'_t_`i'_XX
capture drop N`=increase_XX'_t_`i'_XX_w
capture drop N`=increase_XX'_t_`i'_g_XX
capture drop N_gt_control_`i'_XX
capture drop diff_y_`i'_XX
capture drop dummy_U_Gg`i'_XX
capture drop U_Gg`i'_temp_XX
capture drop U_Gg`i'_XX
capture drop count`i'_core_XX
capture drop U_Gg`i'_temp_var_XX
capture drop U_Gg`i'_var_XX

capture drop never_change_d_`i'_wXX
capture drop distance_to_switch_`i'_wXX

capture drop d_fg`i'_XX
capture drop path_`i'_XX
capture drop num_g_paths_`i'_XX
capture drop cohort_fullpath_`i'_XX

capture drop dof_cohort_`i'_s_t_XX
capture drop dof_cohort_`i'_ns_t_XX
capture drop dof_cohort_`i'_s0_t_XX
capture drop dof_cohort_`i'_s1_t_XX
capture drop dof_cohort_`i'_s2_t_XX
capture drop count_cohort_`i'_s_t_XX
capture drop count_cohort_`i'_ns_t_XX
capture drop count_cohort_`i'_s0_t_XX
capture drop count_cohort_`i'_s1_t_XX
capture drop count_cohort_`i'_s2_t_XX
capture drop total_cohort_`i'_s_t_XX
capture drop total_cohort_`i'_ns_t_XX
capture drop total_cohort_`i'_s0_t_XX
capture drop total_cohort_`i'_s1_t_XX
capture drop total_cohort_`i'_s2_t_XX
capture drop mean_cohort_`i'_s_t_XX
capture drop mean_cohort_`i'_ns_t_XX
capture drop mean_cohort_`i'_s0_t_XX
capture drop mean_cohort_`i'_s1_t_XX
capture drop mean_cohort_`i'_s2_t_XX

capture drop E_hat_gt_`i'_XX
capture drop DOF_gt_`i'_XX

///// Creating long difference of outcome

xtset group_XX time_XX
bys group_XX : gen diff_y_`i'_XX = outcome_XX - L`i'.outcome_XX

///// Creating treatment paths if less_conservative_se option specified

if "`less_conservative_se'" != ""{

// Creating a time-invariant, group-level variable, containing g's treatment at F_g-1+\ell
gen d_fg_XX_temp=treatment_XX if time_XX==F_g_XX + `=-1+`i''
bys group_XX: gegen d_fg`i'_XX=mean(d_fg_XX_temp)

// This variable might be missing, for groups whose treatment never changes, and
// for groups not observed \ell periods after treatment change. Then, we impute
// their treatment at F_g-1+\ell-1. Inconsequential, just to avoid missing values.
// We also need to initialize a variable d_fg0_XX, when \ell=1.
if `i'==1{
gen d_fg0_XX=d_sq_XX
gegen path_0_XX=group(d_fg0_XX F_g_XX)
}
replace d_fg`i'_XX=d_fg`=`i'-1'_XX if d_fg`i'_XX==.
gegen path_`i'_XX = group(path_`=`i'-1'_XX d_fg`i'_XX)
drop d_fg_XX_temp 

// For each group, define a variable counting how many groups belong to the same cohort,
// with cohorts defined as d_fg0_XX F_g_XX, as well as the full path.
if `i'==1{
	bysort path_0_XX: gegen num_g_paths_0_XX=nunique(group_XX)
}
bysort path_`i'_XX: gegen num_g_paths_`i'_XX=nunique(group_XX)

// For each group, generate a dummy for whether at least two groups in their cohort.
gen cohort_fullpath_`i'_XX=(num_g_paths_`i'_XX>1)
if `i'==1{
gen cohort_fullpath_0_XX=(num_g_paths_0_XX>1) 
}
}

///// Identifying the control (g,t)s in the estimation of dynamic effect i 
bys group_XX: gen never_change_d_`i'_XX=(F_g_XX>time_XX) if diff_y_`i'_XX!=.

if "`only_never_switchers'" != "" {
	replace never_change_d_`i'_XX = 0 if F_g_XX > time_XX & F_g_XX < T_max_XX + 1 & diff_y_`i'_XX != .
}

///// Creating N^g_t:
///// number of control groups for g at t
gen never_change_d_`i'_wXX = never_change_d_`i'_XX*N_gt_XX
bys time_XX d_sq_XX `trends_nonparam': gegen N_gt_control_`i'_XX=total(never_change_d_`i'_wXX)

///// Creating binary variable indicating whether g is \ell periods away from switch

// If the same_switchers option is specified:

if ("`same_switchers'"!=""){

capture drop relevant_y_missing_XX
capture drop cum_fillin_XX
capture drop fillin_g_XX 
capture drop fillin_g_pl_XX
capture drop dum_fillin_temp_XX 
capture drop still_switcher_`i'_XX  

sort group_XX time_XX

// Modif. Diego 31-03-2025: long-difference and other variables to tailor distance_to_switch to the groups for which the all the effects can be computed
cap drop N_g_control_check_XX
gen N_g_control_check_XX = 0

forv j = 1/`=`effects'' {
	cap drop diff_y_last_XX
	cap drop never_change_d_last_XX
	cap drop never_change_d_last_wXX
	cap drop N_gt_control_last_XX
	cap drop N_g_control_last_temp_XX
	cap drop N_g_control_last_m_XX
	cap drop diff_y_relev_temp_XX
	cap drop diff_y_relev_XX

	xtset group_XX time_XX
	bys group_XX : gen diff_y_last_XX = outcome_XX - L`j'.outcome_XX
	bys group_XX: gen never_change_d_last_XX=(F_g_XX>time_XX) if diff_y_last_XX!=.
	if "`only_never_switchers'" != "" {
		replace never_change_d_last_XX = 0 if F_g_XX > time_XX & F_g_XX < T_max_XX + 1 & diff_y_last_XX != .
	}
	gen never_change_d_last_wXX = never_change_d_last_XX*N_gt_XX
	bys time_XX d_sq_XX `trends_nonparam': gegen N_gt_control_last_XX=total(never_change_d_last_wXX)

	gen N_g_control_last_temp_XX = N_gt_control_last_XX if time_XX == F_g_XX - 1 + `j'
	bys group_XX: egen N_g_control_last_m_XX = mean(N_g_control_last_temp_XX)

	gen diff_y_relev_temp_XX = diff_y_last_XX if time_XX == F_g_XX - 1 + `j'
	bys group_XX: egen diff_y_relev_XX = mean(diff_y_relev_temp_XX)

	replace N_g_control_check_XX = N_g_control_check_XX + (N_g_control_last_m_XX > 0 & diff_y_relev_XX != .)
}

* If the same_switchers_pl option is specified:

if ("`same_switchers_pl'"!=""){
cap drop N_g_control_check_pl_XX
gen N_g_control_check_pl_XX = 0
forv j = 1/`=`placebo'' {
	cap drop diff_y_last_XX
	cap drop never_change_d_last_XX
	cap drop never_change_d_last_wXX
	cap drop N_gt_control_last_XX
	cap drop N_g_control_last_temp_XX
	cap drop N_g_control_last_m_XX
	cap drop diff_y_relev_temp_XX
	cap drop diff_y_relev_XX

	xtset group_XX time_XX
	bys group_XX : gen diff_y_last_XX = outcome_XX - F`j'.outcome_XX
	bys group_XX: gen never_change_d_last_XX=(F_g_XX>time_XX) if diff_y_last_XX!=.
	if "`only_never_switchers'" != "" {
		replace never_change_d_last_XX = 0 if F_g_XX > time_XX & F_g_XX < T_max_XX + 1 & diff_y_last_XX != .
	}
	gen never_change_d_last_wXX = never_change_d_last_XX*N_gt_XX
	bys time_XX d_sq_XX `trends_nonparam': gegen N_gt_control_last_XX=total(never_change_d_last_wXX)

	gen N_g_control_last_temp_XX = N_gt_control_last_XX if time_XX == F_g_XX - 1 - `j'
	bys group_XX: egen N_g_control_last_m_XX = mean(N_g_control_last_temp_XX)

	gen diff_y_relev_temp_XX = diff_y_last_XX if time_XX == F_g_XX - 1 - `j'
	bys group_XX: egen diff_y_relev_XX = mean(diff_y_relev_temp_XX)

	replace N_g_control_check_pl_XX = N_g_control_check_pl_XX + (N_g_control_last_m_XX > 0 & diff_y_relev_XX != .)
}
* Generate a variable tagging the switchers that should be dropped
* Is the case if at least one of the placebos or effects we try to estimate is missing:
gen relevant_y_missing_XX=(outcome_XX==.&time_XX>=F_g_XX-1-`=`placebo''&time_XX<=F_g_XX-1+`=`effects'') 
* Or if some of the controls are missing:
if "`controls'" != ""{
replace relevant_y_missing_XX=1 if fd_X_all_non_missing_XX==0&time_XX>=F_g_XX-1-`=`placebo''&time_XX<=F_g_XX-1+`=`effects''
}

// Modif. Diego 19-04-25: make the same adjustments as below in case same_switcher_pl is specified
//bys group_XX: gen cum_fillin_XX = sum(relevant_y_missing_XX)
//gen dum_fillin_temp_XX = (cum_fillin_XX==0&time_XX==F_g_XX-1+`=`effects'')
//bys group_XX: gegen fillin_g_XX = total(dum_fillin_temp_XX)

//gen dum_fillin_temp_pl_XX = (cum_fillin_XX==0&time_XX==F_g_XX-1-`=`placebo'')
//bys group_XX: gegen fillin_g_pl_XX = total(dum_fillin_temp_pl_XX)
gen fillin_g_pl_XX = (N_g_control_check_pl_XX == `placebo')

capture drop dum_fillin_temp_XX
capture drop dum_fillin_temp_pl_XX

* tag switchers who have no missings from F_g_XX-1-`=`placebo'' to F_g_XX-1+`=`effects''
gen still_switcher_`i'_XX = (F_g_XX-1+`=`effects''<=T_g_XX & N_g_control_check_XX == `effects')  	

gen distance_to_switch_`i'_XX=(still_switcher_`i'_XX&time_XX==F_g_XX-1+`i'&`i'<=L_g_XX&S_g_XX==increase_XX&N_gt_control_`i'_XX>0&N_gt_control_`i'_XX!=.) if diff_y_`i'_XX!=. 
}

* If the same_switchers_pl option is not specified:
	
if ("`same_switchers_pl'"==""){
* Generate a variable tagging the switchers that should be dropped
* Is the case if at least one of the effects we try to estimate is missing:
gen relevant_y_missing_XX=(outcome_XX==.&time_XX>=F_g_XX-1&time_XX<=F_g_XX-1+`=`effects'') 
* Or if some of the controls are missing:
if "`controls'" != ""{
replace relevant_y_missing_XX=1 if fd_X_all_non_missing_XX==0&time_XX>=F_g_XX&time_XX<=F_g_XX-1+`=`effects''
}

// Modif. Diego 31-03-2025: we do not need this block if we use the changes above
//bys group_XX: gen cum_fillin_XX = sum(relevant_y_missing_XX)
//gen dum_fillin_temp_XX = (cum_fillin_XX==0&time_XX==F_g_XX-1+`=`effects'')
//bys group_XX: gegen fillin_g_XX = total(dum_fillin_temp_XX)

* tag switchers who have no missings from F_g_XX-1 to F_g_XX-1+`=`effects''
// Modif. Diego 31-03-2025: add check for control groups with the same baseline treat to still_switchers for all estimated periods

gen still_switcher_`i'_XX = (F_g_XX-1+`=`effects''<=T_g_XX & N_g_control_check_XX == `effects') 	
gen distance_to_switch_`i'_XX=(still_switcher_`i'_XX&time_XX==F_g_XX-1+`i'&`i'<=L_g_XX&S_g_XX==increase_XX&N_gt_control_`i'_XX>0&N_gt_control_`i'_XX!=.) if diff_y_`i'_XX!=.  

}
}

// If the same_switchers option is not specified

else{
gen distance_to_switch_`i'_XX=(time_XX==F_g_XX-1+`i'&`i'<=L_g_XX&S_g_XX==increase_XX&N_gt_control_`i'_XX>0&N_gt_control_`i'_XX!=.) if diff_y_`i'_XX!=. 
}

///// Creating a variable counting the number of groups \ell periods away from switch at t

gen distance_to_switch_`i'_wXX = distance_to_switch_`i'_XX*N_gt_XX
bys time_XX: gegen N`=increase_XX'_t_`i'_XX=total(distance_to_switch_`i'_wXX)

///// Computing N1_l,N0_l.

// Initializing the N1_`i'_XX/N0_`i'_XX scalar at 0. 
scalar N`=increase_XX'_`i'_XX =0

// Loop over t incrementing the scalar
forvalue t=`=t_min_XX'/`=T_max_XX'{
	sum N`=increase_XX'_t_`i'_XX if time_XX==`t'
	if !missing(r(mean)) scalar N`=increase_XX'_`i'_XX = N`=increase_XX'_`i'_XX + r(mean)
}

// Modif Felix weight
if "`weight'"!=""{

	bys time_XX: gegen N`=increase_XX'_t_`i'_XX_w=total(distance_to_switch_`i'_XX)
	
	scalar N`=increase_XX'_`i'_XX_weight =0

// Loop over t incrementing the scalar
forvalue t=`=t_min_XX'/`=T_max_XX'{
	sum N`=increase_XX'_t_`i'_XX_w if time_XX==`t'
	if !missing(r(mean)) scalar N`=increase_XX'_`i'_XX_weight = N`=increase_XX'_`i'_XX_weight + r(mean)
}
}

///// Creating N^1_{t,\ell,g}/N^0_{t,\ell,g}: 
///// Variable counting number of groups \ell periods away from switch at t, 
///// and with same D_{g,1} and trends_nonparam.

bys time_XX d_sq_XX `trends_nonparam': gegen N`=increase_XX'_t_`i'_g_XX=total(distance_to_switch_`i'_wXX)

///// Creating all the adjustment terms to compute estimators with controls, and their variances
 
if "`controls'" != ""{
	
// Initialize intermediate Variable needed later	
capture drop part2_switch`=increase_XX'_`i'_XX
gen part2_switch`=increase_XX'_`i'_XX=0

// generation of the T_d variable = max_{g:D_g,1 = d} F_g - 1: 
// last period when treatment effects can still be estimated for groups with baseline treatment equal to d

capture drop T_d_XX
gegen T_d_XX = max(F_g_XX), by(d_sq_int_XX)
replace T_d_XX = T_d_XX - 1

// Computing the long differences of the control variables (X_g_t - X_g_t-l)

local count_controls=0

foreach var of varlist `controls'{
		
local count_controls=`count_controls'+1

capture drop diff_X`count_controls'_`i'_XX

xtset group_XX time_XX
gen diff_X`count_controls'_`i'_XX=`var' - L`i'.`var'

// Computing N_g_t * (X_g_t - X_g_t-l)

capture drop diff_X`count_controls'_`i'_N_XX
gen diff_X`count_controls'_`i'_N_XX = N_gt_XX * diff_X`count_controls'_`i'_XX 

foreach l of local levels_d_sq_XX { // index l corresponds to d in the paper 

// intermediate variable to count the number of groups within each not yet switched cohort

// Changes Diego 27-03-25
//capture drop dummy_XX
//gen dummy_XX=(F_g_XX>time_XX & d_sq_int_XX == `l')

// Computing coordinates of vectors m_+_(g,d,l) and m_-_(g,d,l)

*Creating variable inside the summation across t in m^+_{g,d,\ell}/m^-_{g,d,\ell}
capture drop m`=increase_XX'_g_`count_controls'_`l'_`i'_XX 
gen m`=increase_XX'_g_`count_controls'_`l'_`i'_XX = (`i' <= T_g_XX-2 & d_sq_int_XX == `l')* (G_XX / N`=increase_XX'_`i'_XX) * ([distance_to_switch_`i'_XX - (N`=increase_XX'_t_`i'_g_XX/N_gt_control_`i'_XX) * never_change_d_`i'_XX] * (time_XX>=`=`i'+1'&time_XX<=T_g_XX) * diff_X`count_controls'_`i'_N_XX)
*Summing that variable across t, and leaving one non missing observation per g	
capture drop m`=increase_XX'_`l'_`count_controls'_`i'_XX
bys group_XX: gegen m`=increase_XX'_`l'_`count_controls'_`i'_XX=total(m`=increase_XX'_g_`count_controls'_`l'_`i'_XX)
bys group_XX: replace m`=increase_XX'_`l'_`count_controls'_`i'_XX = . if _n != 1

// Computing coordinates of vectors M^+_{d,\ell} and M^-_{d,\ell}

capture drop M`=increase_XX'_`l'_`count_controls'_`i'_XX
egen M`=increase_XX'_`l'_`count_controls'_`i'_XX = total(m`=increase_XX'_`l'_`count_controls'_`i'_XX)
replace M`=increase_XX'_`l'_`count_controls'_`i'_XX = (1/G_XX)*M`=increase_XX'_`l'_`count_controls'_`i'_XX

// number of groups within each not yet switched cohort
capture drop E_hat_denom_`count_controls'_`l'_XX
//// CHANGE BELOW - tks + Changes Diego 27-03-25: replace dummy_XX_2 with dummy_XX

// Modif Clément 20/6/2025:
* Counting number of groups for DOF adjustment
{
cap drop dummy_XX
gen dummy_XX = 0
replace dummy_XX = (F_g_XX>time_XX & d_sq_int_XX == `l') if diff_y_XX < .
	if "`cluster'" == "" {
		bys time_XX : egen E_hat_denom_`count_controls'_`l'_XX = total(dummy_XX) if d_sq_int_XX == `l'
	}
	else {
		cap drop cluster_temp_XX
		gen cluster_temp_XX= `cluster' if dummy_XX == 1
		bys time_XX : gegen E_hat_denom_`count_controls'_`l'_XX = nunique(cluster_temp_XX) if !missing(cluster_temp_XX)
	}
}
// End Modif Clément 20/6/2025

// Add the indicator for at least two groups in the cohort to E_y_hat_gt_`l'_XX (demeaning is possible)
capture drop E_y_hat_gt_`l'_XX
gen E_y_hat_gt_`l'_XX=E_y_hat_gt_int_`l'_XX*(E_hat_denom_`count_controls'_`l'_XX>=2)


// Computing the summation from t=2 to F_g-1 that appears in the last term 
// of U^{+,var,X}_{g,\ell} and U^{-,var,X}_{g,\ell}, defined in the companion paper.
 
capture drop in_sum_temp_`count_controls'_`l'_XX
capture drop N_c_`l'_temp_XX
capture drop N_c_`l'_XX
//// CHANGE BELOW - tks, Changes Diego 27-03-25: Multiply by N_gt_XX
gen N_c_`l'_temp_XX = N_gt_XX * (d_sq_int_XX == `l' & time_XX >= 2 & time_XX <= T_d_XX & time_XX < F_g_XX & diff_y_XX < .)
////
egen N_c_`l'_XX = total(N_c_`l'_temp_XX)
/// Changes Diego 16-06-25: Adjust demeaning when E_hat_denom == 1
cap drop in_sum_temp_adj_`count_controls'_`l'_XX
gen in_sum_temp_adj_`count_controls'_`l'_XX = 0 if E_y_hat_gt_`l'_XX != .
replace in_sum_temp_adj_`count_controls'_`l'_XX = (sqrt((E_hat_denom_`count_controls'_`l'_XX)/(E_hat_denom_`count_controls'_`l'_XX - 1))-1) if E_y_hat_gt_`l'_XX != . & E_hat_denom_`count_controls'_`l'_XX > 1
gen in_sum_temp_`count_controls'_`l'_XX = (prod_X`count_controls'_Ngt_XX*(1+(E_hat_denom_`count_controls'_`l'_XX>=2)*in_sum_temp_adj_`count_controls'_`l'_XX)*(diff_y_XX-E_y_hat_gt_`l'_XX)*(time_XX>=2 & time_XX<=F_g_XX-1)) / N_c_`l'_XX


capture drop in_sum_`count_controls'_`l'_XX
capture drop in_sum_temp_adj_`count_controls'_`l'_XX
bys group_XX: gegen in_sum_`count_controls'_`l'_XX = total(in_sum_temp_`count_controls'_`l'_XX) 

}


// Residualize the outcome difference wrt control differences:
// Yg,t − Yg,t−ℓ − (Xg,t − Xg,t−ℓ)*θ_{Dg,1}

foreach l of local levels_d_sq_XX { 
	if (scalar(useful_res_`l'_XX)>1){
replace diff_y_`i'_XX = diff_y_`i'_XX - coefs_sq_`l'_XX[`=`count_controls'',1]*diff_X`count_controls'_`i'_XX if d_sq_int_XX==`l' 
* N.B. : in the above line, we do not add "&diff_X`count_controls'_`i'_XX!=." because we want to exclude from the estimation any first/long-difference for which the covariates are missing.

// Initialize intermediate Variable needed later
capture drop in_brackets_`l'_`count_controls'_XX	
gen in_brackets_`l'_`count_controls'_XX=0

}
}
}
}

///// Computing the variables used for the demeaning of outcome's long difference diff_y_`i',
///// which we will use to create the U_g^{var} variables, which are used to compute 
///// the estimators' variances. 

// Generate long-differences of outcomes time N_{g,t}, and dummy for (g,t) such that
// diff_y_`i' non missing and N_gt non missing.

capture drop diff_y_`i'_N_gt_XX
gen diff_y_`i'_N_gt_XX=N_gt_XX*diff_y_`i'_XX

capture drop dof_ns_`i'_XX
gen dof_ns_`i'_XX = (N_gt_XX != 0&diff_y_`i'_XX!=.&never_change_d_`i'_XX==1&N`=increase_XX'_t_`i'_XX>0&N`=increase_XX'_t_`i'_XX!=.)
capture drop dof_s_`i'_XX
gen dof_s_`i'_XX = (N_gt_XX != 0&distance_to_switch_`i'_XX==1)

// For never switchers, demeaning wrt to cohorts defined by D_{g,1}, `trends_nonparam' 
//(\mathcal{D}_k in companion paper)

// Modif Clément: we need to add by time
* Mean's denominator
bys d_sq_XX `trends_nonparam' time_XX : gegen count_cohort_`i'_ns_t_XX=total(N_gt_XX) if dof_ns_`i'_XX == 1

* Mean's numerator
bys d_sq_XX `trends_nonparam' time_XX: gegen total_cohort_`i'_ns_t_XX=total(diff_y_`i'_N_gt_XX) if dof_ns_`i'_XX == 1

* Mean 
gen mean_cohort_`i'_ns_t_XX=total_cohort_`i'_ns_t_XX/count_cohort_`i'_ns_t_XX

* Counting number of groups for DOF adjustment
{
	if "`cluster'" == "" {
		bys d_sq_XX `trends_nonparam' time_XX: gegen dof_cohort_`i'_ns_t_XX=total(dof_ns_`i'_XX) if dof_ns_`i'_XX == 1
	}
	else {
		cap drop cluster_dof_`i'_ns_XX
		gen cluster_dof_`i'_ns_XX = `cluster' if dof_ns_`i'_XX == 1
		bys d_sq_XX `trends_nonparam' time_XX: gegen dof_cohort_`i'_ns_t_XX = nunique(cluster_dof_`i'_ns_XX) if !missing(cluster_dof_`i'_ns_XX)
	}
}
// End modif Clément

// For switchers, if option less_conservative_se not specified, demeaning wrt to 
// cohorts defined by D_{g,1}, F_g, D_{g,F_g}, `trends_nonparam' (\mathcal{C}_k in companion paper).

if "`less_conservative_se'" == ""{
	
* Mean's denominator
bys d_sq_XX F_g_XX d_fg_XX `trends_nonparam' : gegen count_cohort_`i'_s_t_XX=total(N_gt_XX) if dof_s_`i'_XX==1

* Mean's numerator
bys d_sq_XX F_g_XX d_fg_XX `trends_nonparam' : gegen total_cohort_`i'_s_t_XX=total(diff_y_`i'_N_gt_XX) if dof_s_`i'_XX==1 

* Mean 
gen mean_cohort_`i'_s_t_XX=total_cohort_`i'_s_t_XX/count_cohort_`i'_s_t_XX	
		
* Counting number of groups for DOF adjustment
// Changes Diego 16-06-25: adjust DoF with clusters
{
	if "`cluster'" == "" {
		bys d_sq_XX F_g_XX d_fg_XX `trends_nonparam' : gegen dof_cohort_`i'_s_t_XX=total(dof_s_`i'_XX) if dof_s_`i'_XX==1
	}
	else {
		cap drop cluster_dof_`i'_s_XX
		gen cluster_dof_`i'_s_XX = `cluster' if dof_s_`i'_XX == 1
		bys d_sq_XX F_g_XX d_fg_XX `trends_nonparam': gegen dof_cohort_`i'_s_t_XX = nunique(cluster_dof_`i'_s_XX) if !missing(cluster_dof_`i'_s_XX)
	}
}
}
//save test_data, replace


// For switchers, if option less_conservative_se specified, demeaning wrt to cohorts 
// defined by D_{g,1} F_g, D_{g,F_g},..., D_{g,F_g+\ell}, if that cohort has at least two groups,
// if not: demeaning wrt to cohorts 
// defined by D_{g,1} F_g, D_{g,F_g}, if that cohort has at least two groups,
// if not: demeaning wrt D_{g,1} F_g.

if "`less_conservative_se'" != ""{
		
* Mean's denominator	
* by D_{g,1}, F_g, `trends_nonparam':
bys path_0_XX `trends_nonparam' : gegen count_cohort_`i'_s0_t_XX=total(N_gt_XX) if dof_s_`i'_XX==1
* by D_{g,1}, F_g, D_{g,F_g}, `trends_nonparam':
bys path_1_XX `trends_nonparam' : gegen count_cohort_`i'_s1_t_XX=total(N_gt_XX) if dof_s_`i'_XX==1
* by D_{g,1}, F_g, D_{g,F_g},..., D_{g,F_g+\ell}, `trends_nonparam':
bys path_`i'_XX `trends_nonparam' : gegen count_cohort_`i'_s2_t_XX=total(N_gt_XX) if dof_s_`i'_XX==1
	
* Mean's numerator
* by D_{g,1}, F_g, `trends_nonparam':
bys path_0_XX `trends_nonparam' : gegen total_cohort_`i'_s0_t_XX=total(diff_y_`i'_N_gt_XX) if dof_s_`i'_XX==1
* by D_{g,1}, F_g, D_{g,F_g}, `trends_nonparam':
bys path_1_XX `trends_nonparam' : gegen total_cohort_`i'_s1_t_XX=total(diff_y_`i'_N_gt_XX) if dof_s_`i'_XX==1	
* by D_{g,1}, F_g, D_{g,F_g},..., D_{g,F_g+\ell}, `trends_nonparam':
bys path_`i'_XX `trends_nonparam' : gegen total_cohort_`i'_s2_t_XX=total(diff_y_`i'_N_gt_XX) if dof_s_`i'_XX==1

* Counting number of groups for DOF adjustment
// Changes Diego 16-06-25: adjust DoF with clusters
{
	if "`cluster'" == "" {
		* by D_{g,1}, F_g, `trends_nonparam':
		bys path_0_XX `trends_nonparam' : gegen dof_cohort_`i'_s0_t_XX=total(dof_s_`i'_XX) if dof_s_`i'_XX==1
		* by D_{g,1}, F_g, D_{g,F_g}, `trends_nonparam':
		bys path_1_XX `trends_nonparam' : gegen dof_cohort_`i'_s1_t_XX=total(dof_s_`i'_XX) if dof_s_`i'_XX==1
		* by D_{g,1}, F_g, D_{g,F_g},..., D_{g,F_g+\ell}, `trends_nonparam':
		bys path_`i'_XX `trends_nonparam' : gegen dof_cohort_`i'_s2_t_XX=total(dof_s_`i'_XX) if dof_s_`i'_XX==1
	}
	else {
		cap drop cluster_dof_`i'_s_XX
		gen cluster_dof_`i'_s_XX = `cluster' if dof_s_`i'_XX == 1
		* by D_{g,1}, F_g, `trends_nonparam':
		bys path_0_XX `trends_nonparam' : gegen dof_cohort_`i'_s0_t_XX = nunique(cluster_dof_`i'_s_XX) if !missing(cluster_dof_`i'_s_XX)
		* by D_{g,1}, F_g, D_{g,F_g}, `trends_nonparam':
		bys path_1_XX `trends_nonparam' : gegen dof_cohort_`i'_s1_t_XX = nunique(cluster_dof_`i'_s_XX) if !missing(cluster_dof_`i'_s_XX)
		* by D_{g,1}, F_g, D_{g,F_g},..., D_{g,F_g+\ell}, `trends_nonparam':
		bys path_`i'_XX `trends_nonparam' : gegen dof_cohort_`i'_s2_t_XX = nunique(cluster_dof_`i'_s_XX) if !missing(cluster_dof_`i'_s_XX)

	}
}

*Attributing the right number of groups depending on which cohort will be used in demeaning:
gen dof_cohort_`i'_s_t_XX=dof_cohort_`i'_s2_t_XX if dof_cohort_`i'_s2_t_XX>=2
replace dof_cohort_`i'_s_t_XX=dof_cohort_`i'_s1_t_XX if dof_cohort_`i'_s2_t_XX<2&dof_cohort_`i'_s1_t_XX>=2 
replace dof_cohort_`i'_s_t_XX=dof_cohort_`i'_s0_t_XX if dof_cohort_`i'_s2_t_XX<2&dof_cohort_`i'_s1_t_XX<2 

* Mean
gen mean_cohort_`i'_s_t_XX=total_cohort_`i'_s2_t_XX/count_cohort_`i'_s2_t_XX if dof_cohort_`i'_s2_t_XX>=2
replace mean_cohort_`i'_s_t_XX=total_cohort_`i'_s1_t_XX/count_cohort_`i'_s1_t_XX if dof_cohort_`i'_s2_t_XX<2&dof_cohort_`i'_s1_t_XX>=2 
replace mean_cohort_`i'_s_t_XX=total_cohort_`i'_s0_t_XX/count_cohort_`i'_s0_t_XX if dof_cohort_`i'_s2_t_XX<2&dof_cohort_`i'_s1_t_XX<2

}

// Modif Clément 30/6/2025:
// If a switcher is the only one in their cohort or if a not-yet-switcher is the only one in their cohort, we demean wrt union of switchers and not-yet switchers, provided switchers and not-yet-switchers do not all come from the same cluster.

cap drop dof_ns_s_`i'_XX
cap drop count_cohort_`i'_ns_s_t_XX
cap drop total_cohort_`i'_ns_s_t_XX
cap drop mean_cohort_`i'_ns_s_t_XX
cap drop dof_cohort_`i'_ns_s_t_XX

gen dof_ns_s_`i'_XX=(dof_s_`i'_XX==1|dof_ns_`i'_XX ==1)

* Mean's denominator
bys d_sq_XX `trends_nonparam' time_XX : gegen count_cohort_`i'_ns_s_t_XX=total(N_gt_XX) if dof_ns_s_`i'_XX==1

* Mean's numerator
bys d_sq_XX `trends_nonparam' time_XX: gegen total_cohort_`i'_ns_s_t_XX=total(diff_y_`i'_N_gt_XX) if dof_ns_s_`i'_XX==1

* Mean 
gen mean_cohort_`i'_ns_s_t_XX=total_cohort_`i'_ns_s_t_XX/count_cohort_`i'_ns_s_t_XX

* Counting number of groups for DOF adjustment
{
	if "`cluster'" == "" {
		bys d_sq_XX `trends_nonparam' time_XX: gegen dof_cohort_`i'_ns_s_t_XX=total(dof_ns_s_`i'_XX) if dof_ns_s_`i'_XX==1
	}
	else {
		cap drop cluster_dof_`i'_ns_s_XX
		gen cluster_dof_`i'_ns_s_XX = `cluster' if dof_ns_s_`i'_XX==1
		bys d_sq_XX `trends_nonparam' time_XX: gegen dof_cohort_`i'_ns_s_t_XX = nunique(cluster_dof_`i'_ns_s_XX) if !missing(cluster_dof_`i'_ns_s_XX)
	}
}	

///// From those parts, generate variables for the demeaning and the DOF adjustment 
// E_hat_(g,t), defined from parts depending on the cohort definition
gen E_hat_gt_`i'_XX=0 if (time_XX<F_g_XX|F_g_XX-1+`i'==time_XX) 
replace E_hat_gt_`i'_XX=mean_cohort_`i'_ns_t_XX if (time_XX<F_g_XX&dof_cohort_`i'_ns_t_XX>=2)
replace E_hat_gt_`i'_XX=mean_cohort_`i'_s_t_XX if (F_g_XX-1+`i'==time_XX&dof_cohort_`i'_s_t_XX>=2)
replace E_hat_gt_`i'_XX=mean_cohort_`i'_ns_s_t_XX if dof_cohort_`i'_ns_s_t_XX>=2&((F_g_XX-1+`i'==time_XX&dof_cohort_`i'_s_t_XX==1)|(time_XX<F_g_XX&dof_cohort_`i'_ns_t_XX==1))

// DOF_(g,t) for DOF adjustement, defined from parts depending on the cohort definition 
// Diego - 02-03-25: when there is only 1 switcher, dof_cohort_`i'_s_t_XX = 1, hence the denominator in the expression below is 0
// The fraction is undefined and Stata puts it to missing
gen DOF_gt_`i'_XX=1 if (time_XX<F_g_XX|F_g_XX-1+`i'==time_XX)
replace DOF_gt_`i'_XX= sqrt(dof_cohort_`i'_s_t_XX/(dof_cohort_`i'_s_t_XX-1)) if (F_g_XX-1+`i'==time_XX & dof_cohort_`i'_s_t_XX > 1)
replace DOF_gt_`i'_XX=  sqrt(dof_cohort_`i'_ns_t_XX/(dof_cohort_`i'_ns_t_XX-1)) if (time_XX<F_g_XX & dof_cohort_`i'_ns_t_XX > 1)
replace DOF_gt_`i'_XX=  sqrt(dof_cohort_`i'_ns_s_t_XX/(dof_cohort_`i'_ns_s_t_XX-1)) if dof_cohort_`i'_ns_s_t_XX>=2&((F_g_XX-1+`i'==time_XX&dof_cohort_`i'_s_t_XX==1)|(time_XX<F_g_XX&dof_cohort_`i'_ns_t_XX==1))

// End modif Clément 30/6/2025.

////////// 3. Computing U_Gg_\ell variables

///// If the dynamic effect can be estimated (as there are switchers), we compute the U_Gg_\ell variables etc.

if (N`=increase_XX'_`i'_XX!=0){

// Creating a dummy variable indicating whether l<=T_g_XX-1

gen dummy_U_Gg`i'_XX = (`i'<=T_g_XX-1) 

// Computing U_+_(G,g,l)

gen U_Gg`i'_temp_XX = dummy_U_Gg`i'_XX*(G_XX / N`=increase_XX'_`i'_XX) * (time_XX>=`=`i'+1'&time_XX<=T_g_XX) * N_gt_XX* [distance_to_switch_`i'_XX - (N`=increase_XX'_t_`i'_g_XX/N_gt_control_`i'_XX) * never_change_d_`i'_XX] 

replace U_Gg`i'_temp_XX = U_Gg`i'_temp_XX* diff_y_`i'_XX 

bysort group_XX : gegen U_Gg`i'_XX=total(U_Gg`i'_temp_XX)

replace U_Gg`i'_XX = U_Gg`i'_XX*first_obs_by_gp_XX

// Counting the number of cells for which we can estimate U_Gg`i'_temp_XX, 
// to compute the "N" displayed by command

gen count`i'_core_XX=0
replace count`i'_core_XX=N_gt_XX if (U_Gg`i'_temp_XX!=.&U_Gg`i'_temp_XX!=0|(U_Gg`i'_temp_XX==0&diff_y_`i'_XX==0&(distance_to_switch_`i'_XX!=0|(N`=increase_XX'_t_`i'_g_XX!=0&never_change_d_`i'_XX!=0))))
 
// Computing U_(+,var)_(G,g,l)

gen U_Gg`i'_temp_var_XX = 0

// Final computation 

replace U_Gg`i'_temp_var_XX= dummy_U_Gg`i'_XX*(G_XX / N`=increase_XX'_`i'_XX) * [distance_to_switch_`i'_XX - (N`=increase_XX'_t_`i'_g_XX/N_gt_control_`i'_XX) * never_change_d_`i'_XX] * (time_XX>=`=`i'+1'&time_XX<=T_g_XX) * N_gt_XX * DOF_gt_`i'_XX *(diff_y_`i'_XX-E_hat_gt_`i'_XX)

///// Adding the additional part of U^(+,var,X)_{G,g,l}/U^(-,var,X)_{G,g,l} when controls are included:
///// sum across values of baseline treatment d of M^+_(d,l)* a term in brackets in companion paper. 

if "`controls'"!=""{

// Loop over values of d_sq_int_XX:sum across values of baseline treatment  
levelsof d_sq_int_XX, local(levels_d_sq_XX)

foreach l of local levels_d_sq_XX {	
	// FELIX: Add this condition here, same for the placebo case!	
	if (scalar(useful_res_`l'_XX)>1){
	
capture drop combined`=increase_XX'_temp_`l'_`i'_XX	
gen combined`=increase_XX'_temp_`l'_`i'_XX=0
		
// Loop over controls: inner product of M^+_(d,l)* term in brackets in companion paper.
forvalues j=1/`count_controls'{

// Loop over k: computation of term in brackets in companion paper.
forvalues k=1/`count_controls'{		
	
// Computation of all cross products between elements of jth line of Den^{-1}_d and term multiplying it
capture drop in_brackets_`l'_`j'_`k'_temp_XX
//// CHANGE BELOW - tks + Changes Diego 27-03-25: remove N_c_`l'_XX from formula below, since we have adjusted inv_Denom above
gen in_brackets_`l'_`j'_`k'_temp_XX = inv_Denom_`l'_XX[`j',`k'] * in_sum_`k'_`l'_XX * (d_sq_int_XX == `l' & F_g_XX>=3) 
////

// Summing over k, to have jth coordinate of vector Den^{-1}_d*...

replace in_brackets_`l'_`j'_XX=in_brackets_`l'_`j'_XX+in_brackets_`l'_`j'_`k'_temp_XX
} // end loop over k

// Withdrawing theta_d
replace in_brackets_`l'_`j'_XX=in_brackets_`l'_`j'_XX - coefs_sq_`l'_XX[`j',1]

// Computation of all cross products between coordinates of M^+_(d,l) and those of term in brackets
capture drop combined`=increase_XX'_temp_`l'_`j'_`i'_XX
gen combined`=increase_XX'_temp_`l'_`j'_`i'_XX=M`=increase_XX'_`l'_`j'_`i'_XX*in_brackets_`l'_`j'_XX

// Summing over j
replace combined`=increase_XX'_temp_`l'_`i'_XX=combined`=increase_XX'_temp_`l'_`i'_XX+combined`=increase_XX'_temp_`l'_`j'_`i'_XX
} // end loop over j

// Final sum over the status quo treatment (outer sum over d in the formula)
// Modif. Diego: dropped  if d_sq_int_XX==`l'
replace part2_switch`=increase_XX'_`i'_XX=part2_switch`=increase_XX'_`i'_XX+combined`=increase_XX'_temp_`l'_`i'_XX
} // Modif FELIX: condition "useful residual" 
} // end loop over l

}	

// Summing the U^(var)_{G,g,l}s over time periods for each group
bys group_XX: gegen U_Gg`i'_var_XX=total(U_Gg`i'_temp_var_XX)

// Modeif Felix 21.03.2025 (adjust order of adding additional term and summing across t)
if "`controls'" != "" {
	// Making the adjustement to U^(+,var)_{G,g,l} when controls are included
	if `=increase_XX'==1{
	replace U_Gg`i'_var_XX=U_Gg`i'_var_XX - part2_switch1_`i'_XX 
	}

	if `=increase_XX'==0{
	//// CHANGE BELOW - tks (switching plus to minus in the sum since the final sign flip won't be until later)
	replace U_Gg`i'_var_XX=U_Gg`i'_var_XX - part2_switch0_`i'_XX 
	}
}

}

////////// 4. Computing adjustements for the normalized and trends_lin options 

///// Compute \delta^D for normalized option

if "`normalized'"!=""{
	
	capture drop sum_treat_until_`i'_XX
	capture drop delta_D_`i'_cum_temp_XX
	
	if `continuous'==0{
		bys group_XX : gegen sum_treat_until_`i'_XX = total(treatment_XX - d_sq_XX) if time_XX>=F_g_XX&time_XX<=F_g_XX-1+`i'&S_g_XX==increase_XX
	}
	// Redefine this with original treatment if continuous is defined (treatment was binarized and staggerized)
	else if `continuous'>0{
		bys group_XX : gegen sum_treat_until_`i'_XX = total(treatment_XX_orig - d_sq_XX_orig) if time_XX>=F_g_XX&time_XX<=F_g_XX-1+`i'&S_g_XX==increase_XX
	}
gen delta_D_`i'_cum_temp_XX = N_gt_XX/N`=increase_XX'_`i'_XX*[sum_treat_until_`i'_XX* S_g_XX + (1-S_g_XX)*(-sum_treat_until_`i'_XX)] if distance_to_switch_`i'_XX==1 

	sum  delta_D_`i'_cum_temp_XX
	scalar delta_norm_`i'_XX = r(sum) 

}

} // End of the loop over l_u_a_XX.
// At this point we have all the U_Gg_`i'_XX we need. Thus, we can sum them now for the trends_lin option.

///// trends_lin option

// As trends_lin relies on summing up the l effects we can only estimate effect l when we can estimate all prior l-1 effects, we verify if this condition is met.

scalar Ntrendslin=1
forvalue i=1/`=l_u_a_XX' {
scalar Ntrendslin=min(Ntrendslin , N`=increase_XX'_`i'_XX )
}

// Compute the U_Gg_\ell for trends_lin

if "`trends_lin'"!="" & Ntrendslin != 0  {
	
	capture drop U_Gg`=l_u_a_XX'_TL
	capture drop U_Gg`=l_u_a_XX'_var_TL
	
	// Initializing at 0
	gen U_Gg`=l_u_a_XX'_TL = 0
	gen U_Gg`=l_u_a_XX'_var_TL = 0

	// summing up the U_Gg's up to the l-th (current) effect
	forvalue i=1/`=l_u_a_XX'{
		replace U_Gg`=l_u_a_XX'_TL = U_Gg`=l_u_a_XX'_TL + U_Gg`i'_XX 		
		replace U_Gg`=l_u_a_XX'_var_TL =  U_Gg`=l_u_a_XX'_var_TL + U_Gg`i'_var_XX 
	}
	
	// replacing the U_Gg's with the one adjusted for group specific linear trends
	replace U_Gg`=l_u_a_XX'_XX = U_Gg`=l_u_a_XX'_TL
	replace U_Gg`=l_u_a_XX'_var_XX = U_Gg`=l_u_a_XX'_var_TL
		
}

////////// 5. Data preparation steps to generate variables necessary for computation of placebo effects	

if `placebo'!=0{
	if `=l_placebo_u_a_XX'>=1{
	
forvalue i=1/`=l_placebo_u_a_XX'{

///// Capture drop of variables created below
capture drop diff_y_pl_`i'_XX
capture drop U_Gg_pl_`i'_temp_XX
capture drop U_Gg_placebo_`i'_XX
capture drop U_Gg_pl_`i'_temp_var_XX
capture drop U_Gg_pl_`i'_var_XX
capture drop dist_to_switch_pl_`i'_XX
capture drop never_change_d_pl_`i'_XX
capture drop N`=increase_XX'_t_placebo_`i'_XX
capture drop N`=increase_XX'_t_pl_`i'_XX_w
capture drop N`=increase_XX'_t_placebo_`i'_g_XX
capture drop N_gt_control_placebo_`i'_XX
capture drop dummy_U_Gg_pl_`i'_XX 
capture drop never_change_d_pl_`i'_wXX
capture drop dist_to_switch_pl_`i'_wXX

capture drop dof_cohort_pl_`i'_s_t_XX
capture drop dof_cohort_pl_`i'_ns_t_XX
capture drop dof_cohort_pl_`i'_s0_t_XX
capture drop dof_cohort_pl_`i'_s1_t_XX
capture drop dof_cohort_pl_`i'_s2_t_XX
capture drop count_cohort_pl_`i'_s_t_XX
capture drop count_cohort_pl_`i'_ns_t_XX
capture drop count_cohort_pl_`i'_s0_t_XX
capture drop count_cohort_pl_`i'_s1_t_XX
capture drop count_cohort_pl_`i'_s2_t_XX
capture drop total_cohort_pl_`i'_s_t_XX
capture drop total_cohort_pl_`i'_ns_t_XX
capture drop total_cohort_pl_`i'_s0_t_XX
capture drop total_cohort_pl_`i'_s1_t_XX
capture drop total_cohort_pl_`i'_s2_t_XX
capture drop mean_cohort_pl_`i'_s_t_XX
capture drop mean_cohort_pl_`i'_ns_t_XX
capture drop mean_cohort_pl_`i'_s0_t_XX
capture drop mean_cohort_pl_`i'_s1_t_XX
capture drop mean_cohort_pl_`i'_s2_t_XX

capture drop E_hat_gt_pl_`i'_XX
capture drop DOF_gt_pl_`i'_XX

//The steps to compute the placebos are:
// 1. to place the corresponding outcome (y_{F_g-1} - y_{F_g - l - 1})) values in the same row of that (y_{F_g + l -1} - y_{F_g - 1}) of the symmetric DID_l. 
// 2. The other variables, such as N_gt, N0_l or N1_l, remain unchanged, except that we have to check if diff_y_placebo ( = y_{F_g - 2l -2}- y_{F_g - l -1}) exists. 

///// Computing the long differences for the placebos
xtset group_XX time_XX
bys group_XX : gen diff_y_pl_`i'_XX = L`=2*`i''.outcome_XX - L`i'.outcome_XX

///// Identifying the control (g,t)s in the estimation of placebo i 
bys group_XX: gen never_change_d_pl_`i'_XX=never_change_d_`i'_XX*(diff_y_pl_`i'_XX!=.) 
gen never_change_d_pl_`i'_wXX=never_change_d_pl_`i'_XX*N_gt_XX

///// number of control groups for g at t
bys time_XX d_sq_XX `trends_nonparam': gegen N_gt_control_placebo_`i'_XX=total(never_change_d_pl_`i'_wXX) 

///// Creating binary variable indicating whether g is \ell periods away from switch & (diff_y_pl_`i'_XX!=.) is well defined -> based on the already defined distance_to_switch_`i'_XX variable 
*Modif Clément 26/09/2025:
*gen dist_to_switch_pl_`i'_XX=distance_to_switch_`i'_XX*(diff_y_pl_`i'_XX!=.)
gen dist_to_switch_pl_`i'_XX=distance_to_switch_`i'_XX*(diff_y_pl_`i'_XX!=.)*(N_gt_control_placebo_`i'_XX>0&N_gt_control_placebo_`i'_XX!=.)
 
if "`same_switchers_pl'"!=""{
	replace dist_to_switch_pl_`i'_XX=dist_to_switch_pl_`i'_XX*fillin_g_pl_XX
}

///// Creating a variable counting the number of groups \ell periods away from switch at t
gen dist_to_switch_pl_`i'_wXX= dist_to_switch_pl_`i'_XX*N_gt_XX
bys time_XX: gegen N`=increase_XX'_t_placebo_`i'_XX=total(dist_to_switch_pl_`i'_wXX)

///// Computing N^1_\ell/N^0_\ell. for the placebos

// Initializing the N1_`i'_XX/N0_`i'_XX scalar at 0. 
scalar N`=increase_XX'_placebo_`i'_XX=0

// Loop over t incrementing the scalar
forvalue t=`=t_min_XX'/`=T_max_XX'{
	sum N`=increase_XX'_t_placebo_`i'_XX if time_XX==`t'
	if !missing(r(mean)) scalar N`=increase_XX'_placebo_`i'_XX = N`=increase_XX'_placebo_`i'_XX + r(mean) // Modif Diego
}

// Modif Felix weight
if "`weight'"!=""{

	bys time_XX: gegen N`=increase_XX'_t_pl_`i'_XX_w=total(dist_to_switch_pl_`i'_XX)
	
	scalar N`=increase_XX'_`i'_pl_XX_weight=0

// Loop over t incrementing the scalar
forvalue t=`=t_min_XX'/`=T_max_XX'{
	sum N`=increase_XX'_t_pl_`i'_XX_w if time_XX==`t'
	if !missing(r(mean)) scalar N`=increase_XX'_`i'_pl_XX_weight = N`=increase_XX'_`i'_pl_XX_weight + r(mean) // Modif Diego
}
}

///// Creating N^1_{t,\ell,g}/N^0_{t,\ell,g} for the placebos: 
///// Variable counting number of groups \ell periods away from switch at t, 
///// and with same D_{g,1} and trends_nonparam.

bys time_XX d_sq_XX `trends_nonparam': gegen N`=increase_XX'_t_placebo_`i'_g_XX=total(dist_to_switch_pl_`i'_wXX)

///// Creating all the adjustment terms to compute estimators with controls, and their variances 
if "`controls'" != ""{

// Initialize intermediate Variable needed later
capture drop part2_pl_switch`=increase_XX'_`i'_XX
gen part2_pl_switch`=increase_XX'_`i'_XX=0

local count_controls=0

// Computing the long differences of the control variables (X_g_t - X_g_t-l)

foreach var of varlist `controls'{

local count_controls=`count_controls'+1

capture drop diff_X_`count_controls'_placebo_`i'_XX

xtset group_XX time_XX
gen diff_X_`count_controls'_placebo_`i'_XX = L`=2*`i''.`var' - L`i'.`var'

// Computing N_g_t * (X_g_t - X_g_t-l)

capture drop diff_X`count_controls'_pl_`i'_N_XX
gen diff_X`count_controls'_pl_`i'_N_XX = N_gt_XX * diff_X_`count_controls'_placebo_`i'_XX 

foreach l of local levels_d_sq_XX { // index l corresponds to d in the paper 

capture drop m`=increase_XX'_pl_g_`count_controls'_`l'_`i'_XX 
gen m`=increase_XX'_pl_g_`count_controls'_`l'_`i'_XX = (`i' <= T_g_XX-2 & d_sq_int_XX == `l')* (G_XX / N`=increase_XX'_placebo_`i'_XX) * ([dist_to_switch_pl_`i'_XX - (N`=increase_XX'_t_placebo_`i'_g_XX/N_gt_control_placebo_`i'_XX) * never_change_d_pl_`i'_XX] * (time_XX>=`=`i'+1'&time_XX<=T_g_XX) * diff_X`count_controls'_pl_`i'_N_XX)
capture drop m_pl`=increase_XX'_`l'_`count_controls'_`i'_XX
bys group_XX: gegen m_pl`=increase_XX'_`l'_`count_controls'_`i'_XX=total(m`=increase_XX'_pl_g_`count_controls'_`l'_`i'_XX)
bys group_XX: replace m_pl`=increase_XX'_`l'_`count_controls'_`i'_XX = . if _n != 1
capture drop M_pl`=increase_XX'_`l'_`count_controls'_`i'_XX 
egen M_pl`=increase_XX'_`l'_`count_controls'_`i'_XX = total(m_pl`=increase_XX'_`l'_`count_controls'_`i'_XX)
replace M_pl`=increase_XX'_`l'_`count_controls'_`i'_XX = (1/G_XX)*M_pl`=increase_XX'_`l'_`count_controls'_`i'_XX
}

foreach l of local levels_d_sq_XX {
	if (scalar(useful_res_`l'_XX)>1){ 
replace diff_y_pl_`i'_XX = diff_y_pl_`i'_XX - coefs_sq_`l'_XX[`=`count_controls'',1]*diff_X_`count_controls'_placebo_`i'_XX if d_sq_int_XX==`l' 

capture drop in_brackets_pl_`l'_`count_controls'_XX	
gen in_brackets_pl_`l'_`count_controls'_XX=0
				
}		
}
}
}


///// Computing the variables used for the demeaning of outcome's placebo long difference diff_y_`i',
///// which we will use to create the U_g^{var} variables, which are used to compute 
///// the placebos' variances. 

// Generate placebo long-differences of outcomes time N_{g,t}, and dummy for (g,t) such that
// diff_y_`i' non missing and N_gt non missing.

capture drop diff_y_pl_`i'_N_gt_XX
gen diff_y_pl_`i'_N_gt_XX=N_gt_XX*diff_y_pl_`i'_XX
capture drop dof_ns_pl_`i'_XX
gen dof_ns_pl_`i'_XX = N_gt_XX!=0&diff_y_pl_`i'_XX!=.&never_change_d_pl_`i'_XX==1&N`=increase_XX'_t_placebo_`i'_XX>0&N`=increase_XX'_t_placebo_`i'_XX!=.
capture drop dof_s_pl_`i'_XX
gen dof_s_pl_`i'_XX = N_gt_XX!=0&dist_to_switch_pl_`i'_XX==1

// For never switchers, demeaning wrt to cohorts defined by D_{g,1}, `trends_nonparam' 
//(\mathcal{D}_k in companion paper)

// Modif Clément: we need to add by time
* Mean's denominator
bys d_sq_XX `trends_nonparam' time_XX : gegen count_cohort_pl_`i'_ns_t_XX=total(N_gt_XX) if dof_ns_pl_`i'_XX == 1

* Mean's numerator
bys d_sq_XX `trends_nonparam' time_XX : gegen total_cohort_pl_`i'_ns_t_XX=total(diff_y_pl_`i'_N_gt_XX) if dof_ns_pl_`i'_XX == 1

* Mean 
gen mean_cohort_pl_`i'_ns_t_XX=total_cohort_pl_`i'_ns_t_XX/count_cohort_pl_`i'_ns_t_XX

* Counting number of groups for DOF adjustment
{
	if "`cluster'" == "" {
		bys d_sq_XX `trends_nonparam' time_XX : gegen dof_cohort_pl_`i'_ns_t_XX=total(dof_ns_pl_`i'_XX) if dof_ns_pl_`i'_XX == 1
	}
	else {
		cap drop cluster_dof_pl_`i'_ns_XX
		gen cluster_dof_pl_`i'_ns_XX = `cluster' if dof_ns_pl_`i'_XX == 1
		bys d_sq_XX `trends_nonparam' time_XX: gegen dof_cohort_pl_`i'_ns_t_XX = nunique(cluster_dof_pl_`i'_ns_XX) if !missing(cluster_dof_pl_`i'_ns_XX)
	}
}
// End modif Clément

// For switchers, for placebos we no longer need to distinguish depending on whether the option
// less_conservative_se specified or not, we always demean wrt to 
// cohorts defined by D_{g,1}, F_g, D_{g,F_g}, `trends_nonparam' (\mathcal{C}_k in companion paper).

* Mean's denominator
bys d_sq_XX F_g_XX d_fg_XX `trends_nonparam' : gegen count_cohort_pl_`i'_s_t_XX=total(N_gt_XX) if dof_s_pl_`i'_XX==1

* Mean's numerator
bys d_sq_XX F_g_XX d_fg_XX `trends_nonparam' : gegen total_cohort_pl_`i'_s_t_XX=total(diff_y_pl_`i'_N_gt_XX) if dof_s_pl_`i'_XX==1 

* Mean 
gen mean_cohort_pl_`i'_s_t_XX=total_cohort_pl_`i'_s_t_XX/count_cohort_pl_`i'_s_t_XX	
		
* Counting number of groups for DOF adjustment
{
	if "`cluster'" == "" {
		bys d_sq_XX F_g_XX d_fg_XX `trends_nonparam' : gegen dof_cohort_pl_`i'_s_t_XX=total(dof_s_pl_`i'_XX) if dof_s_pl_`i'_XX==1
	}
	else {
		cap drop cluster_dof_pl_`i'_s_XX
		gen cluster_dof_pl_`i'_s_XX = `cluster' if dof_s_pl_`i'_XX == 1
		bys d_sq_XX F_g_XX d_fg_XX `trends_nonparam': gegen dof_cohort_pl_`i'_s_t_XX = nunique(cluster_dof_pl_`i'_s_XX) if !missing(cluster_dof_pl_`i'_s_XX)
	}
}

// Modif Clément 19/6/2025:
// If a switcher is the only one in their cohort or if a not-yet-switcher is the only one in their cohort, we demean wrt union of switchers and not-yet switchers.

cap drop dof_ns_s_pl_`i'_XX
cap drop count_cohort_pl_`i'_ns_s_t_XX
cap drop total_cohort_pl_`i'_ns_s_t_XX
cap drop mean_cohort_pl_`i'_ns_s_t_XX
cap drop dof_cohort_pl_`i'_ns_s_t_XX

gen dof_ns_s_pl_`i'_XX=(dof_s_pl_`i'_XX==1|dof_ns_pl_`i'_XX ==1)

* Mean's denominator
bys d_sq_XX `trends_nonparam' time_XX : gegen count_cohort_pl_`i'_ns_s_t_XX=total(N_gt_XX) if dof_ns_s_pl_`i'_XX==1

* Mean's numerator
bys d_sq_XX `trends_nonparam' time_XX: gegen total_cohort_pl_`i'_ns_s_t_XX=total(diff_y_pl_`i'_N_gt_XX) if dof_ns_s_pl_`i'_XX==1

* Mean 
gen mean_cohort_pl_`i'_ns_s_t_XX=total_cohort_pl_`i'_ns_s_t_XX/count_cohort_pl_`i'_ns_s_t_XX

* Counting number of groups for DOF adjustment
{
	if "`cluster'" == "" {
		bys d_sq_XX `trends_nonparam' time_XX: gegen dof_cohort_pl_`i'_ns_s_t_XX=total(dof_ns_s_pl_`i'_XX) if dof_ns_s_pl_`i'_XX==1
	}
	else {
		cap drop cluster_dof_pl_`i'_ns_s_XX
		gen cluster_dof_pl_`i'_ns_s_XX = `cluster' if dof_ns_s_pl_`i'_XX==1
		bys d_sq_XX `trends_nonparam' time_XX: gegen dof_cohort_pl_`i'_ns_s_t_XX = nunique(cluster_dof_pl_`i'_ns_s_XX) if !missing(cluster_dof_pl_`i'_ns_s_XX)
	}
}	

///// From those parts, generate variables for the demeaning and the DOF adjustment 
// E_hat_(g,t), defined from parts depending on the cohort definition 
gen E_hat_gt_pl_`i'_XX=0 if (time_XX<F_g_XX|F_g_XX-1+`i'==time_XX)
replace E_hat_gt_pl_`i'_XX=mean_cohort_pl_`i'_ns_t_XX if (dof_cohort_pl_`i'_ns_t_XX>=2&time_XX<F_g_XX)
replace E_hat_gt_pl_`i'_XX=mean_cohort_pl_`i'_s_t_XX if (dof_cohort_pl_`i'_s_t_XX>=2&F_g_XX-1+`i'==time_XX)
replace E_hat_gt_pl_`i'_XX=mean_cohort_pl_`i'_ns_s_t_XX if dof_cohort_pl_`i'_ns_s_t_XX>=2&(F_g_XX-1+`i'==time_XX&dof_cohort_pl_`i'_s_t_XX==1)|(time_XX<F_g_XX&dof_cohort_pl_`i'_ns_t_XX==1)

// DOF_(g,t) for DOF adjustement, defined from parts depending on the cohort definition 
// Diego - 02-03-25: when there is only 1 switcher, dof_cohort_`i'_s_t_XX = 1, hence the denominator in the expression below is 0
// The fraction is undefined and Stata puts it to missing
gen DOF_gt_pl_`i'_XX=1 if (time_XX<F_g_XX|F_g_XX-1+`i'==time_XX)
replace DOF_gt_pl_`i'_XX=  sqrt(dof_cohort_pl_`i'_s_t_XX/(dof_cohort_pl_`i'_s_t_XX-1)) if (F_g_XX-1+`i'==time_XX & dof_cohort_pl_`i'_s_t_XX > 1)
replace DOF_gt_pl_`i'_XX=  sqrt(dof_cohort_pl_`i'_ns_t_XX/(dof_cohort_pl_`i'_ns_t_XX-1)) if (time_XX<F_g_XX & dof_cohort_pl_`i'_ns_t_XX > 1)
replace DOF_gt_pl_`i'_XX=  sqrt(dof_cohort_pl_`i'_ns_s_t_XX/(dof_cohort_pl_`i'_ns_s_t_XX-1)) if dof_cohort_pl_`i'_ns_s_t_XX>=2&(F_g_XX-1+`i'==time_XX&dof_cohort_pl_`i'_s_t_XX==1)|(time_XX<F_g_XX&dof_cohort_pl_`i'_ns_t_XX==1)

// End modif Clément 19/6/2025.

////////// 6. Computing U_Gg_\ell variables for the placebos (similar to part 4, less commented)

gen dummy_U_Gg_pl_`i'_XX = (`i'<=T_g_XX-1)

if (N`=increase_XX'_placebo_`i'_XX!=0){

gen U_Gg_pl_`i'_temp_XX = dummy_U_Gg_pl_`i'_XX*(G_XX / N`=increase_XX'_placebo_`i'_XX) * (time_XX>=`=`i'+1'&time_XX<=T_g_XX)* N_gt_XX * [dist_to_switch_pl_`i'_XX - (N`=increase_XX'_t_placebo_`i'_g_XX/N_gt_control_placebo_`i'_XX) * never_change_d_pl_`i'_XX] *diff_y_pl_`i'_XX 

bysort group_XX : gegen U_Gg_placebo_`i'_XX=total(U_Gg_pl_`i'_temp_XX)

replace U_Gg_placebo_`i'_XX= U_Gg_placebo_`i'_XX*first_obs_by_gp_XX

capture drop count`i'_pl_core_XX
gen count`i'_pl_core_XX=0
replace count`i'_pl_core_XX= N_gt_XX if (U_Gg_pl_`i'_temp_XX!=.&U_Gg_pl_`i'_temp_XX!=0|(U_Gg_pl_`i'_temp_XX==0&diff_y_pl_`i'_XX==0&(dist_to_switch_pl_`i'_XX!=0|(N`=increase_XX'_t_placebo_`i'_g_XX!=0&never_change_d_pl_`i'_XX!=0))))

gen U_Gg_pl_`i'_temp_var_XX =0

replace U_Gg_pl_`i'_temp_var_XX= dummy_U_Gg_pl_`i'_XX*(G_XX / N`=increase_XX'_placebo_`i'_XX) * [dist_to_switch_pl_`i'_XX - (N`=increase_XX'_t_placebo_`i'_g_XX/N_gt_control_placebo_`i'_XX) * never_change_d_pl_`i'_XX] * (time_XX>=`=`i'+1'&time_XX<=T_g_XX) * N_gt_XX * DOF_gt_pl_`i'_XX *(diff_y_pl_`i'_XX-E_hat_gt_pl_`i'_XX)

if "`controls'"!=""{

levelsof d_sq_int_XX, local(levels_d_sq_XX)
foreach l of local levels_d_sq_XX {	
if (scalar(useful_res_`l'_XX)>1){ // MODIF FELIX
	
capture drop combined_pl`=increase_XX'_temp_`l'_`i'_XX	
gen combined_pl`=increase_XX'_temp_`l'_`i'_XX=0
		
forvalues j=1/`count_controls'{
forvalues k=1/`count_controls'{		
	
capture drop in_brackets_pl_`l'_`j'_`k'_temp_XX
//// CHANGE BELOW - tks + Changes Diego 27-03-25: remove N_c_`l'_XX from formula below, since we have adjusted inv_Denom above
gen in_brackets_pl_`l'_`j'_`k'_temp_XX = inv_Denom_`l'_XX[`j',`k'] * in_sum_`k'_`l'_XX * (d_sq_int_XX == `l' & F_g_XX>=3)
////
replace in_brackets_pl_`l'_`j'_XX=in_brackets_pl_`l'_`j'_XX+in_brackets_pl_`l'_`j'_`k'_temp_XX
}

replace in_brackets_pl_`l'_`j'_XX=in_brackets_pl_`l'_`j'_XX - coefs_sq_`l'_XX[`j',1]

capture drop combined_pl`=increase_XX'_temp_`l'_`j'_`i'_XX 
gen combined_pl`=increase_XX'_temp_`l'_`j'_`i'_XX=M_pl`=increase_XX'_`l'_`j'_`i'_XX*in_brackets_pl_`l'_`j'_XX

replace combined_pl`=increase_XX'_temp_`l'_`i'_XX=combined_pl`=increase_XX'_temp_`l'_`i'_XX+combined_pl`=increase_XX'_temp_`l'_`j'_`i'_XX
} 

replace part2_pl_switch`=increase_XX'_`i'_XX=part2_pl_switch`=increase_XX'_`i'_XX+combined_pl`=increase_XX'_temp_`l'_`i'_XX
} // MODIF FELIX
} // end loop oveer l

}	

bys group_XX: gegen U_Gg_pl_`i'_var_XX=total(U_Gg_pl_`i'_temp_var_XX)

// Modeif Felix 21.03.2025 (adjust order of adding additional term and summing across t)
if "`controls'"!=""{
	if `=increase_XX'==1{
	replace U_Gg_pl_`i'_var_XX=U_Gg_pl_`i'_var_XX - part2_pl_switch1_`i'_XX 
	}

	if `=increase_XX'==0{
	//// CHANGE BELOW - tks (change plus to minus since final sign flip won't be until later)
	replace U_Gg_pl_`i'_var_XX=U_Gg_pl_`i'_var_XX - part2_pl_switch0_`i'_XX 	
	}
}

}

////////// 7. Computing adjustements for the normalized and trends_lin options for placebos
////////// (similar to part 4, not commented) 

if "`normalized'"!=""{
	
	capture drop sum_treat_until_`i'_pl_XX
	capture drop delta_D_pl_`i'_cum_temp_XX

	if `continuous'==0{
	bys group_XX : gegen sum_treat_until_`i'_pl_XX = total(treatment_XX - d_sq_XX) if time_XX>=F_g_XX&time_XX<=F_g_XX-1+`i'& S_g_XX==increase_XX
	} // FELIX 18.03.2025 (delete the diff_y_pl_`i'_XX!=. condition)

	else if `continuous'>0{
		bys group_XX : gegen sum_treat_until_`i'_pl_XX = total(treatment_XX_orig - d_sq_XX_orig) if time_XX>=F_g_XX&time_XX<=F_g_XX-1+`i'&S_g_XX==increase_XX
	} // FELIX 18.03.2025 (delete the diff_y_pl_`i'_XX!=. condition)

gen delta_D_pl_`i'_cum_temp_XX = N_gt_XX/N`=increase_XX'_placebo_`i'_XX*[sum_treat_until_`i'_pl_XX* S_g_XX + (1-S_g_XX)*(-sum_treat_until_`i'_pl_XX)] if dist_to_switch_pl_`i'_XX==1

	sum  delta_D_pl_`i'_cum_temp_XX
	scalar delta_norm_pl_`i'_XX = r(sum)
}



} // End of the loop over l_placebo_u_a_XX.
}

scalar Ntrendslin_pl=1
forvalue i=1/`=l_placebo_u_a_XX' {
scalar Ntrendslin_pl=min(Ntrendslin_pl , N`=increase_XX'_placebo_`i'_XX )
}


if "`trends_lin'"!="" & Ntrendslin_pl != 0 {
	
	capture drop U_Gg_pl_`=l_placebo_u_a_XX'_TL
	capture drop U_Gg_pl_`=l_placebo_u_a_XX'_var_TL
	
	gen U_Gg_pl_`=l_placebo_u_a_XX'_TL = 0
	gen U_Gg_pl_`=l_placebo_u_a_XX'_var_TL = 0

	forvalue i=1/`=l_placebo_u_a_XX'{
		replace U_Gg_pl_`=l_placebo_u_a_XX'_TL = U_Gg_pl_`=l_placebo_u_a_XX'_TL + U_Gg_placebo_`i'_XX
		replace U_Gg_pl_`=l_placebo_u_a_XX'_var_TL = U_Gg_pl_`=l_placebo_u_a_XX'_var_TL + U_Gg_pl_`i'_var_XX
	
	}
	
	replace U_Gg_placebo_`=l_placebo_u_a_XX'_XX = U_Gg_pl_`=l_placebo_u_a_XX'_TL
	replace U_Gg_pl_`=l_placebo_u_a_XX'_var_XX =U_Gg_pl_`=l_placebo_u_a_XX'_var_TL

}
} //End of condition checking if at least one placebo requested.


////////// 8. Computing Average Total Effect estimator

///// Average total effect not estimated if trends_lin option requested:
if "`trends_lin'"==""{

///// Computing the sum of the N1_`i'_XX for the weights w. 
scalar sum_N`=increase_XX'_l_XX = 0
forvalue i=1/`=l_u_a_XX'{
	scalar sum_N`=increase_XX'_l_XX = sum_N`=increase_XX'_l_XX + N`=increase_XX'_`i'_XX
}	

///// Dropping/Initializing needed variables
capture drop U_Gg_XX
capture drop U_Gg_num_XX
capture drop U_Gg_den_XX
capture drop U_Gg_num_var_XX
capture drop  U_Gg_var_XX
gen U_Gg_num_XX=0
gen U_Gg_den_XX=0
gen U_Gg_num_var_XX=0


forvalue i=1/`=l_u_a_XX'{
	
if (N`=increase_XX'_`i'_XX!=0){
capture drop delta_D_`i'_temp_XX
capture drop delta_D_`i'_XX
		 
///// Computing the weights w_+,l
scalar w_`i'_XX = N`=increase_XX'_`i'_XX / sum_N`=increase_XX'_l_XX
		
///// Computing the delta_D_+,l", which enter in the denominator of \hat{\delta}_+/\hat{\delta}_-.

if `continuous'==0{
gen delta_D_`i'_temp_XX = N_gt_XX/N`=increase_XX'_`i'_XX*[(treatment_XX-d_sq_XX)* S_g_XX + (1-S_g_XX)*(d_sq_XX-treatment_XX)] if distance_to_switch_`i'_XX==1
}
// Redefine this with original treatment if continuous option specified (treatment was binarized and staggerized)
else if `continuous'>0{
gen delta_D_`i'_temp_XX = N_gt_XX/N`=increase_XX'_`i'_XX*[(treatment_XX_orig-d_sq_XX_orig)* S_g_XX + (1-S_g_XX)*(d_sq_XX_orig-treatment_XX_orig)] if distance_to_switch_`i'_XX==1	
}	

// For the average cumulative effect we want to keep delta_D_`i'_temp_XX and rescale it by N`=increase_XX'_`i'_XX/N_gt_XX to get an unweightened version of [D_{g,Fg-1+l} - D_{g,1}]	
// Maybe call it delta_D_g_`i'_XX instead of delta_D_`i'_temp_XX and then keep it?	
capture drop delta_D_g_`i'_XX
replace delta_D_`i'_temp_XX=0 if delta_D_`i'_temp_XX==.
gegen delta_D_`i'_XX = total(delta_D_`i'_temp_XX)
gen delta_D_g_`i'_XX=delta_D_`i'_temp_XX * (N`=increase_XX'_`i'_XX/N_gt_XX)
drop delta_D_`i'_temp_XX	
	
///// Computing the numerator of U^+_{G,g}: summing up the U_{G,g,l}s, after weighting them
bys group_XX : replace U_Gg_num_XX = U_Gg_num_XX + w_`i'_XX * U_Gg`i'_XX

///// Computing the numerator of the "alternative" U_{G,g}s for the variance : summing up the U_{G,g,l}_vars, after weighting them
bys group_XX : replace U_Gg_num_var_XX = U_Gg_num_var_XX + w_`i'_XX * U_Gg`i'_var_XX

///// Computing the denominator of U^+_{G,g}: summing up the delta^D_{+,l}s, after weighting them
bys group_XX : replace U_Gg_den_XX = U_Gg_den_XX + w_`i'_XX * delta_D_`i'_XX

}

}
		
///// Computing the U_+_(G,g)s".
bys group_XX : gen U_Gg_XX = U_Gg_num_XX/U_Gg_den_XX

///// Computing the U^+_{G,g}_vars.
bys group_XX : gen U_Gg_var_XX = U_Gg_num_var_XX/U_Gg_den_XX 

}
}
 // end of the quietly condition
	
end


