{smcl}
{* *! version 1  2025-03-25}{...}
{viewerjumpto "Syntax" "did_multiplegt_dyn##syntax"}{...}
{viewerjumpto "Description" "did_multiplegt_dyn##description"}{...}
{viewerjumpto "Further detail" "did_multiplegt_dyn##detail"}{...}
{viewerjumpto "Options" "did_multiplegt_dyn##options"}{...}
{viewerjumpto "Examples" "did_multiplegt_dyn##examples"}{...}
{viewerjumpto "Saved results" "did_multiplegt_dyn##saved_results"}{...}

{title:Title}

{p 4 4}
{cmd:did_multiplegt_dyn} {hline 2} Estimation of heterogeneity-robust 
difference-in-differences (DID) estimators, with a binary, discrete, or 
continuous treatment, in designs where past treatments may 
affect the current outcome.
{p_end}

{marker syntax}{...}
{title:Syntax}

{p 4 8}
{cmd:did_multiplegt_dyn Y G T D} {ifin}
[{cmd:,}
{cmd:effects(}{it:#}{cmd:)}
{cmd:design(}{it:string}{cmd:)}
{cmd:normalized}
{cmd:normalized_weights}
{cmd:effects_equal(}{it:lower bound, upper bound}{cmd:)}
{cmd:placebo(}{it:#}{cmd:)}
{cmd:controls(}{it:varlist}{cmd:)}
{cmd:trends_nonparam(}{it:varlist}{cmd:)}
{cmd:trends_lin}
{cmd:continuous(}{it:#}{cmd:)}
{cmd:weight(}{it:varname}{cmd:)}
{cmd:cluster(}{it:varname}{cmd:)}
{cmd:by(}{it:varname}{cmd:)}
{cmd:by_path(}{it:#}{cmd:)}
{cmd:predict_het(}{it:varlist,numlist}{cmd:)}
{cmd:predict_het_hc2bm}
{cmd:date_first_switch(}[by_baseline_treat]{it:,string}{cmd:)}
{cmd:same_switchers}
{cmd:same_switchers_pl}
{cmd:switchers(}{it:string}{cmd:)}
{cmd:only_never_switchers}
{cmd:ci_level(}{it:#}{cmd:)}
{cmd:graphoptions(}{it:string}{cmd:)}
{cmd:graph_off}
{cmd:save_results(}{it:path}{cmd:)}
{cmd:save_sample}
{cmd:more_granular_demeaning}
{cmd:bootstrap(}{it:#,#}{cmd:)}
{cmd:dont_drop_larger_lower}
{cmd:drop_if_d_miss_before_first_switch}
{cmd:_no_updates}]
{p_end}

{synoptset 28 tabbed}{...}

{marker description}{...}
{title:Description}

{p 4 8}
{cmd:did_multiplegt_dyn} estimates the effect of a treatment 
on an outcome, using group-(e.g. county- or state-) level 
panel data. The command computes the DID event-study estimators 
introduced in de Chaisemartin and D'Haultfoeuille (2024). 
Like other recently proposed DID estimation commands ({cmd:csdid}, {cmd:didimputation}, ...), 
{cmd:did_multiplegt_dyn} can be used with a binary and 
staggered (absorbing) treatment. But unlike those other commands, 
{cmd:did_multiplegt_dyn} can also be used with a non-binary treatment
(discrete or continuous) that can increase or decrease multiple times.
Lagged treatments may affect the outcome, and the current and lagged 
treatments may have heterogeneous effects, across space and/or over time. 
The event-study estimators computed by the command rely on a 
no-anticipation and parallel trends assumptions. 
The panel may be unbalanced:  not all groups have to be observed at every period.  
The data may also be at a more disaggregated level than the group level 
(e.g. individual-level wage data to measure the effect of a 
regional-level minimum-wage on individuals' wages). 
{p_end}

{p 8 8}
{cmd:Y} is the outcome variable.
{p_end}

{p 8 8}
{cmd:G} is the group variable.
{p_end}

{p 8 8}
{cmd:T} is the time period variable.
The command assumes that the time variable is evenly spaced
(e.g.: the panel is at the yearly level,
and no year is missing for all groups).
When it is not (e.g.: the panel is at the yearly level,
but three consecutive years are missing for all
groups), the command can still be used,
see FAQ section below.
{p_end}

{p 8 8}
{cmd:D} is the treatment variable.
{p_end}


{marker detail}{...}
{title:Further detail}

{p 4 8}
{cmd:Non-normalized event-study estimators (default)} - For all "switchers", namely groups 
that experience a change of their treatment over the study period, let F_g denote 
the first time period when g's treatment changes. The command computes the non-normalized 
event-study estimators DID_ℓ.  DID_1 is the average, across all switchers, of DID estimators 
comparing the F_g-1 to F_g outcome evolution of g to that of groups with the same period-one 
treatment as g but whose treatment has not changed yet at F_g.  More generally, DID_ℓ is the 
average, across all switchers, of DID estimators comparing the F_g-1 to F_g-1+ℓ outcome 
evolution of g to that of groups with the same period-one treatment as g but whose treatment 
has not changed yet at F_g-1+ℓ.  Those estimators are unbiased for non-normalized event-study 
effects, which are average effects of having been exposed to a weakly higher treatment dose 
for ℓ periods, where the magnitude and timing of the incremental treatment doses can vary 
across groups.  
{p_end}

{p 4 8}
{cmd:Normalized event-study estimators} - The command also computes the normalized event-study 
estimators DID^n_ℓ, that normalize DID_ℓ by the average cumulative (total) incremental 
treatment dose received by switchers from F_g-1 to F_g-1+ℓ with respect to their period-one 
treatment.  This normalization ensures that DID^n_ℓ estimates a weighted average of the 
effects of the current treatment and of its ℓ-1 first lags on the outcome. While the effects 
of the current and lagged treatments cannot be separately estimated, the weight that DID^n_ℓ 
puts on the effect of each lag can be estimated.    
{p_end}

{p 4 8}
{cmd:Average cumulative (total) effect per dose} - The command also computes an estimated 
average cumulative (total) effect per unit of treatment, where “cumulative effect” refers 
to the sum of the effects of a treatment dose, at the time when it takes place and at 
later periods, see Section 3.3 of de Chaisemartin and D'Haultfoeuille (2024) for further 
details. The command also shows the number of time periods over which the effect of a dose 
is accumulated, on average across all incremental doses received by switchers over the study 
period. By dividing the average cumulative effect by the average number of periods across 
which effects are accumulated, one can get an estimator of the effect of being exposed to 
one more dose for one more period.  
{p_end}

{p 4 8}
{cmd:Placebos} - The command also computes placebo estimators, that average DIDs comparing 
the outcome evolution of switcher g and of its control groups, from F_g-1 to F_g-1-ℓ, namely 
before g's treatment changes for the first time.  Those placebos can be used to test the 
parallel trends and no-anticipation assumptions under which the estimators computed by 
{cmd:did_multiplegt_dyn} are unbiased.  
{p_end}

{p 4 8}
{cmd:Designs compatible with the command} - The command accommodates many DID designs. It 
covers the canonical binary and absorbing treatment case. It also covers more complicated 
treatment paths: groups may have heterogeneous treatments at period one, their treatment 
may change at different dates, some groups may experience increases in their treatment 
while other groups experience decreases, some groups may experience more than one change 
of their treatment, and finally some groups may experience larger treatment changes than 
others. The command can also be used to separately estimate the effects of several treatment 
variables, see references in the FAQ section. The only requirement is that not all groups 
experience their first treatment change at the same date. If groups’ period-one treatment 
is continuously distributed, meaning essentially that all groups have a different period-one 
treatment, the option {cmd:continuous} needs to be used.
{p_end}

{p 4 8}
{cmd:Relaxing parallel trends assumptions} - This command allows for many relaxations of 
the parallel-trends assumption: see the {cmd:controls} option for estimators allowing for  
time-varying covariates, see the {cmd:trends_lin} option for estimators allowing for 
group-specific linear trends, and see the {cmd:trends_nonparam} option for estimators 
allowing to interact time fixed effects with time-invariant variables (e.g. industry*year 
effect with firm-level panel data). 
{p_end}


{marker options}{...}
{title:Options}

{p 4 8}
{cmd:effects(}{it:#}{cmd:)} gives the number of event-study effects to be estimated.
By default, the command estimates non-normalized event-study effects.
Non-normalized event-study effects are averages, across all switchers, of the effect of 
having received their actual rather than their period-one treatment dose, for ℓ periods.
While non-normalized event-study effects can be interpreted as average effects of being 
exposed to a weakly higher treatment dose for ℓ periods,
the magnitude and timing of the incremental treatment doses can vary across groups.
{p_end}

{p 4 8}
{cmd:design(}[{it:float}]{it:, string}{cmd:)}:
this option reports switchers' period-one and subsequent treatments, thus helping the 
analyst understand the treatment paths whose effect is aggregated in the
non-normalized event-study effects. When the number of treatment paths is low, 
one may consider estimating treatment-path-specific event-study effects to facilitate interpretation, see footnote 10
of de Chaisemartin and D'Haultfoeuille (2024) for detailed instructions. When 
the number of treatment paths is large, one may specify a number included between 
0 and 1 in the {it:float} argument. Then the command reports the treatment 
paths common to at least ({it:float}*100)% of switchers. Results can be 
printed in the Stata console specifying {it:console} as the string argument. 
For example, {cmd:did_multiplegt_dyn Y G T D, effects(5) design(0.5, console)} 
reports the treatment paths experienced by at least 50% of the 
switchers and prints the output in the Stata console. Alternatively, 
the output can be stored in an Excel file providing a valid file path as 
the string argument.
{p_end}

{p 4 8}
{cmd:normalized}: when this option is specified,
the command estimates normalized event-study effects,
that are equal to a weighted average of the effects
of the current treatment and of its ℓ-1 first lags on the outcome. 
See Section 3.2 of de Chaisemartin and D'Haultfoeuille (2024)
for further details.
{p_end}

{p 4 8}
{cmd:normalized_weights}: when this option and the {cmd:normalized} option are specified,
the command reports the weights that
normalized effect ℓ puts on the effect of the current treatment, 
on the effect of the first treatment lag, etc.
{p_end}

{p 4 8}
{cmd:effects_equal(}{it:lower bound, upper bound}{cmd:)} or {cmd:effects_equal(}{it:"all"}{cmd:)} : When this option is
specified with a lower 
and upper bound, the command performs an F-test to determine
whether all effects within the specified range are equal. The lower and upper bounds
should belong to the range of estimated effects. If the argument {it:“all”} is provided, the
command defaults to testing whether all estimated effects are equal. When used with the
{cmd:normalized} option, this test can help assess whether the current and lagged treatments
have the same effect on the outcome or if their effects differ. See Lemma 7 of de
Chaisemartin and D'Haultfoeuille (2024).
{p_end}

{p 4 8}
{cmd:placebo(}{it:#}{cmd:)} gives the number of placebo estimators to be computed.
Placebos compare the outcome evolution of switchers and of their controls,
before switchers' treatment changes for the first time.
Under the parallel trends and no-anticipation assumptions
underlying the event-study estimators computed by {cmd:did_multiplegt_dyn}, 
the expectation of the placebos is equal to zero.
Thus, placebos can be used to test those assumptions, by testing the null that
all placebos are equal to zero.
If the user requests that at least two placebos be estimated,
the command computes the p-value of a joint test of that null hypothesis.
The number of placebos requested can be at most
equal to the number of time periods in the data minus 2,
though most often only a smaller number of placebos can be computed.
Also, the number of placebos requested cannot be larger
than the number of effects requested.
{p_end}

{p 4 8}
{cmd:controls(}{it:varlist}{cmd:)} gives the names of
the control variables to be included in the estimation.
Estimators with controls are similar
to those without controls,
except that the first-difference of the outcome is
replaced by residuals from regressions
of the first-difference of the outcome
on the first-differences of the controls and time fixed effects.
Those regressions are estimated in the sample of control (g,t)s:
(g,t)s such that group g's treatment has not changed yet at t.
Those regressions are also estimated separately
for each value of the period-one treatment.
Estimators with controls are unbiased
even if groups experience differential trends,
provided such differential trends can be
fully explained by a linear model in covariates changes.
To control for time-invariant covariates,
one can for instance input the product of those covariates and of the time variable {cmd:T} into the option.
See Section 1.2 of the Web Appendix
of de Chaisemartin and D'Haultfoeuille (2024) for further details.
{p_end}

{p 4 8}
{cmd:trends_lin}: when this option is specified, the estimation of the treatment effects allows for group-specific linear trends.
Estimators with linear trends start by computing event-study effects on the outcome's 
first-difference, rather than on the outcome itself, thus allowing for group-specific linear trends.
Then, to recover event-study effect ℓ on the outcome, event-study effects on the outcome's 
first-difference are summed from 1 to ℓ. See Section 1.3 of the Web Appendix
of de Chaisemartin and D'Haultfoeuille (2024) for further details. When this option is 
specified, the estimated average cumulative (total)
effect per unit of treatment is not computed.
{p_end}

{p 4 8}
{cmd:trends_nonparam(}{it:varlist}{cmd:)}: when this option is specified, the 
DID estimators computed by the command only compare switchers to controls whose 
treatment has not changed yet, with the same period-one treatment, and with the 
same value of {it:varlist}.  Estimators with the {cmd:trends_nonparam(}{it:varlist}{cmd:)} 
option are unbiased even if groups experience differential trends, provided 
all groups with the same value of {it:varlist} experience parallel trends. 
{it:varlist} can only include time-invariant variables, and the interaction 
of those variables has to be coarser than the group variable.  For instance, 
if one works with a county*year data set and one wants to allow for state-specific 
trends, one should specify {cmd:trends_nonparam(}state{cmd:)}, where state is the 
state identifier. Similarly, if one works with a firm*year data and one wants to 
allow for industry-specific trends, one should specify {cmd:trends_nonparam(}industry{cmd:)}. 
See Section 1.4 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2024) for further details.
{p_end}

{p 4 8}
{cmd:continuous(}{it:#}{cmd:)} allows to use the command even when groups' 
period-one treatment is continuous, meaning that all groups have a different period-one treatment value. 
With a discrete period-one treatment, the command compares the outcome evolution of 
switchers and non-switchers with the same period-one treatment. 
But with a truly continuous period-one treatment, there will be no two groups with 
the same period-one treatment. Then, the command assumes that group's status-quo 
outcome evolution is a polynomial in their period-one treatment. The user's chosen 
polynomial order is the option's argument. See Section 1.10 of the Web Appendix
of de Chaisemartin and D'Haultfoeuille (2024) for further details.
Unlike the other variance estimators computed by the command, those 
computed when the {cmd:continuous} option is specified are not backed by 
a proven asymptotic normality result. Preliminary simulation evidence indicates that when the option is used with a 
correctly-specified polynomial order, those variance estimators are conservative. 
On the other hand, when the specified polynomial order is strictly larger than needed, 
those variance estimators can become liberal. Thus, when this option is specified, we 
recommend using the bootstrap for inference, using the {cmd:bootstrap} option. 
At least, one should perform a robustness check where one compares 
the analytic variance computed by the command to a bootstrapped variance.
This option can not be combined with the {cmd:design} option.
{p_end}

{p 4 8}
{cmd:weight(}{it:varname}{cmd:)} specifies the name of
a variable used to weight the data.
For instance,
if one works with a district*year data set
and one wants to weight the estimation
by each district*year's population,
one should specify {cmd:weight(}population{cmd:)}.
If the data set is at a more disaggregated level than group*time,
the command aggregates it at the group*time
level internally, and weights the estimation
by the number of observations in each group*time cell
if the weight option is not specified,
or by the sum of the weights of the observations
in each group*time cell if the weight option is specified.
{p_end}

{p 4 8}
{cmd:cluster(}{it:varname}{cmd:)} can be used to cluster the estimators' standard
errors. Only one clustering variable is allowed.
A common practice in DID analysis is to cluster standard
errors at the group level. Such clustering is implemented by default by the command.
Standard errors can be clustered at a more aggregated level than the group level,
but they cannot be clustered at a more disaggregated level.
{p_end}

{p 4 8}
{cmd:by(}{it:varname}{cmd:)}: when this option is specified, the command estimates all 
the effects separately by the levels of {it:varname}, a group-level and time-invariant variable.
If {it:varname} is a binary variable for example, then the estimation is carried out once for groups with {it:varname}=0 and
once for groups with {it:varname}=1. Then, the command reports on a graph event-study plots
for all values of {it:varname}, thus allowing to assess effect heterogeneity by {it:varname}. 
{p_end}

{p 4 8}
{cmd:by_path(}{it:#}{cmd:)}: when this option is specified, the command estimates all the effects separately for the
{it:#} most common treatment paths from F_g-1 to F_g-1+ℓ, where ℓ is the argument inputted to the {it:effects} option.
If you want to estimate effects separately for all treatment paths, you can input {it:all} as the option’s argument.
This option can not be combined with the {cmd:by} option.
{p_end}

{p 4 8}
{cmd:predict_het(}{it:varlist,numlist}{cmd:)}: when this option is specified, the command outputs tables 
showing whether the group-level and time-invariant variables in {it:varlist} predict groups' 
estimated event-study effects. By default, with this option the command produces one table 
per event-study effect estimated, each displaying the coefficients from regressions of the 
group-level estimate of the event-study effect on the variables in {it:varlist}. This method 
to analyze heterogeneous treatment effects assumes that switchers' counterfactual outcome 
evolutions is uncorrelated with the variables in {it:varlist}. To placebo test this condition, 
the command also shows placebo regression tables, where switchers' outcome evolutions before 
their treatment changed is regressed on the covariates. The p-value of a test that all 
coefficients are equal to zero is shown below each table. If you are interested in predicting 
all the event-study effects estimated, you can specify all as the option's second argument, 
instead of {it:numlist}. This option cannot be specified together with {cmd:normalized} or 
{cmd:controls}. See Section 1.5 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2024) 
for further details.
{p_end}

{p 4 8}
{cmd:predict_het_hc2bm}: when this option is specified with the {cmd: predict_het} option, the command computes HC2 standard errors that allow for intragroup correlation within groups defined by the variable specified in the {cmd: cluster} option. Degrees of freedom are adjusted following Bell and McCaffrey (2002) based on the variable specified in the {cmd: cluster} option. If no variable is specified in {cmd: cluster}, it will be clustered at the (g) level. 
{p_end}

{p 4 8}
{cmd:date_first_switch(}[by_baseline_treat]{it:,string}{cmd:)}:
the option reports the dates at which switchers experience their first treatment change,
and how many groups experienced a first change at each date. The reference population are switchers 
for which the last event-study effect can be estimated.  
If by_baseline_treat is specified as the first argument, separate 
tables are displayed for each level of the period-one treatment. 
Results can be printed in the Stata console specifying {it:console} in
the second argument. Alternatively, the output can be stored in an 
Excel file providing a valid file path in the second argument.
{p_end}

{p 4 8}
{cmd:same_switchers}: if this option is specified
and the user requests that at least two event-study effects be estimated,
the command will restrict the estimation
of the effects to switchers
for which all effects can be estimated,
to avoid compositional changes.
{p_end}

{p 4 8}
{cmd:same_switchers_pl}: this option can be specified when {cmd:same_switchers} is also specified. Then, the placebos are estimated 
only for switchers for which all the requested effects and placebos can be estimated. 
{p_end}

{p 4 8}
{cmd:switchers(}{it:string}{cmd:)}: one may be interested in
estimating separately the treatment effect
of switchers-in, whose treatment after
they switch is larger than their period-one treatment,
and of switchers-out, whose treatment after
they switch is lower than their period-one treatment.
In that case, one should run the command first with
the {cmd:switchers(}{it:in}{cmd:)} option,
and then with the {cmd:switchers(}{it:out}{cmd:)} option.
{p_end}

{p 4 8}
{cmd:only_never_switchers}: if this option is specified,
the command estimates the event-study effects using only
never-switchers as control units.
{p_end}

{p 4 8}
{cmd:ci_level(}{it:#}{cmd:)}: with this option, one can change the level of the confidence intervals shown in the output tables
and on the graph. The default value is 95, thus yielding 95% level confidence intervals.
{p_end}

{p 4 8}
{cmd:graphoptions(}{it:string}{cmd:)}:
one can use the {cmd:graphoptions(}{it:string}{cmd:)}
option to modify the appearance of the graph produced by the command.
Options requested have to follow the syntax of Stata {cmd:twoway_options}.
Do not use quotation marks for text passed into the arguments of {cmd:twoway_options}.
For instance, if you want the title of your graph to be "Event-study graph", you should type
{cmd:graphoptions(}title(Event-study graph){cmd:)}. This option can not be combined with the {cmd:by_path} option.
{p_end}

{p 4 8}
{cmd:graph_off}: when this option is specified,
the command does not return a graph.
{p_end}

{p 4 8}
{cmd:save_results(}{it:path}{cmd:)}: if this option is specified,
the command saves the estimators requested,
their standard error,
their 95% confidence interval,
and the number of observations used in the estimation in a separate data set,
at the location specified in {it:path}.
{p_end}

{p 4 8}
{cmd:save_sample}: if this option is specified, the command generates a
group-level variable {it:_did_sample}, tagging all groups used in the estimation.
This variable can take three non-missing values: ‘Never-switcher’ for groups whose treatment status never change.
‘Switcher-in’ for groups used as switchers-in, and ‘Switcher-out’ for groups used as
switchers out. {it:_did_sample} is missing for groups not used in the estimation. For
switchers-in or switchers-out, the command generates a (g,t) level variable 
{it:_effect}, that indicates the number of the event-study effect 
for which the cell is used in the estimation.
{p_end}

{p 4 8}
{cmd:more_granular_demeaning}: when groups' treatment can change multiple times, the standard errors reported
by default by the command may be conservative. Then, standard errors that may be less conservative when the 
sample size is large enough can be obtained by specifying this option. See 
de Chaisemartin et al. (2024) for further details.
{p_end}

{p 4 8}
{cmd:bootstrap(}{it:reps,seed}{cmd:)}: when this option is specified, bootstraped instead of analytical standard errors are reported.
The number of bootstrap replications is the option's first argument, the seed is the option's second argument. The two arguments need to
be separated by a comma. You always need to specify the comma, even if you leave either or both arguments blank.
In this case, the default values of both arguments are 50 replications and not setting a seed. If the {cmd:cluster} option is also requested,
the bootstrap is clustered at the level requested in the {cmd:cluster} option. If in the original sample, one of the effects or placebos 
requested can only be computed for a small number of switchers, it could be the case this effect or placebo cannot be computed at all in a 
bootstrap sample, because those switchers are not drawn into that bootstrap sample. This will lead the command to crash, with the following 
error message: ‘e(b) not found’. In this case, a first solution is to change the seed till you find one for which all effects and all placebos 
can be computed for all bootstrap samples. A second solution is to drop from the estimation placebos and dynamic effects that can only be 
computed for a small number of switchers.
{p_end}

{p 4 8}
{cmd:dont_drop_larger_lower}: by default, the command drops all the (g,t) cells such that at t,
group g has experienced both a strictly larger and a strictly
lower treatment than its period-one treatment.
de Chaisemartin and D'Haultfoeuille (2024) show that dropping those cells is necessary to ensure
that non-normalized event-study effects can be interpreted as effects of having been exposed to a weakly larger treatment
for ℓ periods.
The option {cmd:dont_drop_larger_lower} allows one to keep those cells.
{p_end}

{p 4 8}
{cmd:drop_if_d_miss_before_first_switch}: This option is relevant
when the treatment of some groups is missing at some time periods.
Then,
the command imputes some of those missing treatments.
Those imputations are detailed in Appendix A of de Chaisemartin et al (2024).
In designs where
groups' treatments can change at most once,
all those imputations are justified by the design.
In other designs, some of those
imputations may be liberal.
{cmd:drop_if_d_miss_before_first_switch} can be used to overrule
liberal imputations that are not innocuous
for the non-normalized event-study estimators.
See Appendix A of de Chaisemartin et al (2024) for further details.
{p_end}

{p 4 8}
{cmd:_no_updates}: this option stops 
automatic self-updates of the 
program, which are performed 
(on average) every 100 runs.
{p_end}

{marker Example}{...}
{title:Example: estimating the effect of banking deregulations on loans volume, using the data of Favara and Imbs (2015)}

{p 4 4}
To preserve space we only give one example in this
help file. See 
{browse "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5337463": Using did multiplegt dyn in Stata to Estimate Event-Study Effects in Complex Designs: Four Examples Based on Real Datasets}
for four other examples. The first example has a binary treatment that can
turn on an off. The second example has a continuous 
absorbing treatment. The third example has
a discrete multivalued treatment that can 
increase or decrease multiple times over time. The fourth
example has two, binary and absorbing treatments, 
where the second treatment always happens after
the first. 
{p_end}

{p 4 4}
The data for this example can be downloaded by running:
{p_end}

{phang2}{stata ssc install did_multiplegt_dyn}{p_end}
{phang2}{stata net get did_multiplegt_dyn}{p_end}
{phang2}{stata use favara_imbs_did_multiplegt_dyn.dta, clear}{p_end}

{p 4 4}
Estimating eight non-normalized event-study effects 
and three placebo effects of banking deregulations on loans volume:
{p_end}

{phang2}{stata did_multiplegt_dyn Dl_vloans_b county year inter_bra, effects(8) placebo(3) cluster(state_n)}{p_end}

{p 4 4}
Estimating eight normalized event-study effects 
and three placebo effects of banking deregulations on loans volume, 
restricting the estimation to switchers for which all effects can 
be estimated, and testing that effects are equal:
{p_end}

{phang2}{stata did_multiplegt_dyn Dl_vloans_b county year inter_bra, effects(8) cluster(state_n) normalized same_switchers effects_equal(all)}{p_end}

{p 4 4}
Estimating eight non-normalized event-study effects 
and three placebo effects of banking deregulations on house prices, 
separately for the four most common treatment paths:
{p_end}

{phang2}{stata did_multiplegt_dyn Dl_hpi county year inter_bra, effects(8) placebo(3) cluster(state_n) by_path(2)}{p_end}

{marker FAQ}{...}
{title:FAQ}

{p 4 4}
{it:did_multiplegt_dyn does not output exactly the same results as did_multiplegt,}
{it:is this normal?}
{p_end}

{p 4 4}
Yes, the two commands can sometimes output different results.
This is mostly due to different conventions
in the way the two commands deal with missing values.
See Appendix B of de Chaisemartin et al (2024) for further details.
{p_end}

{p 4 4}
{it: Do I have to include group and time fixed effects as controls when using the did_multiplegt_dyn package?}
{p_end}

{p 4 4}
No, you do not have to.
Group and time fixed effects are automatically controlled for.
{p_end}

{p 4 4}
{it:My group-level panel is unbalanced: some groups (e.g. counties)}
{it:are not observed in every year. Can I still use the command?}
{p_end}

{p 4 4}
You can. A frequent case of unbalancedness
is when some groups are not observed over the full duration of the panel.
For instance, your data may be a yearly county-level panel from 1990 to 2000,
where some counties appear after 1990 while some exit before 2000.
Then, the command just redefines group's period-one treatment
as their treatment at the first period when they are observed.
{p_end}

{p 4 4}
It may also be that some groups enter and exit the data multiple times.
For instance, you observe a
county in 1990, 1991, 1994, 1996, and 2000. Then,
the command may impute some of that county's missing treatments.
Those imputations are detailed in Appendix A of de Chaisemartin et al (2024).
In designs where
groups' treatments can change at most once,
all those imputations are justified by the design.
In other designs, some of those
imputations may be liberal.
{cmd:drop_if_d_miss_before_first_switch} can be used to overrule
the potentially liberal imputations
that are not innocuous for the non-normalized event-study estimators. 
See Appendix A of de Chaisemartin et al (2024) for further details.
{p_end}

{p 4 4}
Finally, it may also be the case that the data
is fully missing at one or several time periods.
For instance, you have data for 1990, 1991,
and 1993, but 1992 is missing for every group.
Then, it is important to fill the gap in the data,
as otherwise the estimation will assume that 1991 and 1993 are as far apart as 1990 and 1991.
There are two ways of doing so.
First, you can append to your data a data set identical to your 1991 data,
but with the year equal to 1992,
and the outcome missing for every observation.
This is a conservative solution,
where no first treatment change occurring between 1991
and 1993 will be used in the estimation,
which may be reasonable because the year in which the change occurred is effectively unknown.
Second, you can append to your data a data set identical to your 1993 data,
with the year equal to 1992,
and the outcome missing for every observation.
Then, treatment changes occurring between 1991
and 1993 will be used in the estimation,
assuming they all took place between 1991 and 1992.
{p_end}

{p 4 4}
{it: Related to imbalanced panels,}
{it:my outcomes (and potentially the control variables) are measured}
{it:less frequently than the treatment.}
{it:For instance, the outcome is measured every two years,}
{it:but I know the treatment of every group in every year.}
{it:How should I proceed?}
{p_end}

{p 4 4}
To fix ideas,
let us first assume
that the outcome is measured every two years,
but you know the treatment of every group in every year.
Then, you should split the sample into two subsamples,
and run the command twice,
one time on each of the subsamples.
In the first estimation,
you should include all group*time cells (g,t)
such that at t, g's treatment has never changed
since the start of the panel, and all (g,t)s such that i) g's
treatment has changed at
least once at t and ii) the change occurred at a period
where the outcome is observed.
Since the outcome is measured every two years,
in that subsample the first event-study effect (denoted effect_1)
is the effect of being exposed to a higher treatment for one period,
the second effect (effect_2)
is the effect of being exposed to a higher treatment for three periods, etc.
In the second estimation,
you should include all group*time cells (g,t)
such that at t, g's treatment has never changed
since the start of the panel, and all (g,t)s such that i) g's
treatment has changed at
least once at t and ii) the change occurred at a period
where the outcome is not observed.
In that subsample, the first event-study effect (denoted effect_1)
is the effect of being exposed to a higher treatment for two periods,
the second effect (effect_2)
is the effect of being exposed to a higher treatment for four periods, etc.
You may then combine the two sets of estimated effects
into one event-study graph, with the only caveat that
the "odd" and "even"
effects are estimated on different subsamples.
Importantly, the two estimations have to be run
on a dataset at the same bi-yearly level as the outcome
variable: the yearly level treatment information
should only be used to select the relevant subsamples.
{p_end}

{p 4 4}
If the treatment is observed three times more often than the
treatment, you can follow the same logic,
splitting the sample into three subsamples
and running the command three times, etc.
{p_end}

{p 4 4}
A short do file with a simple example where the treatment status is observed in each period while the outcome
is only observed every second period can be found {browse "https://drive.google.com/uc?export=download&id=1NBwfsFeNltU3XSOsORdthUW49LIezm1z":here}. 
{p_end}

{p 4 4}
{it:What is the maximum number of event-study effects I can estimate?}
{p_end}

{p 4 4}
With a balanced panel of groups,
the maximum number of event-study effects one can estimate
can be determined as follows.
For each value of the period-one treatment d,
start by computing the difference between the last period
at which at least one group has had treatment d since period 1,
and the first period at which a group with treatment d at period 1
changed its treatment.
Add one to this difference.
Then, the maximum number of event-study effects is equal to
the maximum of the obtained values,
across all values of the period-one treatment.
With an unbalanced panel,
this method can still be used to derive an upper bound
of the maximum number of event-study effects one can estimate.
{p_end}

{p 4 4}
{it:How many control variables can I include in the estimation?}
{p_end}

{p 4 4}
Estimators with control variables are similar to those without controls,
except that the first-difference of the outcome
is replaced by residuals from regressions
of the first-difference of the outcome on
the first-differences of the controls and time fixed effects.
Those regressions are estimated in the sample of control (g,t)s:
(g,t)s such that group g's treatment has not changed yet at period t.
Those regressions are also estimated separately
for each value of the period-one treatment.
If at period one, treatment takes values 0, 1, 2, 3, and 4,
one regression is estimated
for control (g,t)s with a period-one treatment equal to 0,
one regression is estimated for control
(g,t)s with a period-one treatment equal to 1, etc.
The number of control variables
needs to be significantly smaller than
the number of control (g,t)s in each of those regressions.
Otherwise, those regressions will overfit and produce noisy estimates.
If the number of observations is lower than the number
of variables in one of those regressions,
the command will run but will not take into account all
the controls for all values of the period-one treatment.
An error message will let the user know that
they are encountering this situation, and may thus want to reduce their
number of control variables.
{p_end}

{p 4 4}
{it:My design is such that treatment is binary, and groups can enter the treatment, and then leave it once.}
{it:Can I use the command to separately estimate the effect of joining and leaving the treatment?}
{p_end}

{p 4 4}
Yes you can. See Section 1.6 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2024) for further details.
{p_end}

{p 4 4}
{it:My design has several treatments.}
{it:Can I use the command to estimate the event-study effects of a treatment controlling for other treatments?}
{p_end}

{p 4 4}
Yes.  See Section 3.2 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2023) for 
further details, keeping in mind that the {cmd:did_multiplegt_dyn} command referenced at the time is now superseded by this command. 
{p_end}

{p 4 4}
{it:Can I perform triple difference-in-differences with the command?}
{p_end}

{p 4 4}
Yes. Suppose for instance your third difference is across men
and women in the same (g,t) cell. Then, 
for each (g,t) cell, you just need to compute the difference between the average 
outcome of men and women in cell (g,t).
Then, you simply run the command with this new outcome.
{p_end}

{p 4 4}
{it:Is it possible to compute switchers' average counterfactual outcome at periods F_g, F_g+1 etc,} 
{it:so as to then express event-study effects in percentage points of the counterfactual outcome level?}
{p_end}

{p 4 4}
Yes. You just need to define a new outcome variable 
Y' = - Y * 1{t < F_g},
where F_g is the first date at which g's treatment has changed.
Essentially, you replace the outcome by 0 after the treatment change, 
and by -Y before the treatment change.
Then, you compute non-normalized event-study 
estimators with Y' as the outcome.
{p_end}

{p 4 4}
{it: Can the command be used in fuzzy designs, where the treatment varies within group*time cells?}
{p_end}

{p 4 4}
Yes, it can, see Section 1.7 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2024) for further details.
{p_end}

{p 4 4}
{it:Instead of using an F-test to jointly test that all placebos or all effects are zero, I would like to use a sup t-test. Is this possible?}
{p_end}

{p 4 4}
Yes, {cmd:did_multiplegt_dyn} is compatible with
the {cmd:sotable} package. Here's how to produce 
sup t-tests on all placebos and effects
in post estimation:
{p_end}

{phang2}{stata net get did_multiplegt_dyn}{p_end}
{phang2}{stata use favara_imbs_did_multiplegt_dyn.dta, clear}{p_end}
{phang2}{stata did_multiplegt_dyn Dl_vloans_b county year inter_bra, effects(8) placebo(3) cluster(state_n) graph_off}{p_end}
{phang2}{stata sotable, pnames(`=e(placebo)') normal}{p_end}
{phang2}{stata sotable, pnames(`=e(effects)') normal}{p_end}

{title:References}

{p 4 8}
Bell, R. M., McCaffrey, D. F. (2002). 
{browse "https://www150.statcan.gc.ca/n1/pub/12-001-x/2002002/article/9058-eng.pdf": Bias reduction in standard errors for linear regression with multi-stage samples.} 
Survey Methodology.
{p_end}
{p 4 8}
de Chaisemartin, C, D'Haultfoeuille, X (2024).
{browse "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3731856":Difference-in-Differences Estimators of Intertemporal Treatment Effects}. 
Forthcoming, Review of Economics and Statistics.
{p_end}
{p 4 8}
de Chaisemartin, C, D'Haultfoeuille, X (2023).
{browse "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3751060":Two-way fixed effects regressions with several treatments}. Journal of Econometrics.
{p_end}
{p 4 8}
de Chaisemartin, C, Ciccia, D, D'Haultfoeuille, X, Knau, F, Malézieux, M, Sow, D (2024).
{browse "https://drive.google.com/file/d/1NGgScujLCCS4RrwdN-PC1SnVigfBa32h/view?usp=drive_link": Event-Study Estimators and Variance Estimators Computed by the did_multiplegt_dyn Command}.
{p_end}

{title:Auxiliary packages}

{p 4 4}
The command requires that the gtools package be installed on the user's machine.
{p_end}

{title:Authors}

{p 4 4}
Clément de Chaisemartin, Economics Department, Sciences Po, France.
{p_end}
{p 4 4}
Diego Ciccia, Northwestern University, USA.
{p_end}
{p 4 4}
Xavier D'Haultfoeuille, CREST-ENSAE, France.
{p_end}
{p 4 4}
Felix Knau, LMU Munich, Germany.
{p_end}
{p 4 4}
Mélitine Malézieux, Stockholm School of Economics, Sweden.
{p_end}
{p 4 4}
Doulo Sow, Princeton University, USA.
{p_end}

{title:Contact}

{p 4 4}
Mail:
{browse "mailto:chaisemartin.packages@gmail.com":chaisemartin.packages@gmail.com}
{p_end}

{p 4 4}
GitHub:
{browse "https://github.com/chaisemartinPackages/did_multiplegt_dyn"}.
{p_end}

{marker saved_results}{...}
{title:Saved results}

{p 4 8}
{cmd:{ul:Matrix}:}
{p_end}

{p 8 10}
{cmd:e(estimates)}: Column vector storing the estimated event-study and placebo effects.
{p_end}

{p 8 10}
{cmd:e(variances)}: Vector storing the corresponding variance estimates
{p_end}

{p 8 10}
{cmd:e(b)}: Row vector storing the estimated event-study and placebo effects
{p_end}

{p 8 10}
{cmd:e(V)}: Variance/Covariance Matrix of the estimated event-study and placebo effects
{p_end}

{p 8 10}
{cmd:e(effect_het_ℓ_XX)}: Matrix storing the outputs from the {cmd:predict_het} option
for effect i.
{p_end}

{p 4 8}
{cmd:{ul:Macro}:}
{p_end}

{p 8 10}
{cmd:e(cmd)}: macro equal to "did_multiplegt_dyn", the name of the command.
{p_end}

{p 4 8}
{cmd:{ul:Scalar}:}
{p_end}

{p 8 10}
{cmd:e(Effect_ℓ)}: estimated event-study effect ℓ.
{p_end}

{p 8 10}
{cmd:e(N_effect_ℓ)}: number of observations used in the estimation of {cmd:e(Effect_ℓ)}.
{p_end}

{p 8 10}
{cmd:e(N_switchers_effect_ℓ)}: number of switchers {cmd:e(Effect_ℓ)} applies to.
{p_end}

{p 8 10}
{cmd:e(se_effect_ℓ)}: estimated standard error of {cmd:e(Effect_ℓ)}.
{p_end}

{p 8 10}
{cmd:e(p_jointeffects)}: p-value of a joint test that all effects are equal to 0, if two or more estimators were requested.
{p_end}

{p 8 10}
{cmd:e(p_equality_effects)}: p-value of a joint test
that all effects are equal, when the option {cmd:effects_equal} is specified.
{p_end}

{p 8 10}
{cmd:e(Placebo_ℓ)}: estimated placebo ℓ.
{p_end}

{p 8 8}
{cmd:e(N_placebo_ℓ)}: number of observations used
in the estimation of {cmd:e(Placebo_ℓ)}.
{p_end}

{p 8 10}
{cmd:e(N_switchers_placebo_ℓ)}: number of switchers {cmd:e(Placebo_ℓ)} applies to.
{p_end}

{p 8 10}
{cmd:e(se_placebo_ℓ)}: estimated standard error of {cmd:e(Placebo_ℓ)}.
{p_end}

{p 8 10}
{cmd:e(p_jointplacebo)}: p-value of the joint test that all the placebos are equal to 0,
if two or more placebo estimators were requested.
{p_end}

{p 8 10}
{cmd:e(Av_tot_effect)}: estimated average cumulative (total)
effect per unit of treatment, where “cumulative effect”
refers to the sum of the effects of a treatment
increment, at the time when it takes place and at later periods.
{p_end}

{p 8 10}
{cmd:e(N_avg_total_effect)}: number of observations used
in the estimation of {cmd:e(Av_tot_effect)}.
{p_end}

{p 8 10}
{cmd:e(N_switchers_effect_average)}: number of switchers*periods
{cmd:e(effect_average)} applies to.
{p_end}

{p 8 10}
{cmd:e(se_avg_total_effect)}: estimated standard error of {cmd:e(effect_average)}.
{p_end}
