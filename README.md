🔊🔊 **Check out [this](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5337463) new document with examples on how to implement the did_multiplegt_dyn on various, complex designs!** We provide detailed guidelines on how to address designs with (1) a binary treatment that can turn on an off, (2) a continuous absorbing treatment, (3) a discrete multivalued treatment that can increase or decrease multiple times over time, and (4) two, binary and absorbing treatments, where the second treatment always happens after the first. You can find the code and data for replicating this examples in the "Examples did_multiplegt_dyn on various designs" folder. 🔊🔊



# did_multiplegt_dyn
Estimation in Difference-in-Difference (DID) designs with multiple groups and periods.

[Short description](#Short-description) | [Vignettes](#vignettes) | [Setup](#Setup) | [Syntax](#Syntax) | [Description](#Description)

[Options](#Options) | [Example](#Example) | [FAQ](#FAQ) | [References](#References) | [Authors](#Authors)

## Short description

Estimation of event-study Difference-in-Difference (DID) estimators in designs with multiple groups and periods, and with a potentially non-binary treatment that may increase or decrease multiple times.  

## Vignettes

+ [did_multiplegt_dyn with periodically missing outcomes: a tutorial with toy electoral data (Stata, R)](https://github.com/chaisemartinPackages/did_multiplegt_dyn/blob/main/vignettes/vignette_1.md)
+ [did_multiplegt_dyn and esttab (Stata)](https://github.com/chaisemartinPackages/did_multiplegt_dyn/blob/main/vignettes/vignette_2.md)
+ [Retrieving all feasible placebos from did_multiplegt_dyn (Stata, R)](https://github.com/chaisemartinPackages/did_multiplegt_dyn/blob/main/vignettes/vignette_3.md)

## Setup

### Stata
```s
ssc install did_multiplegt_dyn, replace
```

### R

#### CRAN

```s
install.packages("DIDmultiplegtDYN")
```
#### Github (Development version)

```r
library(devtools)
devtools::install_github("chaisemartinPackages/did_multiplegt_dyn/R")
```

#### R (Recent Macs)

Some users might encounter an issue with the R version of the package due to dependencies incompatibility. Switching off this dependency (rgl) usually solves this issue. You can either run the following line of code before loading the package or add the line to your .Rprofile.

```r
Sys.setenv(RGL_USE_NULL = TRUE)
```

## Syntax

### Stata

**did_multiplegt_dyn Y G T D** [if] [in] [, **effects**(#) **design**(*float*, *string*) **normalized normalized_weights effects_equal placebo**(#) **controls**(*varlist*)  **trends_nonparam**(*varlist*) **trends_lin continuous**(#) **weight**(*varname*) **cluster**(*varname*) **by**(*varname*) **predict_het**( *varlist,numlist*) **date_first_switch**( [*by_baseline_treat*],*string*) **same_switchers same_switchers_pl** **switchers**(*string*) **ci_level**(#) **graphoptions**(*string*) **graph_off** **save_results**(*path*) **save_sample** **less_conservative_se** **drop_larger_lower drop_if_d_miss_before_first_switch**]

### R
**did_multiplegt_dyn** <- function(**df** = *dataframe*, **outcome** = *string*, **group** = *string*, **time** = *string*, **treatment** = *string*, **effects** = 1, **design** = NULL, **normalized** = FALSE, **normalized_weights** = FALSE, **effects_equal** = FALSE, **placebo** = 0, **controls** = NULL, **trends_nonparam** = NULL, **trends_lin** = FALSE, **continuous** = NULL, **weight** = NULL, **cluster** = NULL, **by** = NULL, **predict_het** = NULL, **date_first_switch** = NULL, **same_switchers** = FALSE, **same_switchers_pl** = FALSE, **switchers** = "", **ci_level** = 95, **graph_off** = FALSE, **save_results** = NULL, **save_sample** = FALSE, **less_conservative_se** = FALSE, **dont_drop_larger_lower** = FALSE, **drop_if_d_miss_before_first_switch** = FALSE)


## Description

**did_multiplegt_dyn** estimates the effect of a treatment on an outcome, using group-(e.g. county- or state-) level panel data. The command computes the DID event-study estimators introduced in de Chaisemartin and D'Haultfoeuille (2024). Like other recently proposed DID estimation commands (**csdid**, **didimputation**,...), **did_multiplegt_dyn** can be used with a binary and staggered (absorbing) treatment. But unlike those other commands, **did_multiplegt_dyn** can also be used with a non-binary treatment (discrete or continuous) that can increase or decrease multiple times. Lagged treatments may affect the outcome, and the current and lagged treatments may have heterogeneous effects, across space and/or over time.  The event-study estimators computed by the command rely on a no-anticipation and parallel trends assumptions. The panel may be unbalanced:  not all groups have to be observed at every period.  The data may also be at a more disaggregated level than the group level (e.g. individual-level wage data to measure the effect of a regional-level minimum-wage on individuals' wages).

For all "switchers", namely groups that experience a change of their treatment over the study period, let $F_g$ denote the first time period when g's treatment changes. The command computes the non-normalized event-study estimators $DID_\ell$. $DID_1$ is the average, across all switchers, of DID estimators comparing the $F_{g-1}$ to $F_g$ outcome evolution of $g$ to that of groups with the same period-one treatment as $g$ but whose treatment has not changed yet at $F_g$. More generally, $DID_1$ is the average, across all switchers, of DID estimators comparing the $F_{g-1}$ to $F_{g-1+\ell}$ outcome evolution of $g$ to that of groups with the same period-one treatment as $g$ but whose treatment has not changed yet at $F_{g-1+\ell}$. Non-normalized event-study effects are average effects of having been exposed to a weakly higher treatment dose for $\ell$ periods, where the magnitude and timing of the incremental treatment doses can vary across groups. The command also computes the normalized event-study estimators $DID^n_\ell$, that normalize $DID_\ell$ by the average total incremental treatment dose received by switchers from $F_{g-1}$ to $F_{g-1+\ell}$ with respect to their period-one treatment. This normalization ensures that $DID^n_\ell$ estimates a weighted average of the effects of the current treatment and of its $\ell-1$ first lags on the outcome. The command also computes an estimated average total effect per unit of treatment, where "total effect" refers to the sum of the effects of a treatment increment, at the time when it takes place and at later periods, see Section 3.3 of de Chaisemartin and D'Haultfoeuille (2024) for further details. Finally, the command also computes placebo estimators, that average DIDs comparing the outcome evolution of switcher $g$ and of its control groups, from $F_{g-1}$ to $F_{g-1-\ell}$, namely before g's treatment changes for the first time. Those placebos can be used to test the parallel trends and no-anticipation assumptions under which the estimators computed by **did_multiplegt_dyn** are unbiased.

**Y** (R: **outcome**) is the outcome variable.

**G** (R: **group**) is the group variable.

**T** (R: **time**) is the time period variable.  The command assumes that the time variable is evenly spaced
    (e.g.: the panel is at the yearly level, and no year is missing for all groups).  When it is not
    (e.g.: the panel is at the yearly level, but three consecutive years are missing for all groups),
    the command can still be used, though it requires a bit of tweaking, see FAQ section below.

**D** (R: **treatment**) is the treatment variable.
    

## Options   
**effects(**#**)**: gives the number of event-study effects to be estimated. By default, the command estimates non-normalized event-study effects. Non-normalized event-study effects are averages, across all switchers, of the effect of having received their actual rather than their period-one treatment dose, for $\ell$ periods. While non-normalized event-study effects can be interpreted as average effects of being exposed to a weakly higher treatment dose for $\ell$ periods, the magnitude and timing of the incremental treatment doses can vary across groups.

**design(***float*, *path***)**: (R, list with 2 args: float, char path) this option reports switchers' period-one and subsequent treatments, thus helping the analyst understand the treatment paths whose effect is aggregated in the non-normalized event-study effects. When the number of treatment paths is low, one may consider estimating treatment-path-specific event-study effects to facilitate interpretation, see footnote 10 of de Chaisemartin and D'Haultfoeuille (2024) for detailed instructions. When the number of treatment paths is large, one may specify a number included between 0 and 1 in the float argument. Then the command reports the treatment paths common to at least (*float* $\times$ 100)% of switchers. Results can be printed in the Stata console specifying "console" as the string argument.  For example, *design(0.5, "console")* reports the treatment paths experienced by at least 50% of the switchers and prints the output in the Stata console. Alternatively, the output can be stored in an Excel file providing a valid file path as the string argument.

**normalized**: when this option is specified, the command estimates normalized event-study effects, that are equal to a weighted average of the effects of the current treatment and of its $\ell-1$ first lags on the outcome. See Sections 3.1 and 3.2 of de Chaisemartin and D'Haultfoeuille (2020a) for further details.

**normalized_weights**: the command reports the weights that normalized effect $\ell$ puts on the effect of the current treatment, on the effect of the first treatment lag, etc.

**effects_equal**: when this option is specified and the user requests that at least two effects be estimated, the command performs an F-test that all effects are equal. When the **normalized** option is specified, this test can be useful to assess if the current and lagged treatments all have the same effect on the outcome or if their effects differ, see Lemma 3 of de Chaisemartin and D'Haultfoeuille (2020a).

**placebo(**#**)**: gives the number of placebo estimators to be computed. Placebos compare the outcome evolution of switchers and of their controls, before switchers' treatment changes for the first time. Under the parallel trends and no-anticipation assumptions underlying the event-study estimators computed by did_multiplegt_dyn, the expectation of the placebos is equal to zero. Thus, placebos can be used to test those assumptions, by testing the null that all placebos are equal to zero. If the user requests that at least two placebos be estimated, the command computes the p-value of a joint test of that null hypothesis. The number of placebos requested can be at most equal to the number of time periods in the data minus 2, though most often only a smaller number of placebos can be computed. Also, the number of placebos requested cannot be larger than the number of effects requested.

**controls(***varlist***)**: gives the names of the control variables to be included in the estimation. Estimators with controls are similar to those without controls, except that the first-difference of the outcome is replaced by residuals from regressions of the first-difference of the outcome on the first-differences of the controls and time fixed effects. Those regressions are estimated in the sample of control $(g,t)$s: $(g,t)$s such that group $g$'s treatment has not changed yet at $t$. Those regressions are also estimated separately for each value of the baseline treatment. Estimators with controls are unbiased even if groups experience differential trends, provided such differential trends can be fully explained by a linear model in covariates changes. To control for time-invariant covariates, one needs to interact them with the time variable **T**, or with time fixed effects. See Section 1.2 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2020a) for further details.

**trends_lin**: when this option is specified, the estimation of the treatment effects allows for group-specific linear trends. Estimators with linear trends start by computing event-study effects on the outcome's first-difference, rather than on the outcome itself, thus allowing for group-specific linear trends. Then, to recover event-study effect $\ell$ on the outcome, event-study effects on the outcome's first-difference are summed from 1 to $\ell$. See Section 1.3 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2024) for further details. When this option is specified, the estimated average total effect per unit of treatment is not computed.

**trends_nonparam(***varlist***)**: when this option is specified, the DID estimators computed by the command only compare switchers to controls whose treatment has not changed yet, with the same baseline treatment, and with the same value of the varlist. Estimators with the **trends_nonparam** option are unbiased even if groups experience differential trends, provided all groups with the same value of the varlist experience parallel trends. The vector can only include time-invariant variables, and the interaction of those variables has to be coarser than the group variable. For instance, if one works with a county $\times$ year data set and one wants to allow for state-specific trends, then one should write *trends_nonparam(state)*, where state is the state identifier. See Section 1.4 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2024) for further details.

**continuous(**#**)**: allows to use the command even when groups' period-one treatment is continuous, meaning that all groups have a different period-one treatment value. With a discrete period-one treatment, the command compares the outcome evolution of switchers and non-switchers with the same period-one treatment.  But with a truly continuous period-one treatment, there will be no two groups with the same period-one treatment. The command assumes that group's status-quo outcome evolution is a polynomial in their period-one treatment. The user's chosen polynomial order is the option's argument. See Section 1.10 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2024) for further details. Unlike the other variance estimators computed by the command, those computed when the continuous option is specified are not backed by a proven asymptotic normality result. Preliminary simulation evidence indicates that when the option is used with a correctly-specified polynomial order, those variance estimators are conservative. On the other hand, when the specified polynomial order is strictly larger than needed, those variance estimators can become liberal. Thus, when this option is specified, we recommend using the bootstrap for inference, by manually bootstrapping the command. At least, one should perform a robustness check where one compares the analytic variance computed by the command to a bootstrapped variance.

**weight(***varlist***)**: gives the name of a variable to be used to weight the data. For instance, if one works with a district $\times$ year data set and one wants to weight the estimation by each district $\times$ year's population, one should write weight(population), where population is the population of each district $\times$ year. If the data set is at a more disaggregated level than group $\times$ time, the command aggregates it at the group $\times$ time level internally, and weights the estimation by the number of observations in each group $\times$ time cell if the weight option is not specified, or by the sum of the weights of the observations in each group $\times$ time cell if the weight option is specified.

**cluster(***varname***)**: can be used to cluster the estimators' standard errors. Only one clustering variable is allowed. A common practice in DID analysis is to cluster standard errors at the group level. Such clustering is implemented by default by the command. Standard errors can be clustered at a more aggregated level than the group level, but they cannot be clustered at a more disaggregated level.

**by(***varname***)**: when this option is specified, the command estimates all the effects by the different levels of the specified variable. If the variable is a binary variable for example, then the estimation is carried out once for the sample of groups with var=0 and once for the sample of groups with var=1. Then, the command reports on a graph event-study plots for all values of the by() argument, thus allowing to assess effect heterogeneity by the specified variable. 

**by_path(***#***)**:  when this option is specified, the command estimates all the effects separately for the # most common treatment paths from $F_{g-1}$ to $F_{g-1+\ell}$, where $\ell$ is the argument inputted to the **effects** option. If you want to estimate effects separately for all treatment paths, you can input -1 as the option’s argument. This option can not be combined with the **by** option.

**predict_het(***varlist*, *numlist***)**: (R, list with 2 args: char or vector of char, -1 or vector of positive integers)  when this option is specified, the command outputs tables showing whether the group-level and time-invariant variables in the char varlist predict groups' estimated event-study effects. With the second argument is void (R: set to -1), with this option the command produces one table per event-study effect estimated, each displaying the coefficients from regressions of the group-level estimate of the event-study effect on the variables in the char varlist. The p-value of a test on the null hypothesis that all heterogeneity estimates are equal to zero is shown below each table. If you are only interested in predicting a subset of the event-study effects estimated, you can specify those inside an integer vector as the second argument. This option cannot be specified with **normalized** and when the controls option is specified. See Section 1.5 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2024) for further details.

**date_first_switch(**\[*by_baseline_treat*\], *path***)**: (R, list with 2 args: char in ("", "by_baseline_treat"), char path) the option reports the dates at which switchers experience their first treatment change, and how many groups experienced a first change at each date. The reference population are switchers for which the last event-study effect can be estimated. If "by_baseline_treat" is specified as the first argument, separate tables are displayed for each level of the period-one treatment. Results can be printed in the Stata console specifying "console" in the second argument. Alternatively, the output can be stored in an Excel file providing a valid file path in the second argument.

**same_switchers**: if this option is specified and the user requests that at least two effects be estimated, the command will restrict the estimation of the event-study effects to switchers for which all effects can be estimated, to avoid compositional changes.

**same_switchers_pl**: the command restricts the estimation of event-study and placebo effects to switchers for which all event-study and placebos effects can be estimated. 

**switchers(***string***)**: one may be interested in estimating separately the treatment effect of switchers-in, whose average treatment after they switch is larger than their baseline treatment, and of switchers-out, whose average treatment after they switch is lower than their baseline treatment. In that case, one should run the command first with the switchers = "in" option, and then with the switchers = "out" option.

**only_never_switchers**: if this option is specified, the command estimates the event-study effects using only never-switchers as control units.

**ci_level**: with this option you can change the level of the confidence intervals displayed in the output tables and the graphs. The default value is fixed at 95, yielding a 95% coverage.

**graphoptions(***string***)**:  one can use the **graphoptions**(*string*) option to modify the appearance of the graph produced by the command.  Options requested have to follow the syntax of Stata **twoway_options**.  Do not use quotation marks for text passed into the arguments of **twoway_options**.  For instance, if you want the title of your graph to be "Graph to convince skeptical referee", you should type **graphoptions**(*title(Graph to convince skeptical referee)*). This option is not available in R, since one could just modify the did_multiplegt_dyn ggplot object in the global envirnoment using the ggplot library.

**graph_off**: when this option is specified, the command does not print a graph. Regardless, a ggplot object will be still generated and stored in the did_multiplegt_dyn class object.

**save_results**: if this option is specified, the command saves the estimators requested, their standard error, their 95% confidence interval, and the number of observations used in the estimation in a separate data set, at the location specified in the char argument.

**save_sample**: if this option is specified, the command generates a (numeric) variable *did_sample*, tagging all $(g,t)$ cells used in the estimation. This variable may take on three non-missing values: 0 for $(g,t)$ cells used as controls, 1 for $(g,t)$ cells used as switchers-in, and -1 for cells used as switchers-out. This variable is missing for all $(g,t)$ cells not used in the estimation. This option also generates a *did_effect* variable that indicates the number of the event-study effect for which the cell is used in the estimation.

**less_conservative_se**: when groups' treatment can change multiple times, the standard errors reported by default by the command may be conservative. Then, less conservative standard errors can be obtained by specifying this option. See de Chaisemartin et al. (2024) for further details.

**bootstrap(***reps*, *seed***)**: when this option is specified, bootstraped instead of analytical standard errors are reported.  The number of bootstrap replications is the option's first argument, the *seed* is the option's second argument. The two arguments need to be separated by a comma. You always need to specify the comma, even if you leave either or both arguments blank.  In this case, the default values of both arguments are 50 replications and not setting a seed. If the **cluster** option is also requested, the bootstrap is clustered at the level requested in the cluster option.

**dont_drop_larger_lower**: by default, the command drops all the $(g,t)$ cells such that at $t$, group $g$ has experienced both a strictly larger and a strictly lower treatment than its baseline treatment. de Chaisemartin and D'Haultfoeuille (2020a) recommend this procedure, if you are interested in more details you can see their Section 3.1. The option **dont_drop_larger_lower** allows to overwrite this procedure and keeps $(g,t)$ cells such that at $t$, group $g$ has experienced both a strictly larger and a strictly lower treatment than its baseline treatment in the estimation sample.

**drop_if_d_miss_before_first_switch**: This option is relevant when the treatment of some groups is missing at some time periods. Then, the command imputes some of those missing treatments. Those imputations are detailed in Appendix A of de Chaisemartin et al (2024). In designs where groups' treatments can change at most once, all those imputations are justified by the design. In other designs, some of those imputations may be liberal. drop_if_d_miss_before_first_switch can be used to overrule the potentially liberal imputations that are not innocuous for the non-normalized event-study estimators. See Appendix A of de Chaisemartin et al (2024) for further details.

**<ins>Options compatibility and interaction:</ins>** 
Here are some highlights that one should be aware of when combining some options in the command:

 **i.** The option **by**(*varname*) and the option **predict_het**(*varname*) are not compatible unless they
    receive different inputs (varname). In such case (i.e, two different inputs), the command carries
    out the heterogeneity prediction, according to the variable specified in **predict_het**(),
    conditional on the different values taken by the variable specified in **by**().

 **ii.** If the option **by**() is specified, and ones requests the data to be saved using the option
    **save**(), the command will save the estimation results as usual except that the names of the
    columns are indexed by the level of the variable inputed in **by**().  E.g., if the variable (let's
    call it by_var) has 4 levels:  in the saved dataset, one will have point_estimate1 (for
    by_var==1), point_estimate2 (for by_var==2) etc.  as the estimates of effects estimated
    conditional on the sample such that by_var==1, by_var==2, etc. respectively.

**iii.** Option **by**() and Option **design**(): If one requests the design to be displayed in the console,
    the command displays succesively the design for each level of the variable inputed in **by**().
    Otherwise, if one requests the design to be stored in an Excel file, the command stores each
    design in a specific sheet.  Exactly the same reasoning applies when specifying the **by**() option
    together with the **date_first_switch**() option.

**iv.** The option **normalized** should not be specified if one wants to use **predict_het**().  For more
    details see Lemma 6 of [de Chaisemartin, C, D'Haultfoeuille, X (2020a)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3731856).
    
**<ins>Technical note on R output:</ins>** 

The standard output of did_multiplegt_dyn in R is a list of objects with *did_multiplegt_dyn* class. This allows for customized *print* and *summary* method dispatch. The basic output list of the program includes: (i) a list of the command arguments (*args*), (ii) a list with all the results from the estimation (*results*), (iii) a ggplot object for the event-study graph (*plot*). Additional options enrich the output list and normally add up to other items in the *results* sublist. You can inspect recursively the content of the output by assigning the did_multiplegt_dyn to an *object*, running names(*object*) and then running *object*$*name* for each name in the names list.

When the command is run with the **by** option, the output list is reshaped to include the results from all subsets of the original data having specific levels of the *by* variable. The new output list will still include *args* and *plot*, even though the ggplot object is the combination of the all the graphs from the subsamples. Moreover, the output list will include a *by_levels* list with all the values of the by option and $N$ *by_level_*$i$ lists, with $N$ number of levels of the *by* variable and $1 \leq i \leq N$. Each of these lists will include *results* and *plot* lists based on their respective subsample.


## Example

This example is estimating the effect of banking deregulations on loans volume, using the data of Favara and Imbs (2015)

### Stata

```applescript
ssc install did_multiplegt_dyn
net get did_multiplegt_dyn
use favara_imbs_did_multiplegt_dyn.dta, clear
```

Estimating eight non-normalized event-study effects and three placebo effects of banking
deregulations on loans volume:

```applescript
did_multiplegt_dyn Dl_vloans_b county year inter_bra, effects(8) placebo(3) cluster(state_n)
```

Estimating eight normalized event-study effects and three placebo effects of banking
deregulations on loans volume, restricting the estimation to switchers for which all effects can be estimated, and testing that effects are equal:

```applescript
 did_multiplegt_dyn Dl_vloans_b county year inter_bra, effects(8) cluster(state_n) normalized same_switchers effects_equal
```

### R

Same steps and data as above.

```applescript
library(polars)
library(DIDmultiplegtDYN)
data(favara_imbs)
```

```applescript
summary(did_multiplegt_dyn(
    df = favara_imbs,
    outcome = "Dl_vloans_b",
    group = "county",
    time = "year",
    treatment = "inter_bra",
    effects = 8,
    placebo = 3,
    cluster = "state_n",
    graph_off = TRUE
))
```

```applescript
summary(did_multiplegt_dyn(
    df = favara_imbs,
    outcome = "Dl_vloans_b",
    group = "county",
    time = "year",
    treatment = "inter_bra",
    effects = 8,
    placebo = 3,
    cluster = "state_n",
    normalized = TRUE,
    same_switchers = TRUE,
    effects_equal = TRUE
))
```
Please note that some of the standard errors displayed above could differ from those reported in de Chaisemartin and D'Haultfoeuille (2020b) due to coverage-improving changes to the variance estimator.

## FAQ

 > :question: *did_multiplegt_dyn does not output exactly the same results as did_multiplegt, is this normal?*

Yes, the two commands can sometimes output different results.  This is mostly due to different
    conventions in the way the two commands deal with missing values.  See Appendix B of de Chaisemartin et al
    (2024) for further details.

> :question: *Do I have to include group and time fixed effects as controls when using the did_multiplegt_dyn
    package?*

No, you do not have to.  Group and time fixed effects are automatically controlled for.

> :question: *My group-level panel is unbalanced: some groups (e.g. counties) are not observed in every year.
    Can I still use the command?*

You can. A frequent case of unbalancedness is when some groups are not observed over the full
    duration of the panel.  For instance, your data may be a yearly county-level panel from 1990 to
    2000, where some counties appear after 1990 while some exit before 2000.  Then, the command just
    redefines group's baseline treatment as their treatment at the first period when they are
    observed.

It may also be that some groups enter and exit the data multiple times.  For instance, you
    observe a county in 1990, 1991, 1994, 1996, and 2000. Then, the command may impute some of that
    county's missing treatments.  Those imputations are detailed in de Chaisemartin et al (2023a).
    In designs where groups' treatments can change at most once, all those imputations are justified
    by the design.  In other designs, some of those imputations may be liberal.
    **drop_if_d_miss_before_first_switch** can be used to overrule the potentially liberal imputations
    that are not innocuous for the non-normalized event-study estimators.  See de Chaisemartin et al
    (2023a) for further details.

Finally, it may also be the case that the data is fully missing at one or several time periods.
    For instance, you have data for 1990, 1991, and 1993, but 1992 is missing for every group.  Then,
    it is important to fill the gap in the data, as otherwise the estimation will assume that 1991
    and 1993 are as far apart as 1990 and 1991.  There are two ways of doing so.  First, you can
    append to your data a data set identical to your 1991 data, but with the year equal to 1992, and
    the outcome missing for every observation.  This is a conservative solution, where no first
    treatment change occurring between 1991 and 1993 will be used in the estimation, which may be
    reasonable because the year in which the change occurred is effectively unknown.  Second, you can
    append to your data a data set identical to your 1993 data, with the year equal to 1992, and the
    outcome missing for every observation.  Then, treatment changes occurring between 1991 and 1993
    will be used in the estimation, assuming they all took place between 1991 and 1992.

> :question: *Related to imbalanced panels, my outcomes (and potentially the control variables) are measured
    less frequently than the treatment.  For instance, the outcome is measured every two years, but I
    know the treatment of every group in every year.  How should I proceed?*

To fix ideas, let us first assume that the outcome is measured every two years, but you know the treatment of every group in every year. Then, you should split the sample into two subsamples, and run the command twice, one time on each of the subsamples. In the first estimation, you should include all group $\times$ time cells $(g,t)$ such that at $t$, $g$'s treatment has never changed since the start of the panel, and all $(g,t)$s such that (i) $g$'s treatment has changed at least once at $t$ and (ii) the change occurred at a period where the outcome is observed. Since the outcome is measured every two years, in that subsample the first event-study effect (denoted effect_1) is the effect of being exposed to a higher treatment for one period, the second effect (effect_2) is the effect of being exposed to a higher treatment for three periods, etc. In the second estimation, you should include all group $\times$ time cells $(g,t)$ such that at $t$, $g$'s treatment has never changed since the start of the panel, and all $(g,t)$s such that (i) $g$'s treatment has changed at least once at $t$ and (ii) the change occurred at a period where the outcome is not observed. In that subsample, the first event-study effect (denoted effect_1) is the effect of being exposed to a higher treatment for two periods, the second effect (effect_2) is the effect of being exposed to a higher treatment for four periods, etc. You may then combine the two sets of estimated effects into one event-study graph, with the only caveat that the "odd" and "even" effects are estimated on different subsamples. Importantly, the two estimations have to be run on a dataset at the same bi-yearly level as the outcome variable: the yearly level treatment information should only be used to select the relevant subsamples.

If the treatment is observed three times more often than the treatment, you can follow the same
    logic, splitting the sample into three subsamples and running the command three times, etc.

A short do file with a simple example where the treatment status is observed in each period while
    the outcome is only observed every second period can be found [here](https://drive.google.com/uc?export=download&id=1NBwfsFeNltU3XSOsORdthUW49LIezm1z).

> :question: *What is the maximum number of event-study effects I can estimate?*

With a balanced panel of groups, the maximum number of event-study effects one can estimate can be determined as follows. For each value of the period-one treatment $d$, start by computing the difference between the last period at which at least one group has had treatment $d$ since period 1, and the first period at which a group with treatment $d$ at period 1 changed its treatment. Add one to this difference. Then, the maximum number of event-study effects is equal to the maximum of the obtained values, across all values of the period-one treatment. With an unbalanced panel, this method can still be used to derive an upper bound of the maximum number of event-study effects one can estimate.

> :question: *How many control variables can I include in the estimation?*

Estimators with control variables are similar to those without controls, except that the
    first-difference of the outcome is replaced by residuals from regressions of the first-difference
    of the outcome on the first-differences of the controls and time fixed effects.  Those
    regressions are estimated in the sample of control $(g,t)$s:  $(g,t)$s such that group $g$'s treatment
    has not changed yet at period $t$.  Those regressions are also estimated separately for each value
    of the baseline treatment.  If the treatment takes values 0, 1, 2, 3, and 4, one regression is
    estimated for control $(g,t)$s with a treatment equal to 0, one regression is estimated for control
    (g,t)s with a treatment equal to 1, etc.  The number of control variables needs to be
    significantly smaller than the number of control $(g,t)$s in each of those regressions.  Otherwise,
    those regressions will overfit and produce noisy estimates.  If the number of observations is
    lower than the number of variables in one of those regressions, the command will run but will not
    take into account all the controls for all values of the baseline treatment.  An error message
    will let the user know that they are encountering this situation, and may thus want to reduce
    their number of control variables.

> :question: *In my application, groups' baseline treatment is a continuous variable, meaning that all groups
    have a different period-one treatment.  Therefore, Assumption 1 in de Chaisemartin and
    D'Haultfoeuille (2020a) fails. Can I still use the command?*

 Yes you can estimate non-normalized event-study effects.  Essentially, you just need to define a
    new treatment variable equal to 0 if g's treatment has never changed at t, to 1 if g's treatment
    has changed at t and g's period-t treatment is larger than its baseline treatment, and to -1 if
    g's treatment has changed at t and g's period-t treatment is lower than its baseline treatment.
    Then, you run the command with this new treatment, including interactions of period fixed effects
    and a polynomial in the baseline treatment as control variables.  For instance, if one wants to
    model the relationship between the counterfactual outcome trend and the baseline treatment as
    quadratic, and the data has 12 periods, one needs to include 22 variables as controls: the
    baseline treatment interacted with the period 2 to 12 fixed effects, and the baseline treatment
    squared interacted with the period 2 to 12 fixed effects.  See Section 1.5 of de Chaisemartin et
    al (2022) for further details. If groups' baseline treatment is not continuous but takes many
    values, pursuing this strategy may yield more precise estimators, applying to a larger number of
    switchers, than just running the command with the original treatment, at the expense of incurring
    a potential bias if the model for the counterfactual outcome trend is misspecified.

> :question: *My design is such that treatment is binary, and groups can enter the treatment, and then leave it
    once.  Can I use the command to separately estimate the effect of joining and leaving the
    treatment?*

 Yes you can. See Section 1.5 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2024)
    for further details.

> :question: *My design has several treatments that may all have dynamic effects.  Can I use the command to
    estimate the effect of a treatment controlling for other treatments?*

 Yes, if those treatments follow binary and staggered designs.  See Section 3.2 of the Web
    Appendix of de Chaisemartin and D'Haultfoeuille (2023) for further details.

> :question: *Can I perform triple difference-in-differences with the command?*

Yes. Suppose for instance your third difference is across men and women in the same $(g,t)$ cell.
    Then, for each $(g,t)$ cell, you just need to compute the difference between the average outcome of
    men and women in cell $(g,t)$.  Then, you simply run the command with this new outcome.

> :question: *Is it possible to compute the percentage increase in the outcome of the treated relative to its
    counterfactual?*

Yes. The command already outputs the average total treatment effect, that is, the numerator in
    the percentage increase.  To compute the denominator, you need to define a new outcome variable
    $\tilde{Y} = - Y * 1\{t < F_g\}$, where F_g is the first date at which $g$'s treatment has changed.
    Essentially, you replace the outcome by 0 after the treatment change, and by -Y before the
    treatment change.  Finally, you can just compute un-normalized event-study estimators with
    $\tilde{Y}$ as the outcome.

> :question: *Instead of using an F-test to jointly test that all placebos or all effects
    are zero, I would like to use a sup t-test. Is this possible?*

 Yes, did_multiplegt_dyn is compatible with the sotable package. Here's how
    to produce sup t-tests on all placebos and effects in post estimation:

        net get did_multiplegt_dyn
        use favara_imbs_did_multiplegt_dyn.dta, clear
        did_multiplegt_dyn Dl_vloans_b county year inter_bra, effects(8)
            placebo(3) cluster(state_n) graph_off
        sotable, pnames(`=e(placebo)') normal
        sotable, pnames(`=e(effects)') normal


## References

de Chaisemartin, C, D'Haultfoeuille, X (2024).  [Difference-in-Differences Estimators of
Intertemporal Treatment Effects](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3731856).
    
de Chaisemartin, C, D'Haultfoeuille, X (2020b).  [Two-way fixed effects regressions with several
treatments](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3751060).

de Chaisemartin, C, D'Haultfoeuille, X, Pasquier, F, Vazquez-Bare, G (2022).
[Difference-in-Differences Estimators for Treatments Continuously Distributed at Every Period](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4011782).
    
de Chaisemartin, C, Ciccia, D, D'Haultfoeuille, X, Knau, F, Malézieux, M, Sow, D (2024). [Estimators and Variance Estimators Computed by the did_multiplegt_dyn Command](https://drive.google.com/file/d/1NGgScujLCCS4RrwdN-PC1SnVigfBa32h/view).

## Authors

Clément de Chaisemartin, Economics Department, Sciences Po, France.  
Xavier D'Haultfoeuille, CREST-ENSAE, France.  
Diego Ciccia, Economics Department, Sciences Po, France.  
Felix Knau, Economics Department, Sciences Po, France.  
Mélitine Malézieux, Economics Department, Sciences Po, France.  
Doulo Sow, CREST-ENSAE, France.  

**<ins>Contact:</ins>**  
[chaisemartin.packages@gmail.com](mailto:chaisemartin.packages@gmail.com)






