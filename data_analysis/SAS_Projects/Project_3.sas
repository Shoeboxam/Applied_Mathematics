%LET seed = 160830;

DATA prostate;
    INFILE '/folders/myfolders/Prostate.dat';
    INPUT ID PSAL CV WGT AGE BPH SV CP GS;

DATA models;
    INFILE DATALINES DLM=',';
    INPUT variables $40.;
    DATALINES;
    CV
    CV SV
    CV SV GS_8
    CV BPH SV GS_8
    CV AGE BPH SV GS_8
    CV AGE BPH SV CP GS_8
    CV AGE BPH SV CP GS_7 GS_8
    CV WGT AGE BPH SV CP GS_7 GS_8
    ;

DATA gpa;
    INFILE '/folders/myfolders/gpa.csv' DLM=',' FIRSTOBS=2;
    INPUT gpa act;

%MACRO fit_stepwise(datatable, predictors, predicted);
    
    ODS EXCLUDE ALL;   
    PROC RSQUARE DATA=&datatable;
        MODEL &predicted=&predictors / CP ADJRSQ SSE MSE;
        ODS OUTPUT SubsetSelSummary=SubsetSummary;
    RUN;
    ODS EXCLUDE NONE;
    
    PROC PRINT DATA=SubsetSummary;
    RUN;
    
    PROC SQL;
        CREATE TABLE SubsetSummaryStats AS SELECT NumInModel, MAX(RSquare) AS RSquareMax, MAX(AdjRSq) AS AdjRSqMax, MIN(MSE) AS MSEMin, MIN(Cp) AS CpMin FROM SubsetSummary GROUP BY NumInModel;
    
    PROC SGPLOT DATA=SubsetSummaryStats;
        SERIES X=NumInModel Y=RSquareMax;
    RUN;
    
    PROC SGPLOT DATA=SubsetSummaryStats;
        SERIES X=NumInModel Y=AdjRSqMax;
    RUN;
    
    PROC SGPLOT DATA=SubsetSummaryStats;
        SERIES X=NumInModel Y=CpMin;
    RUN;
    
    PROC SGPLOT DATA=SubsetSummaryStats;
        SERIES X=NumInModel Y=MSEMin;
    RUN;
    
    PROC SGPLOT DATA=SubsetSummary;
        SCATTER X=NumInModel Y=RSquare;
    RUN;
    
    PROC SQL;
        CREATE TABLE models AS 
            SELECT d1.NumInModel, d1.VarsInModel FROM SubsetSummary AS d1 
                LEFT OUTER JOIN SubsetSummary AS d2 ON (d1.NumInModel = d2.NumInModel and d1.Cp < d2.Cp) 
                WHERE d2.NumInModel IS NULL GROUP BY d1.NumInModel;
    
    PROC PRINT DATA=models;
    RUN;
    
%MEND fit_stepwise;

%MACRO test_influential(datatable, predictors, predicted);
    ODS EXCLUDE ALL;
    PROC REG DATA=&datatable;
       MODEL &predicted=&predictors / INFLUENCE R;
       ODS OUTPUT outputstatistics=results;
    RUN;
    ODS EXCLUDE NONE;
    
    /* Just to count the number of rows */
    PROC IML;
        USE &datatable;
        READ ALL VAR{&predicted}; 
        CALL SYMPUTX("n", NROW(&predicted));
   
    DATA results; set results;
        %LET p = %LENGTH(&predictors);
        
        hilev = HatDiagonal > 2*(&p/&n);
        dfflag = ABS(DFFITS) > 1;
        Fpercent = 100*probf(CooksD, &p, &n - &p); /* calculate percentile for each Cook's D value using F(p, n-p) dist*/

        /* check DFB for each parameter */
        %DO i=1 %TO %SYSFUNC(COUNTW(&predictors));
            %LET var = %SCAN(&predictors, &i);
            &var._flag = ABS(DFB_&var > 1);
            IF &var._flag THEN flag = 1;
        %END;
    RUN;
    
    %LET dfbetas_flags = ;
    %DO i=1 %TO %SYSFUNC(COUNTW(&predictors));
        %LET var = %SCAN(&predictors, &i);
        %LET dfbetas_flags = &dfbetas_flags DFB_&var;
    %END;
    
    TITLE2 'flagged observations';
    PROC PRINT DATA=results NOOBS;
        WHERE hilev=1 OR dfflag=1 OR Fpercent>20 OR flag=1;
        VAR Observation DepVar CooksD HatDiagonal DFFITS Fpercent &dfbetas_flags;
    RUN;
    
    TITLE2;
%MEND test_influential;

%MACRO test_collinearity(datatable, predictors, predicted);
    PROC REG DATA=&datatable;
        ODS SELECT WHERE=(_name_ = 'CollinDiag' | _name_ = 'ParameterEstimates');
        MODEL &predicted = &predictors / VIF TOL COLLIN;
    
    /* multiple correlation coefficients */
/*     %DO i=1 %TO %SYSFUNC(COUNTW(&predictors, ' ')); */
/*         %LET label = %SCAN(&predictors, &i, ' '); */
/*  */
/*         %LET otherParams = %SYSFUNC(TRANWRD(&predictors, &label, %STR( ))); */
/*         PROC REG DATA=&datatable; */
/*             ODS SELECT WHERE=(_name_ = 'CollinDiag' | _name_ = 'ParameterEstimates'); */
/*             MODEL &label = &otherParams / VIF TOL COLLIN; */
/*         RUN; */
/*     %END; */

%MEND test_collinearity;


%MACRO outliers(datatable, predictors, predicted, cutoff);
    PROC SQL;
        ALTER TABLE &datatable DROP student, residuals, indicator;

    ODS EXCLUDE ALL;
    PROC REG DATA=&datatable
        PLOTS(label)=RStudentByLeverage;
/*      ODS SELECT WHERE=(_name_ ? 'RStudentByLeverage'); */
        MODEL &predicted=&predictors;
        OUTPUT OUT=&datatable STUDENT=student;
    RUN;
    ODS EXCLUDE NONE;
    
    /* add an indicator variable for outliers */
    PROC SQL;
        ALTER TABLE &datatable ADD indicator VARCHAR(10);
        UPDATE &datatable SET indicator = CASE WHEN ABS(student) > &cutoff THEN 'Outlier' ELSE 'Included' END;
    RUN;
    
    TITLE2 "Highlighted outliers";
    ODS GRAPHICS ON / WIDTH=8in HEIGHT=3in;
    PROC SGSCATTER DATA=&datatable;
       COMPARE Y=&predicted X=(&predictors) / GROUP=indicator;
    RUN;
    ODS GRAPHICS ON / WIDTH=3in HEIGHT=3in;    
    TITLE2;
%MEND outliers;



%MACRO assumption_homoscedasticity(datatable, predictors, predicted);

    PROC SQL;
        ALTER TABLE &datatable DROP rstudent, residual;
    
    ODS EXCLUDE ALL;
    PROC REG DATA=&datatable;
        MODEL &predicted=&predictors;
/*         ODS SELECT WHERE=(_name_ = 'ANOVA'); */
        OUTPUT OUT=&datatable RSTUDENT=rstudent RESIDUAL=residual;
    RUN;
    
    PROC SQL;
        ALTER TABLE &datatable ADD residual_square NUMERIC;
        UPDATE &datatable SET residual_square=residual*residual;
    
    PROC REG DATA=&datatable OUTEST=outest;
        MODEL residual_square=&predictors / RSQUARE;
    RUN;
    ODS EXCLUDE NONE;
    
    PROC IML; 
        USE &datatable; 
        READ ALL VAR{&predicted}; 
        CALL SYMPUTX("n", NROW(&predicted));

    RUN;
    
    TITLE2 "Homoscedasticity of &predicted Breusch-Pagan via Lagrange Multipliers";
    PROC IML;
        USE outest;
        READ ALL VAR {_RSQ_};
        bp_test = &n * _RSQ_;
        PRINT bp_test;
        bp_p_value = 1 - CDF("chisquare", bp_test, %SYSFUNC(COUNTW(&predictors)));
        PRINT bp_p_value;
        QUIT;

    
    %DO j=1 %TO %SYSFUNC(COUNTW(&predictors));
        %LET variable = %SCAN(&predictors, &j);
        PROC IML;
            USE &datatable;
            READ ALL VAR {&variable};
            CALL SYMPUTX("median", MEDIAN(&variable));
            QUIT;
        
        DATA &datatable; SET &datatable; 
            group = &variable > &median;
        RUN;
        
        TITLE2 "Homoscedasticity of &predicted vs &variable Brown-Forsythe";
        PROC GLM DATA=&datatable;
            ODS SELECT WHERE=(_name_ ? 'HOVFTest');
            CLASS group; MODEL rstudent=group; MEANS group / HOVTEST=BF;
        RUN;
    %END;
    
    PROC SQL;
        ALTER TABLE &datatable DROP rstudent, residual, residual_square, group;
    TITLE2;
%MEND assumption_homoscedasticity;


%MACRO assumption_normality(datatable, predictors, predicted);
    TITLE2 "Normality of &predicted";
    
    ODS EXCLUDE ALL;
    PROC REG DATA=&datatable;
        MODEL &predicted=&predictors;
        OUTPUT OUT=&datatable RESIDUAL=residual;
    RUN;
    ODS EXCLUDE NONE;

    PROC UNIVARIATE DATA=&datatable NORMAL;
        ODS SELECT WHERE=(_name_ = 'TestsForNormality');
        VAR residual;
    RUN;
    
    ODS GRAPHICS ON / WIDTH=3in HEIGHT=3in;
    PROC REG DATA=&datatable
        PLOTS(label)=QQPlot;
        ODS SELECT WHERE=(_name_ ? 'QQPlot');
        MODEL &predicted=&predictors;
    RUN;
    
    PROC SQL;
        ALTER TABLE &datatable DROP residual;
    ODS GRAPHICS OFF;
    TITLE2;
%MEND assumption_normality;


%MACRO assumption_linear_indep(datatable, predictors, predicted);
    
    PROC SQL;
        ALTER TABLE &datatable DROP residual;
    RUN;

    ODS GRAPHICS ON / WIDTH=3in HEIGHT=3in;
    PROC REG DATA=&datatable;
        MODEL &predicted=&predictors;
        OUTPUT OUT=&datatable RESIDUAL=residual;
        ODS SELECT WHERE=(_name_ = 'ResidualPlot');
    RUN;
    TITLE2;
%MEND assumption_linear_indep;



%MACRO fit(datatable, predictors, predicted);
    
    TITLE "Fit for &predicted vs &predictors";
    ODS GRAPHICS ON;
    PROC REG DATA=&datatable;
        ODS SELECT WHERE=(_name_ = 'ANOVA' | _name_ = 'ParameterEstimates' | _name_ = 'FitStatistics' | _name_ = 'FitPlot');
        MODEL &predicted=&predictors / CLB;
    RUN;
    
    
%MEND fit;




%MACRO problem_1();
    
    PROC SQL;
        ALTER TABLE prostate DROP ID;
        
        ALTER TABLE prostate ADD GS_7 NUMERIC, GS_8 NUMERIC;
        UPDATE prostate SET GS_7= CASE WHEN GS=7 THEN 1 ELSE 0 END;
        UPDATE prostate SET GS_8= CASE WHEN GS=8 THEN 1 ELSE 0 END;
        
        ALTER TABLE prostate DROP GS;

    %LET predictors=CV WGT AGE BPH SV CP GS_7 GS_8;
    
    %fit_stepwise(prostate, &predictors, PSAL);
    
    PROC PRINT DATA=stepwise_perf;
    RUN;
    
    PROC IML;
        USE models;
        READ ALL VAR{NumInModel}; 
        CALL SYMPUTX("numModels", NROW(variables));
    
    %DO j=1 %TO &numModels;
        PROC IML;
            USE models;
            READ ALL VAR{NumInModel};
            
            CALL SYMPUTX("modelVars", NumInModel[&j]);
        
        TITLE "Diagnostics: &modelVars";
        %test_influential(prostate, &modelVars, PSAL);
        
        TITLE "Collinearity: &modelVars";
        %test_collinearity(prostate, &modelVars, PSAL);
    %END;
    
    %LET best_predictors = CV SV GS_8;
    TITLE "Homoscedasticity checks for &best_predictors";
    %assumption_homoscedasticity(prostate, &best_predictors, PSAL);
    TITLE "Normality checks for &best_predictors";
    %assumption_normality(prostate, &best_predictors, PSAL);
    TITLE "Linearity/independence checks for &best_predictors";
    %assumption_linear_indep(prostate, &best_predictors, PSAL);
    
    TITLE "Outliers";
    %LET alpha = .05;
    PROC IML;
        USE prostate;
        READ ALL VARS{PSAL};
        CALL SYMPUTX('tvalue', TINV(1 - &alpha/(2 * 4), NROW(PSAL)));

    %outliers(prostate, &best_predictors, PSAL, &tvalue);
    PROC PRINT DATA=prostate;
        WHERE indicator='Outlier';
    
    RUN;
    %fit(prostate, &best_predictors, PSAL);
    
    TITLE "BoxCox(y)=&label(x) transformed regression assumption checks";
    PROC TRANSREG DATA=prostate;
        MODEL BoxCox(PSAL)=identity(&best_predictors);
        OUTPUT OUT=transdata;
    RUN;
    
    TITLE "Homoscedasticity checks for &best_predictors after transformation";
    %assumption_homoscedasticity(transdata, &best_predictors, TPSAL);
    TITLE "Normality checks for &best_predictors after transformation";
    %assumption_normality(transdata, &best_predictors, TPSAL);
    
    TITLE "Diagnostics: &best_predictors";
    %test_influential(transdata, &best_predictors, TPSAL);
    
    TITLE "Collinearity: &best_predictors";
    %test_collinearity(transdata, &best_predictors, TPSAL);
    %fit(transdata, &best_predictors, TPSAL);
%MEND problem_1;


%MACRO problem_2();
    
    TITLE "scatter plot of GPA vs ACT";
    PROC SGPLOT DATA=gpa;
        SCATTER Y=act X=gpa;
    
    PROC CORR DATA=gpa OUTP=correlation_constant NOPRINT;
        VAR gpa act;
    RUN;
    
    PROC IML;
        USE correlation_constant;
        READ ALL VAR {gpa} WHERE(_name_='act');
        CALL SYMPUTX('correlation_constant', gpa);
    
    %LET numSamples = 1000;
    
    PROC SURVEYSELECT DATA=gpa SEED=&seed NOPRINT
        OUT=bootstrapSample(rename=(Replicate=SampleID))
        METHOD=urs /* resampling with replacement */
        SAMPRATE=1 /* each bootstrap resample has N observations */
        REPS=&numSamples;
        
    
    PROC CORR DATA=bootstrapSample OUTP=correlations_with_fluff NOPRINT;
        VAR gpa act;
        BY SampleId;
        FREQ NumberHits;
    
    PROC SQL;
        CREATE TABLE correlations AS SELECT gpa AS values FROM correlations_with_fluff WHERE _name_ EQ 'act';
    
    TITLE "histogram of (nonparametric) bootstrap distribution of the point estimate";
    PROC SGPLOT DATA=correlations; /* histogram of bootstrap distribution */
        HISTOGRAM values;
    RUN;

    PROC MEANS DATA=correlations NOPRINT;
        VAR values;
        OUTPUT OUT=correlation_stats MEAN=mean STD=std;
    RUN;
    
    TITLE "constant POINT ESTIMATE of ρ";
    PROC IML;
        PRINT(&correlation_constant);
    RUN;
    
    TITLE "bootstrap POINT ESTIMATE and STANDARD ERROR of ρ";
    PROC PRINT DATA=correlation_stats NOOBS;    
        VAR mean std;
    RUN;
    
    TITLE "Bootstrap estimate of bias";
    PROC IML;
        USE correlation_stats;
        READ ALL VAR {mean};
        bias = mean - &correlation_constant;
        PRINT bias;
        CALL SYMPUTX('bias', bias);

    TITLE;    
    DATA biases;
        SET correlations;
        bias = values - &correlation_constant;

    PROC UNIVARIATE DATA=correlations NOPRINT;
        VAR values;
        OUTPUT OUT=correlation_confidences PCTLPRE=CI95_
        PCTLPTS=2.5 97.5 /* compute 2.5th & 97.5th percentiles of sampling distribution of median */
        PCTLNAME=Lower Upper;
    RUN;

    PROC UNIVARIATE DATA=biases NOPRINT;
        VAR bias;
        OUTPUT OUT=bias_confidences PCTLPRE=CI95_
        PCTLPTS=2.5 97.5 /* compute 2.5th & 97.5th percentiles of sampling distribution of bias */
        PCTLNAME=Lower Upper;
    RUN;
    
    /* normal approximation */
    DATA normal;
        SET correlation_stats;
        z1 = QUANTILE('NORMAL', .975);
        z2 = QUANTILE('NORMAL', .025);
        lower_normal = (&correlation_constant - &bias) - z1 * std;
        upper_normal = (&correlation_constant - &bias) - z2 * std;
    RUN;
    TITLE "Normal 95% confidence interval for correlation";
    PROC PRINT DATA=normal NOOBS;
        VAR lower_normal upper_normal;
        
    /* basic bootstrap */
    DATA basic;
        SET correlation_confidences;
        lower_basic = 2 * &correlation_constant - CI95_Upper;
        upper_basic = 2 * &correlation_constant - CI95_Lower;
    RUN;
    TITLE "Basic 95% confidence interval for correlation";
    PROC PRINT DATA=basic NOOBS;
        VAR lower_basic upper_basic;

    /* percentile bootstrap */
    DATA percentile;
        SET bias_confidences;
        lower_percentile = CI95_Lower;
        upper_percentile = CI95_Upper;
    RUN;
    TITLE "Percentile 95% confidence interval for bias";
    PROC PRINT DATA=percentile NOOBS;
        VAR lower_percentile upper_percentile;
    
    RUN;
    TITLE;

%MEND problem_2;


%problem_1();
/* %problem_2(); */