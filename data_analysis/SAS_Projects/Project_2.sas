%LET seed=160830;

DATA prostate;
    INFILE '/folders/myfolders/Prostate.dat';
    INPUT ID PSA_l Cancer_v Weight Age Benign_ph Seminal_vi Capsular_p Gleason_s;

DATA cardio_all;
    INFILE '/folders/myfolders/Cardiodata.csv' DSD FIRSTOBS=2;
    INPUT age bmi waisthip smok choles trig hdl ldl sys dia Uric sex alco apoa;

PROC SQL;
    CREATE TABLE cardio AS SELECT uric, dia, hdl, choles, trig, alco FROM cardio_all;

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
    ODS GRAPHICS ON / WIDTH=3in HEIGHT=3in;
    PROC SGSCATTER DATA=&datatable;
       COMPARE Y=&predicted X=(&predictors) / GROUP=indicator;
    RUN;
    
    TITLE2;
%MEND outliers;


%MACRO diagnostics(datatable, predictors, predicted, n);
    ODS EXCLUDE ALL;
    PROC REG DATA=&datatable;
       MODEL &predicted=&predictors / INFLUENCE R;
       ODS OUTPUT outputstatistics=results;
    RUN;
    ODS EXCLUDE NONE;
    
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

    TITLE2 'Diagnostics: flagged observations';
    PROC PRINT DATA=results NOOBS;
        WHERE hilev=1 OR dfflag=1 OR Fpercent>20 OR flag=1;
        VAR Observation DepVar CooksD HatDiagonal DFFITS Fpercent;
    RUN;
    
    TITLE2;
%MEND diagnostics;


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
    ODS EXCLUDE ALL;
    %DO j=1 %TO %SYSFUNC(COUNTW(&predictors));
        %LET var = %SCAN(&predictors, &j);
        PROC UNIVARIATE DATA=&datatable;
            VAR &var;
            OUTPUT OUT=quantiles_&j P25=q1 P50=q2 P75=q3;
        RUN;
    %END;
    ODS EXCLUDE NONE;
    
    DATA quantiles_h;
        SET %DO j=1 %TO %SYSFUNC(COUNTW(&predictors));quantiles_&j %END;;
    
    PROC TRANSPOSE DATA=quantiles_h OUT=quantiles;

    /* give names to columns */
    DATA quantiles; 
        SET quantiles;
        %DO j=1 %TO %SYSFUNC(COUNTW(&predictors));
            %LET var = %SCAN(&predictors, &j);
            &var = COL&j;
        %END;
        
   /* remove unnamed columns and vertical stack quantiles and data */
    DATA regdata;   
        SET &datatable quantiles (DROP=COL1 -- COL%SYSFUNC(COUNTW(&predictors)));;
    
    ODS GRAPHICS / RESET=all;
    PROC REG DATA=regdata OUTEST=est ALPHA=.05;
        ODS SELECT WHERE=(_name_ = 'ANOVA' | _name_ = 'ParameterEstimates' | _name_ = 'FitStatistics' | _name_ = 'FitPlot');
        MODEL &predicted=&predictors / CLB;
        OUTPUT OUT=outdata PREDICTED=y_hat LCLM=lower UCLM=upper;
    RUN;
    
    TITLE2 "Intervals for mean at quantile points";
    PROC PRINT DATA=outdata NOOBS;
        WHERE &predicted IS NULL;
        VAR _name_ _label_ &predictors y_hat lower upper;
    RUN;
    
    PROC SQL;
        DROP TABLE outdata;
    
    TITLE;
%MEND fit;


%MACRO test_transformations(datatable, predictors, predicted);

    %LET labels=ident sqrt recip square log;
    
    PROC SQL;
        ALTER TABLE &datatable ADD 
            &predictors._ident NUMERIC,
            &predictors._sqrt NUMERIC,
            &predictors._recip NUMERIC,
            &predictors._square NUMERIC,
            &predictors._log NUMERIC;
        UPDATE &datatable SET
            &predictors._ident=&predictors,
            &predictors._sqrt=sqrt(&predictors),
            &predictors._recip=1/&predictors,
            &predictors._square=&predictors * &predictors,
            &predictors._log=log(&predictors);
           

    %DO i=1 %TO %SYSFUNC(COUNTW(&labels));
        %LET label = %SCAN(&labels, &i);
        PROC SQL;
            DROP TABLE transdata;
            
        TITLE "BoxCox(y)=&label(x) transformed regression assumption checks";
        PROC TRANSREG DATA=&datatable;
            MODEL BoxCox(&predicted)=identity(&predictors._&label);
            OUTPUT OUT=transdata;
        RUN;
        %assumption_normality(transdata, &predictors._&label, T&predicted);
        %assumption_homoscedasticity(transdata, &predictors._&label, T&predicted);
        %assumption_linear_indep(transdata, &predictors._&label, T&predicted);
        RUN;
    %END;
    RUN;
%MEND test_transformations;


%MACRO ellipse(datatable, predictors, predicted, stepsize);
    PROC SQL;
        DROP TABLE xdata, est;
    
    ODS EXCLUDE ALL;
    PROC MEANS DATA=&datatable;
        VAR &predictors;
        OUTPUT OUT=xdata N=n MEAN=xbar CSS=Sxx;
    RUN;
    
    PROC REG DATA=&datatable OUTEST=est;
        MODEL &predicted=&predictors; 
    RUN;
    ODS EXCLUDE NONE;
 
    DATA est; SET est;
        s = _rmse_;             /* root MSE = estimated standard deviation */
        b0 = intercept;         /* estimated intercept b0 */
        b1 = &predictors;       /* estimated slope b1 */
        KEEP s b0 b1;
    RUN;
    
    DATA ellipse; 
        MERGE xdata est;
        sb1 = s/SQRT(Sxx);             /* standard deviation of b1 */
        sb0 = s*SQRT(1/n+xbar**2/Sxx); /* standard deviation of b0 */
        
        F95=finv(0.95,2,n-2)*2*s**2;   /* 95% upper bound for the quadratic form */
        
        DO beta0=b0-3*sb0 BY &stepsize TO b0+3*sb0; /* for a fixed value of beta0, solve for beta1 using the quadratic form of ellipse */ 
            D = (n*xbar*(beta0-b0))**2 - (n*xbar**2+Sxx)*(n*(beta0-b0)**2-F95); /*discriminant */
            if D < 0 then do; upperbeta1 = .; lowerbeta1 = .; end; /* discard beta0 values with D<0 */
            else do; upperbeta1 = b1+(n*xbar*(b0-beta0)+sqrt(D))/(n*xbar**2+Sxx);
            lowerbeta1 = b1+(n*xbar*(b0-beta0)-sqrt(D))/(n*xbar**2+Sxx); end;
        OUTPUT;
        END;
    RUN;
    
    PROC PLOT DATA=ellipse; TITLE 'Confidence region';
        PLOT (lowerbeta1 upperbeta1)*beta0 = '*' / overlay;
    RUN;
    
    TITLE;
%MEND ellipse;




%MACRO problem1();
    %LET datatable = prostate;
    %LET predicted = PSA_l;
    
    TITLE "1. Visual comparison against &predicted";
    ODS GRAPHICS ON / WIDTH=8in HEIGHT=3in;
    PROC SGSCATTER DATA=&datatable;
      COMPARE Y=&predicted X=(Cancer_v Weight Age Benign_ph Capsular_p);
    RUN;
    ODS GRAPHICS / RESET=all;
    
    TITLE "1. &datatable correlation matrix";
    ODS GRAPHICS ON;
    ODS SELECT WHERE=(_label_ ? 'Pearson');
    PROC CORR DATA=&datatable;
        VAR PSA_l Cancer_v Weight Age Benign_ph Seminal_vi Capsular_p Gleason_s;
    RUN;
    ODS GRAPHICS OFF;
    
    /* correlation matrix and plots indicate that Cancer_v is the best predictor */
    %LET predictors=Cancer_v;
    
    /* Rule of thumb is 2, could also use tinv for bonferroni, but would be far more restrictive */
    TITLE "1.a outliers";
    %outliers(&datatable, &predictors, &predicted, 2);
    TITLE "1.a regression diagnostics";
    %diagnostics(&datatable, &predictors, &predicted, n=97);
    
    TITLE "1.a test assumptions";
    %assumption_homoscedasticity(&datatable, &predictors, &predicted);
    %assumption_normality(&datatable, &predictors, &predicted);
    %assumption_linear_indep(&datatable, &predictors, &predicted);
    
    TITLE "1.a regression fit &predicted vs &predictors";
    %fit(&datatable, &predictors, &predicted);
    
    PROC TRANSREG DATA=&datatable;
        MODEL log(&predicted)=identity(&predictors);
        OUTPUT OUT=transdata RESIDUALS;
    RUN;
    TITLE "1.a transformed outliers";
    %outliers(transdata, &predictors, T&predicted, 2);
    
    %assumption_normality(transdata, &predictors, T&predicted);
    %assumption_homoscedasticity(transdata, &predictors, T&predicted);
    
    TITLE "1.a transformed regression assumption checks";
    %assumption_linear_indep(transdata, &predictors, T&predicted);
    
    TITLE "1.a regression fit &predicted vs &predictors";
    %fit(transdata, &predictors, T&predicted);
    
    
/*     PART B */
    %LET predictors = Cancer_v Capsular_p;
    TITLE "1.b outliers";
    %outliers(&datatable, &predictors, &predicted, 2);
    
    TITLE "1.b test assumptions";
    %assumption_homoscedasticity(&datatable, &predictors, &predicted);
    %assumption_normality(&datatable, &predictors, &predicted);
    %assumption_linear_indep(&datatable, &predictors, &predicted);
    
    TITLE "1.b regression fit &predicted vs &predictors";
    %fit(&datatable, &predictors, &predicted);
    
    PROC TRANSREG DATA=&datatable;
        MODEL log(&predicted)=identity(&predictors);
        OUTPUT OUT=transdata RESIDUALS;
    RUN;
    TITLE "1.a transformed outliers";
    %outliers(transdata, &predictors, T&predicted, 2);
    
    
    TITLE "1.b transformed regression assumption checks";
    %assumption_homoscedasticity(transdata, &predictors, T&predicted);
    %assumption_normality(transdata, &predictors, T&predicted);
    %assumption_linear_indep(transdata, &predictors, T&predicted);
    
    TITLE "1.b regression fit T&predicted vs &predictors";
    %fit(transdata, &predictors, T&predicted);
%MEND problem1;


%MACRO problem2();
    %LET basetable = cardio;
    %LET predicted = uric;
    
    TITLE "2. Visual comparison against &predicted";
    ODS GRAPHICS ON / WIDTH=8in HEIGHT=3in;
    PROC SGSCATTER DATA=&basetable;
      COMPARE Y=&predicted X=(dia hdl choles trig alco);
    RUN;
    ODS GRAPHICS / RESET=all;
    
    TITLE "2. &basetable correlation matrix";
    ODS GRAPHICS ON;
    ODS SELECT WHERE=(_label_ ? 'Pearson');
    PROC CORR DATA=&basetable; 
        VAR uric dia hdl choles trig alco;
    RUN;
    ODS GRAPHICS OFF;
    
/*  correlation matrix and plots indicate that trig is the best predictor */
    %LET predictor=trig;
    
    %MACRO analysis(part);
        
        TITLE "&part test assumptions";
        %assumption_homoscedasticity(&basetable, &predictor, &predicted);
        %assumption_normality(&basetable, &predictor, &predicted);
        %assumption_linear_indep(&basetable, &predictor, &predicted);
        
        TITLE "&part regression fit &predicted vs &predictor";
        %fit(&basetable, &predictor, &predicted);
        %ellipse(&basetable, &predictor, &predicted, 0.5);
        
        %test_transformations(&basetable, &predictor, &predicted);
        %fit(transdata, &predictor._log, T&predicted);
        
        TITLE "&part regression fit &predicted vs &predictor";
        %ellipse(transdata, &predictor._log, T&predicted, 0.0005);
    %MEND analysis;
    
    TITLE "2.a outliers";
    %outliers(&basetable, &predictor, &predicted, 3.4);
    %analysis('2.a');
    
    TITLE '2.b Observations removed';
    PROC PRINT DATA=&basetable;
        WHERE indicator='Outlier';
    PROC SQL;
        DELETE FROM &basetable WHERE indicator='Outlier';
    
    TITLE "2.b outliers";
    %outliers(&basetable, &predictor, &predicted, 2);

    %analysis('2.b');

%MEND problem2;

%problem1();
%problem2();
